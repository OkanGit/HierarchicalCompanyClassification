# hierarchical_gics_transformers.py
"""
Hierarchical GICS classification experiments using transformer-based document embeddings.

"""

import os
from collections import defaultdict
from typing import List, Dict

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

# sklearn pieces kept the same
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

# transformers + torch for embeddings
import torch
from transformers import AutoTokenizer, AutoModel

# Optional MLflow import
try:
    import mlflow

    MLFLOW_AVAILABLE = True
except Exception:
    mlflow = None
    MLFLOW_AVAILABLE = False


# ------------------------------- Utilities ---------------------------------


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def full_path_label(row: pd.Series, levels: List[str]) -> str:
    """Concatenate hierarchical labels into a single path label string (all cast to str)."""
    return "|".join([str(row[l]) for l in levels])


def build_tree_from_csv(gics_csv_path: str, levels: List[str], sep=";") -> Dict:
    """Return a tree mapping parent->children and child->parent mapping."""
    df = pd.read_csv(gics_csv_path, sep=sep, dtype=str)
    parents = {}
    children = defaultdict(set)
    root = "ROOT"
    for _, r in df.iterrows():
        prev = root
        for lvl in levels:
            node = str(r[lvl])
            children[prev].add(node)
            parents[node] = prev
            prev = node
    children = {k: sorted(list(v)) for k, v in children.items()}
    return {"children": children, "parent": parents, "root": root}


def shared_prefix_depth(true_path: List[str], pred_path: List[str]) -> int:
    depth = 0
    for a, b in zip(true_path, pred_path):
        if a == b:
            depth += 1
        else:
            break
    return depth


def hierarchical_metrics(
    y_true_paths: List[List[str]], y_pred_paths: List[List[str]], max_depth: int
) -> Dict:
    """Compute simple hierarchical precision/recall/f1 based on shared prefix depth."""
    common_list = []
    h_prec_list = []
    h_rec_list = []
    h_f1_list = []
    loss_list = []

    for t, p in zip(y_true_paths, y_pred_paths):
        common = shared_prefix_depth(t, p)
        pred_depth = len(p) if len(p) > 0 else 1
        true_depth = len(t) if len(t) > 0 else 1
        h_prec = common / pred_depth if pred_depth > 0 else 0.0
        h_rec = common / true_depth if true_depth > 0 else 0.0
        h_f1 = (2 * h_prec * h_rec / (h_prec + h_rec)) if (h_prec + h_rec) > 0 else 0.0
        loss = 1.0 - (common / max_depth) if max_depth > 0 else 1.0
        common_list.append(common)
        h_prec_list.append(h_prec)
        h_rec_list.append(h_rec)
        h_f1_list.append(h_f1)
        loss_list.append(loss)

    return {
        "h_precision_micro": float(np.mean(h_prec_list)),
        "h_recall_micro": float(np.mean(h_rec_list)),
        "h_f1_micro": float(np.mean(h_f1_list)),
        "hierarchical_loss_mean": float(np.mean(loss_list)),
        "common_depth_mean": float(np.mean(common_list)),
    }


# ---------------------------- Embedding utilities ---------------------------


def chunk_text_to_token_windows(
    tokenizer, text: str, max_length: int, stride: int, add_special_tokens: bool = True
) -> List[Dict]:
    """
    Split a single long text into a list of chunk token dicts suitable for tokenizer.encode_plus batch.

    Uses tokenizer.encode with return_overflowing_tokens is possible, but we implement explicit
    sliding windows for greater control and clear batching.
    """
    # encode without truncation to get token ids
    enc = tokenizer.encode(text, add_special_tokens=False)
    n = len(enc)
    if n == 0:
        return [{"input_ids": []}]

    windows = []
    start = 0
    while start < n:
        end = min(start + max_length, n)
        window_tokens = enc[start:end]
        windows.append({"input_ids": window_tokens})
        if end == n:
            break
        start += max(1, (max_length - stride))
    return windows


def collate_windows_batch(
    tokenizer,
    windows_batch: List[List[Dict]],
    device: torch.device,
    return_tensors="pt",
):
    """
    Given a batch of texts where each text is a list of token dicts (from chunk_text_to_token_windows),
    this function returns padded tensors with mapping metadata so model outputs can be re-assembled.

    We will convert windows_batch (list of list of dicts) into:
      - input_ids: Tensor(B_total, L)
      - attention_mask: Tensor(B_total, L)
    and an index map: list_of_counts telling how many windows belong to each original text
    """
    flat_input_ids = []
    # We will convert each window token id list into tokenizer.prepare_for_model style input ids w/ special tokens later
    for windows in windows_batch:
        for wd in windows:
            # wd is dict with 'input_ids'
            flat_input_ids.append(wd.get("input_ids", []))

    # use tokenizer.pad to pad a list of dicts; but tokenizer expects 'input_ids' lists:
    padded = tokenizer.pad(
        [{"input_ids": ids} for ids in flat_input_ids],
        padding=True,
        max_length=None,
        return_tensors=return_tensors,
    )
    # count windows per original text
    counts = [len(windows) for windows in windows_batch]
    return padded, counts


@torch.no_grad()
def compute_document_embeddings(
    texts,
    tokenizer,
    model,
    device,
    chunk_max_length=512,
    chunk_stride=256,
    embed_batch_size=8,
    cache_path=None,
    show_progress=True,
):
    """
    Compute document embeddings by chunking long texts into overlapping segments,
    averaging the resulting embeddings, and caching if desired.

    Returns:
        numpy.ndarray of shape (n_texts, hidden_size)
    """

    # If cache exists, load it (joblib used for robustness across types)
    if cache_path and os.path.exists(cache_path):
        try:
            cached = joblib.load(cache_path)
            # ensure numpy array
            return np.asarray(cached)
        except Exception:
            # fall back to recompute if cache load failed
            pass

    # make sure model is on correct device and in eval
    model.to(device)
    model.eval()

    embeddings = []
    hidden_size = getattr(model.config, "hidden_size", None)

    for text in tqdm(texts, disable=not show_progress, desc="Encoding docs"):
        # Tokenize full text without truncation to get all tokens (ids)
        tokens = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors=None,
        )

        input_ids = tokens.get("input_ids", [])
        # single string -> often a single list inside
        if (
            isinstance(input_ids, list)
            and len(input_ids) == 1
            and isinstance(input_ids[0], list)
        ):
            input_ids = input_ids[0]

        # if still not a list, coerce
        if not isinstance(input_ids, list):
            input_ids = list(input_ids)

        # Split into chunks with overlap
        chunks = []
        start = 0
        n_tokens = len(input_ids)
        if n_tokens == 0:
            # represent empty doc as empty list; will be handled below
            chunks = []
        else:
            while start < n_tokens:
                end = start + chunk_max_length
                chunk = input_ids[start:end]
                chunks.append(chunk)
                if end >= n_tokens:
                    break
                start += max(1, (chunk_max_length - chunk_stride))

        chunk_embs = []

        # If there are no chunks (empty text), create a zero vector based on model hidden size
        if len(chunks) == 0:
            if hidden_size is None:
                # fallback hidden size guess
                hidden_size = 768
            doc_emb = torch.zeros(hidden_size, device="cpu", dtype=torch.float32)
            embeddings.append(doc_emb)
            continue

        for i in range(0, len(chunks), embed_batch_size):
            batch_chunks = chunks[i : i + embed_batch_size]
            # Properly add CLS/SEP per chunk if available; fall back gracefully if tokens missing
            cls_id = (
                tokenizer.cls_token_id if tokenizer.cls_token_id is not None else None
            )
            sep_id = (
                tokenizer.sep_token_id if tokenizer.sep_token_id is not None else None
            )

            batch_input_ids = []
            for chunk in batch_chunks:
                ids = []
                if cls_id is not None:
                    ids.append(cls_id)
                ids.extend(chunk)
                if sep_id is not None:
                    ids.append(sep_id)
                batch_input_ids.append(ids)

            batch_inputs = tokenizer.pad(
                {
                    "input_ids": [
                        [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
                        for chunk in batch_chunks
                    ],
                },
                padding=True,
                return_tensors="pt",
            )

            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            outputs = model(**batch_inputs, output_hidden_states=True)
            # Use the mean pooled last hidden state as chunk embedding
            last_hidden = outputs.last_hidden_state  # [B, T, H]
            mask = batch_inputs["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
            masked = last_hidden * mask
            # prevent division by zero
            denom = mask.sum(dim=1).clamp(min=1e-9)
            chunk_emb = masked.sum(dim=1) / denom
            chunk_embs.append(chunk_emb.detach().cpu())

        # Average chunk embeddings for this doc (concatenate along 0)
        doc_emb = torch.cat(chunk_embs, dim=0).mean(dim=0)
        embeddings.append(doc_emb)

    # stack and convert to numpy
    embeddings = torch.stack(embeddings).cpu().numpy()

    # cache if requested
    if cache_path:
        try:
            joblib.dump(embeddings, cache_path)
        except Exception:
            # ignore caching failures
            pass

    return embeddings


# -------------------------- Helper: probabilistic clf ----------------------


def _build_probabilistic_clf(X_vec, y_vec, random_state):
    """Return a fitted sklearn classifier that supports predict_proba (robust fallback)."""
    classes, counts = np.unique(y_vec, return_counts=True)
    min_count = counts.min()
    n_classes = classes.size

    if n_classes == 1:
        # Always predict the single class
        dummy = DummyClassifier(strategy="constant", constant=classes[0])
        dummy.fit(X_vec, y_vec)
        return dummy
    # choose calibration or fallback
    if min_count >= 3:
        cv_folds = min(3, int(min_count))
        if cv_folds < 2:
            cv_folds = 2
        base = SGDClassifier(
            loss="log_loss", max_iter=1000, tol=1e-3, random_state=random_state
        )
        try:
            clf_cal = CalibratedClassifierCV(base, cv=cv_folds)
            clf_cal.fit(X_vec, y_vec)
            return clf_cal
        except Exception:
            # fallback to LogisticRegression if calibration fails
            lr = LogisticRegression(
                multi_class="multinomial", max_iter=2000, random_state=random_state
            )
            lr.fit(X_vec, y_vec)
            return lr
    else:
        lr = LogisticRegression(
            multi_class="multinomial", max_iter=2000, random_state=random_state
        )
        lr.fit(X_vec, y_vec)
        return lr


# --------------------------- Main runner (transformers) --------------------


def run_hierarchical_experiments_transformers(
    df: pd.DataFrame,
    text_col: str,
    target_levels: List[str],
    gics_csv_path: str,
    out_dir: str,
    model_name: str = "allenai/longformer-base-4096",
    embed_batch_size: int = 64,
    chunk_max_length: int = 256,
    chunk_stride: int = 50,
    test_size: float = 0.2,
    random_state: int = 42,
    mlflow_enabled: bool = True,
    cache_embeddings: bool = True,
    embedding_cache_name: str = "embeddings_cache.joblib",
    device: str = None,
):
    """
    Run all paradigms using transformer embeddings.

    Important params:
      - model_name: huggingface model identifier. Defaults to a small, performant sentence-transformers model.
      - chunk_max_length: tokens per chunk (including special tokens will be added). Keep <= model's max positions.
      - chunk_stride: overlap between chunks.
      - embed_batch_size: number of windows passed to model per forward pass (tune for your GPU memory).
      - cache_embeddings: reuse/save embeddings to speed repeated runs.

    Returns:
        results_summary: dict mapping paradigm -> hierarchical_metrics dict
    """
    # Basic validation
    assert text_col in df.columns, f"text_col '{text_col}' not found in dataframe"
    assert all(
        l in df.columns for l in target_levels
    ), "Some target level columns missing from dataframe"

    ensure_dir(out_dir)
    tree = build_tree_from_csv(gics_csv_path, target_levels)
    max_depth = len(target_levels)

    # prepare dataframe and full path label (cast labels to str to avoid dtype issues)
    df = df.copy()
    df = df.dropna(subset=target_levels + [text_col]).reset_index(drop=True)
    df["path_label"] = df.apply(lambda r: full_path_label(r, target_levels), axis=1)

    # train/test split (stratify by path_label if possible)
    stratify_opt = (
        df["path_label"] if df["path_label"].value_counts().min() > 1 else None
    )
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify_opt
    )

    X_train_texts = train_df[text_col].astype(str).tolist()
    X_test_texts = test_df[text_col].astype(str).tolist()
    y_train_paths = [r.split("|") for r in train_df["path_label"].astype(str)]
    y_test_paths = [r.split("|") for r in test_df["path_label"].astype(str)]

    # device selection
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # load tokenizer + model (encoder-only)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)

    # ensure model on device and eval (compute_document_embeddings also ensures this but we set here early)
    model.to(device)
    model.eval()

    # Embedding cache path
    cache_path = (
        os.path.join(out_dir, embedding_cache_name) if cache_embeddings else None
    )

    # compute embeddings for train and test
    train_cache_path = (cache_path + ".train") if cache_path else None
    test_cache_path = (cache_path + ".test") if cache_path else None

    print("➡️  Computing train embeddings (this may take a while for large corpora)...")
    X_train_emb = compute_document_embeddings(
        X_train_texts,
        tokenizer,
        model,
        device,
        chunk_max_length=chunk_max_length,
        chunk_stride=chunk_stride,
        embed_batch_size=embed_batch_size,
        cache_path=train_cache_path,
        show_progress=True,
    )
    print("➡️  Computing test embeddings...")
    X_test_emb = compute_document_embeddings(
        X_test_texts,
        tokenizer,
        model,
        device,
        chunk_max_length=chunk_max_length,
        chunk_stride=chunk_stride,
        embed_batch_size=embed_batch_size,
        cache_path=test_cache_path,
        show_progress=True,
    )

    # convenience: a function to save artifacts and log to mlflow if requested
    def _save_and_log(paradigm_name: str, model_obj, flat_report: Dict, hmetrics: Dict):
        pdir = os.path.join(out_dir, paradigm_name)
        ensure_dir(pdir)
        model_path = os.path.join(pdir, "model.joblib")
        joblib.dump(model_obj, model_path)

        rep_df = pd.DataFrame(flat_report).transpose()
        rep_df.to_csv(
            os.path.join(pdir, "classification_report.csv"),
            sep=";",
            decimal=",",
            index=True,
        )

        h_df = pd.DataFrame([hmetrics])
        h_df.to_csv(
            os.path.join(pdir, "hierarchical_metrics.csv"),
            sep=";",
            decimal=",",
            index=False,
        )

        if mlflow_enabled:
            if not MLFLOW_AVAILABLE:
                print(
                    f"[warning] mlflow_enabled=True but mlflow not available; skipping MLflow logging for {paradigm_name}"
                )
            else:
                try:
                    mlflow.set_experiment("transformers")
                    with mlflow.start_run(run_name=paradigm_name):
                        mlflow.log_param("paradigm", paradigm_name)
                        for k, v in hmetrics.items():
                            try:
                                mlflow.log_metric(k, float(v))
                            except Exception:
                                pass
                        mlflow.log_artifact(
                            os.path.join(pdir, "classification_report.csv")
                        )
                        mlflow.log_artifact(
                            os.path.join(pdir, "hierarchical_metrics.csv")
                        )
                        mlflow.log_artifact(model_path)
                except Exception as e:
                    print(f"[warning] mlflow logging failed for {paradigm_name}: {e}")

    results_summary = {}

    # ------------------------ Global Flat (single multiclass) -----------------
    paradigm_name = "global_flat"
    print(f"\n===== Running paradigm: {paradigm_name} =====")
    ensure_dir(os.path.join(out_dir, paradigm_name))

    path_le = LabelEncoder()
    path_le.fit(train_df["path_label"].astype(str))
    y_train_path_enc = path_le.transform(train_df["path_label"].astype(str))

    # train classifier on embeddings
    clf = SGDClassifier(
        loss="log_loss", max_iter=1000, tol=1e-3, random_state=random_state
    )
    clf.fit(X_train_emb, y_train_path_enc)

    # predict and decode back to path strings
    y_pred_enc = clf.predict(X_test_emb)
    y_pred_paths_flat = [str(v) for v in path_le.inverse_transform(y_pred_enc)]
    y_pred_paths = [p.split("|") for p in y_pred_paths_flat]

    rep = classification_report(
        test_df["path_label"].astype(str),
        y_pred_paths_flat,
        output_dict=True,
        zero_division=0,
    )
    hmetrics = hierarchical_metrics(y_test_paths, y_pred_paths, max_depth)
    _save_and_log(
        paradigm_name,
        {"embed_model_name": model_name, "clf": clf, "label_encoder": path_le},
        rep,
        hmetrics,
    )
    results_summary[paradigm_name] = hmetrics

    # ------------------- Global Multi-output (predict each level jointly) ---
    paradigm_name = "global_multioutput"
    print(f"\n===== Running paradigm: {paradigm_name} =====")
    ensure_dir(os.path.join(out_dir, paradigm_name))

    level_encoders: Dict[str, LabelEncoder] = {}
    y_train_enc_cols = []
    for lvl in target_levels:
        le = LabelEncoder()
        le.fit(train_df[lvl].astype(str))
        level_encoders[lvl] = le
        y_train_enc_cols.append(le.transform(train_df[lvl].astype(str)))
    y_train_enc = np.vstack(y_train_enc_cols).T  # (n_samples, n_levels)

    base = SGDClassifier(
        loss="log_loss", max_iter=1000, tol=1e-3, random_state=random_state
    )
    multi = MultiOutputClassifier(base, n_jobs=-1)
    multi.fit(X_train_emb, y_train_enc)
    y_pred_enc = multi.predict(X_test_emb)  # (n_samples, n_levels)

    # decode
    y_pred_levels = []
    for col_idx, lvl in enumerate(target_levels):
        le = level_encoders[lvl]
        y_pred_levels.append(le.inverse_transform(y_pred_enc[:, col_idx]))

    y_pred_paths = []
    for i in range(len(X_test_emb)):
        path = [
            str(y_pred_levels[level_idx][i]) for level_idx in range(len(target_levels))
        ]
        y_pred_paths.append(path)
    y_pred_path_strings = ["|".join(p) for p in y_pred_paths]

    rep = classification_report(
        test_df["path_label"].astype(str),
        y_pred_path_strings,
        output_dict=True,
        zero_division=0,
    )
    hmetrics = hierarchical_metrics(y_test_paths, y_pred_paths, max_depth)
    _save_and_log(
        paradigm_name,
        {
            "embed_model_name": model_name,
            "multi": multi,
            "level_encoders": level_encoders,
        },
        rep,
        hmetrics,
    )
    results_summary[paradigm_name] = hmetrics

    # ---------------------- Local Classifier per Node (LCN) ------------------
    paradigm_name = "local_classifier_per_node"
    print(f"\n===== Running paradigm: {paradigm_name} =====")
    ensure_dir(os.path.join(out_dir, paradigm_name))

    # Build index of samples per node (train)
    children = tree["children"]
    root = tree["root"]
    node_samples = defaultdict(list)
    for i, p in enumerate(y_train_paths):
        node_samples[root].append(i)
        for step in p:
            node_samples[step].append(i)

    node_models = {}
    for node, childs in tqdm(children.items(), desc="PerNode: training nodes"):
        if len(childs) <= 1:
            continue
        idxs = node_samples.get(node, [])
        if not idxs:
            continue
        X = X_train_emb[idxs]
        y = []
        for i in idxs:
            p = y_train_paths[i]
            if node == root:
                child = p[0] if len(p) > 0 else node
            else:
                try:
                    pos = p.index(node)
                    child = p[pos + 1] if pos + 1 < len(p) else node
                except Exception:
                    child = node
            y.append(str(child))
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        clf_node = clone(
            SGDClassifier(
                loss="log_loss", max_iter=1000, tol=1e-3, random_state=random_state
            )
        )
        clf_node.fit(X, y_enc)
        node_models[node] = {"le": le, "clf": clf_node}

    # predict per sample top-down
    y_pred_paths = []
    for vec in tqdm(X_test_emb, desc="PerNode: predicting"):
        path = []
        node = root
        for _ in range(max_depth):
            model_info = node_models.get(node)
            if model_info is None:
                break
            pred_enc = model_info["clf"].predict(vec.reshape(1, -1))[0]
            child = model_info["le"].inverse_transform([int(pred_enc)])[0]
            path.append(str(child))
            node = child
        y_pred_paths.append(path)

    y_pred_path_strings = ["|".join(p) for p in y_pred_paths]
    rep = classification_report(
        test_df["path_label"].astype(str),
        y_pred_path_strings,
        output_dict=True,
        zero_division=0,
    )
    hmetrics = hierarchical_metrics(y_test_paths, y_pred_paths, max_depth)
    _save_and_log(
        paradigm_name,
        {"embed_model_name": model_name, "node_models": node_models},
        rep,
        hmetrics,
    )
    results_summary[paradigm_name] = hmetrics

    # ---------------------- Local Classifier per Level (LCL) ----------------
    paradigm_name = "local_classifier_per_level"
    print(f"\n===== Running paradigm: {paradigm_name} =====")
    ensure_dir(os.path.join(out_dir, paradigm_name))

    level_models: Dict[str, Dict] = {}
    for lvl in tqdm(target_levels, desc="LCL: training levels"):
        le = LabelEncoder()
        le.fit(train_df[lvl].astype(str))
        y_train_lvl = le.transform(train_df[lvl].astype(str))
        clf_lvl = SGDClassifier(
            loss="log_loss", max_iter=1000, tol=1e-3, random_state=random_state
        )
        clf_lvl.fit(X_train_emb, y_train_lvl)
        level_models[lvl] = {"le": le, "clf": clf_lvl}

    # Predict each level independently
    y_pred_paths = []
    for vec in tqdm(X_test_emb, desc="LCL: predicting"):
        preds = []
        for lvl in target_levels:
            info = level_models[lvl]
            enc = info["clf"].predict(vec.reshape(1, -1))[0]
            pred_label = info["le"].inverse_transform([int(enc)])[0]
            preds.append(str(pred_label))
        y_pred_paths.append(preds)
    y_pred_path_strings = ["|".join(p) for p in y_pred_paths]

    rep = classification_report(
        test_df["path_label"].astype(str),
        y_pred_path_strings,
        output_dict=True,
        zero_division=0,
    )
    hmetrics = hierarchical_metrics(y_test_paths, y_pred_paths, max_depth)
    _save_and_log(
        paradigm_name,
        {"embed_model_name": model_name, "level_models": level_models},
        rep,
        hmetrics,
    )
    results_summary[paradigm_name] = hmetrics

    # -------------------------- Top-down cascade ---------------------------
    paradigm_name = "top_down_cascade"
    print(f"\n===== Running paradigm: {paradigm_name} =====")
    ensure_dir(os.path.join(out_dir, paradigm_name))

    cascade_models: Dict[str, Dict] = {}
    for lvl in tqdm(target_levels, desc="Cascade: training levels"):
        le = LabelEncoder()
        le.fit(train_df[lvl].astype(str))
        y_train_lvl = le.transform(train_df[lvl].astype(str))
        clf_prob = _build_probabilistic_clf(X_train_emb, y_train_lvl, random_state)
        cascade_models[lvl] = {"le": le, "clf": clf_prob}

    # Predict top-down using allowed children at each step
    y_pred_paths = []
    for vec in tqdm(X_test_emb, desc="Cascade: predicting"):
        preds = []
        current_parent = tree["root"]
        for lvl in target_levels:
            info = cascade_models[lvl]
            # get probabilities
            if hasattr(info["clf"], "predict_proba"):
                probs = info["clf"].predict_proba(vec.reshape(1, -1))[0]
                classes_enc = info["clf"].classes_
            else:
                pred_enc = info["clf"].predict(vec.reshape(1, -1))[0]
                classes_enc = np.array([pred_enc])
                probs = np.array([1.0])

            label_probs = {}
            for idx, enc_val in enumerate(classes_enc):
                label = info["le"].inverse_transform([int(enc_val)])[0]
                label_probs[str(label)] = float(probs[idx])

            allowed = set(tree["children"].get(current_parent, []))
            if allowed:
                allowed_probs = {
                    lbl: p for lbl, p in label_probs.items() if lbl in allowed
                }
                if allowed_probs:
                    chosen = max(allowed_probs.items(), key=lambda x: x[1])[0]
                else:
                    chosen = max(label_probs.items(), key=lambda x: x[1])[0]
            else:
                chosen = max(label_probs.items(), key=lambda x: x[1])[0]

            preds.append(str(chosen))
            current_parent = chosen
        y_pred_paths.append(preds)

    y_pred_path_strings = ["|".join(p) for p in y_pred_paths]
    rep = classification_report(
        test_df["path_label"].astype(str),
        y_pred_path_strings,
        output_dict=True,
        zero_division=0,
    )
    hmetrics = hierarchical_metrics(y_test_paths, y_pred_paths, max_depth)
    _save_and_log(
        paradigm_name,
        {"embed_model_name": model_name, "cascade_models": cascade_models},
        rep,
        hmetrics,
    )
    results_summary[paradigm_name] = hmetrics

    # -------------------------- Path-based probability ---------------------
    paradigm_name = "path_probability_via_levels"
    print(f"\n===== Running paradigm: {paradigm_name} =====")
    ensure_dir(os.path.join(out_dir, paradigm_name))

    level_proba_models: Dict[str, Dict] = {}
    for lvl in tqdm(target_levels, desc="PathProb: training levels"):
        le = LabelEncoder()
        le.fit(train_df[lvl].astype(str))
        y_train_lvl = le.transform(train_df[lvl].astype(str))
        clf_prob = _build_probabilistic_clf(X_train_emb, y_train_lvl, random_state)
        level_proba_models[lvl] = {"le": le, "clf": clf_prob}

    # candidate paths = unique training paths
    candidate_path_strings = sorted(train_df["path_label"].astype(str).unique())
    candidate_paths = [p.split("|") for p in candidate_path_strings]

    def _predict_best_path_by_product(vec: np.ndarray):
        per_level_probs = []
        for lvl in target_levels:
            info = level_proba_models[lvl]
            if hasattr(info["clf"], "predict_proba"):
                probs = info["clf"].predict_proba(vec.reshape(1, -1))[0]
                classes_enc = info["clf"].classes_
            else:
                pred_enc = info["clf"].predict(vec.reshape(1, -1))[0]
                classes_enc = np.array([pred_enc])
                probs = np.array([1.0])
            label_probs = {}
            for idx, enc_val in enumerate(classes_enc):
                label = info["le"].inverse_transform([int(enc_val)])[0]
                label_probs[str(label)] = float(probs[idx])
            per_level_probs.append(label_probs)

        best_path = None
        best_score = -1.0
        for p in candidate_paths:
            score = 1.0
            for lvl_idx, label in enumerate(p):
                score *= per_level_probs[lvl_idx].get(str(label), 1e-12)
            if score > best_score:
                best_score = score
                best_path = p
        return best_path or candidate_paths[0]

    y_pred_paths = []
    for vec in tqdm(X_test_emb, desc="PathProb: predicting"):
        p = _predict_best_path_by_product(vec)
        y_pred_paths.append([str(x) for x in p])

    y_pred_path_strings = ["|".join(p) for p in y_pred_paths]
    rep = classification_report(
        test_df["path_label"].astype(str),
        y_pred_path_strings,
        output_dict=True,
        zero_division=0,
    )
    hmetrics = hierarchical_metrics(y_test_paths, y_pred_paths, max_depth)
    _save_and_log(
        paradigm_name,
        {"embed_model_name": model_name, "level_proba_models": level_proba_models},
        rep,
        hmetrics,
    )
    results_summary[paradigm_name] = hmetrics

    # -------------------------- Save overall summary -----------------------
    summary_rows = []
    for paradigm, metrics in results_summary.items():
        row = {"paradigm": paradigm}
        row.update(metrics)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(out_dir, "results_summary.csv"), sep=";", decimal=",", index=False
    )

    print("\n✅ Finished all paradigms. Results (CSV + models) saved to:", out_dir)
    return results_summary


# If run as script demonstration
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hierarchical GICS experiments using transformers embeddings."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="CSV containing data (with text column and target levels)",
    )
    parser.add_argument("--text_col", default="text", help="Name of the text column")
    parser.add_argument(
        "--levels",
        nargs="+",
        required=True,
        help="List of hierarchical level column names (top -> bottom)",
    )
    parser.add_argument("--gics_csv", required=True, help="GICS csv to build tree")
    parser.add_argument(
        "--out_dir", default="./gics_transformer_results", help="Output directory"
    )
    parser.add_argument(
        "--model_name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--chunk_max_length",
        type=int,
        default=256,
        help="Chunk max tokens (including special tokens).",
    )
    parser.add_argument(
        "--chunk_stride", type=int, default=50, help="Chunk stride overlap."
    )
    parser.add_argument(
        "--embed_batch_size",
        type=int,
        default=64,
        help="Window batch size to embed at once.",
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--no_cache", dest="cache", action="store_false")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    run_hierarchical_experiments_transformers(
        df,
        text_col=args.text_col,
        target_levels=args.levels,
        gics_csv_path=args.gics_csv,
        out_dir=args.out_dir,
        model_name=args.model_name,
        embed_batch_size=args.embed_batch_size,
        chunk_max_length=args.chunk_max_length,
        chunk_stride=args.chunk_stride,
        test_size=args.test_size,
        random_state=args.random_state,
        cache_embeddings=args.cache,
    )
