"""hierarchical_gics_pipeline.py

Hierarchical GICS classification experiments with bag-of-words features.
"""

import os
from collections import defaultdict
from typing import List, Dict

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
import joblib

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


def build_tree_from_csv(gics_csv_path: str, levels: List[str], sep=';') -> Dict:
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


def hierarchical_metrics(y_true_paths: List[List[str]], y_pred_paths: List[List[str]], max_depth: int) -> Dict:
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
        "common_depth_mean": float(np.mean(common_list))
    }


def make_vectorizers(max_features=20000, ngram_range=(1, 2), stop_words='english'):
    """Return vectorizer templates (unfitted). Clone before fitting."""
    return {
        "count": CountVectorizer(max_features=max_features, stop_words=stop_words, ngram_range=ngram_range),
        "tfidf": TfidfVectorizer(max_features=max_features, stop_words=stop_words, ngram_range=ngram_range),
        "hashing": HashingVectorizer(n_features=2 ** 18, stop_words=stop_words, alternate_sign=False)
    }


# -------------------------- Per-node classifier ----------------------------

class PerNodeClassifier:
    """Train a classifier for each node in the hierarchy."""

    def __init__(self, tree: Dict, vectorizer_template, base_clf=None):
        self.tree = tree
        self.vectorizer_template = vectorizer_template
        self.base_clf = base_clf or SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)
        self.node_models = {}  # node -> {"le","clf","vec"}

    def fit(self, texts: List[str], paths: List[List[str]]):
        children = self.tree['children']
        root = self.tree['root']

        node_samples = defaultdict(list)
        for i, p in enumerate(paths):
            node_samples[root].append(i)
            for step in p:
                node_samples[step].append(i)

        for node, childs in tqdm(children.items(), desc="PerNode: training nodes"):
            if len(childs) <= 1:
                continue
            idxs = node_samples.get(node, [])
            if not idxs:
                continue
            X = [texts[i] for i in idxs]
            y = []
            for i in idxs:
                p = paths[i]
                if node == root:
                    child = p[0]
                else:
                    try:
                        pos = p.index(node)
                        child = p[pos + 1]
                    except Exception:
                        child = node
                y.append(str(child))
            le = LabelEncoder()
            y_enc = le.fit_transform(y)
            clf = clone(self.base_clf)
            vec = clone(self.vectorizer_template)
            X_vec = vec.fit_transform(X)
            clf.fit(X_vec, y_enc)
            self.node_models[node] = {"le": le, "clf": clf, "vec": vec}

    def predict(self, texts: List[str], max_depth: int) -> List[List[str]]:
        preds = []
        for text in tqdm(texts, desc="PerNode: predicting"):
            path = []
            node = self.tree['root']
            for _ in range(max_depth):
                model_info = self.node_models.get(node)
                if model_info is None:
                    break
                vec = model_info['vec'].transform([text])
                pred_enc = model_info['clf'].predict(vec)[0]
                child = model_info['le'].inverse_transform([pred_enc])[0]
                path.append(str(child))
                node = child
            preds.append(path)
        return preds


# ------------------------------ Main runner -------------------------------

def run_hierarchical_experiments(
    df: pd.DataFrame,
    text_col: str,
    target_levels: List[str],
    gics_csv_path: str,
    out_dir: str,
    test_size: float = 0.2,
    random_state: int = 42,
    mlflow_enabled: bool = True,
):
    """Run all paradigms and save results (CSV) and models (joblib) to out_dir.

    Returns:
        results_summary: dict mapping paradigm -> hierarchical_metrics dict
    """
    # Basic validation
    assert text_col in df.columns, f"text_col '{text_col}' not found in dataframe"
    assert all(l in df.columns for l in target_levels), "Some target level columns missing from dataframe"

    ensure_dir(out_dir)
    tree = build_tree_from_csv(gics_csv_path, target_levels)
    max_depth = len(target_levels)

    # prepare dataframe and full path label (cast labels to str to avoid dtype issues)
    df = df.copy()
    df = df.dropna(subset=target_levels + [text_col]).reset_index(drop=True)
    df['path_label'] = df.apply(lambda r: full_path_label(r, target_levels), axis=1)

    # train/test split (stratify by path_label if possible)
    stratify_opt = df['path_label'] if df['path_label'].value_counts().min() > 1 else None
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify_opt)

    X_train_texts = train_df[text_col].astype(str).tolist()
    X_test_texts = test_df[text_col].astype(str).tolist()
    y_train_paths = [r.split('|') for r in train_df['path_label'].astype(str)]
    y_test_paths = [r.split('|') for r in test_df['path_label'].astype(str)]

    vectorizers = make_vectorizers()
    results_summary = {}

    def _save_and_log(paradigm_name: str, model_obj, flat_report: Dict, hmetrics: Dict):
        """Save model (joblib), classification report (csv) and hierarchical metrics (csv)."""
        pdir = os.path.join(out_dir, paradigm_name)
        ensure_dir(pdir)
        model_path = os.path.join(pdir, 'model.joblib')
        joblib.dump(model_obj, model_path)

        rep_df = pd.DataFrame(flat_report).transpose()
        rep_df.to_csv(os.path.join(pdir, 'classification_report.csv'), sep=";", decimal=",", index=True)

        h_df = pd.DataFrame([hmetrics])
        h_df.to_csv(os.path.join(pdir, 'hierarchical_metrics.csv'), sep=";", decimal=",", index=False)

        if mlflow_enabled:
            if not MLFLOW_AVAILABLE:
                print(f"[warning] mlflow_enabled=True but mlflow not available; skipping MLflow logging for {paradigm_name}")
            else:
                try:
                    mlflow.set_experiment("bagOfWords")
                    with mlflow.start_run(run_name=paradigm_name):
                        mlflow.log_param("paradigm", paradigm_name)
                        for k, v in hmetrics.items():
                            try:
                                mlflow.log_metric(k, float(v))
                            except Exception:
                                pass
                        mlflow.log_artifact(os.path.join(pdir, 'classification_report.csv'))
                        mlflow.log_artifact(os.path.join(pdir, 'hierarchical_metrics.csv'))
                        mlflow.log_artifact(model_path)
                except Exception as e:
                    print(f"[warning] mlflow logging failed for {paradigm_name}: {e}")

    # ------------------------ Global Flat (single multiclass) -----------------
    paradigm_name = "global_flat"
    print(f"\n===== Running paradigm: {paradigm_name} =====")
    ensure_dir(os.path.join(out_dir, paradigm_name))

    # Fit path label encoder on train only (strings)
    path_le = LabelEncoder()
    path_le.fit(train_df['path_label'].astype(str))
    y_train_path_enc = path_le.transform(train_df['path_label'].astype(str))

    # vectorize and train
    vec = clone(vectorizers['tfidf'])
    X_train_vec = vec.fit_transform(X_train_texts)
    X_test_vec = vec.transform(X_test_texts)
    clf = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=random_state)
    clf.fit(X_train_vec, y_train_path_enc)

    # predict and decode back to path strings
    y_pred_enc = clf.predict(X_test_vec)
    y_pred_paths_flat = [str(v) for v in path_le.inverse_transform(y_pred_enc)]
    y_pred_paths = [p.split("|") for p in y_pred_paths_flat]

    rep = classification_report(test_df['path_label'].astype(str), y_pred_paths_flat, output_dict=True, zero_division=0)
    hmetrics = hierarchical_metrics(y_test_paths, y_pred_paths, max_depth)
    _save_and_log(paradigm_name, {'vectorizer': vec, 'clf': clf, 'label_encoder': path_le}, rep, hmetrics)
    results_summary[paradigm_name] = hmetrics

    # ------------------- Global Multi-output (predict each level jointly) ---
    paradigm_name = "global_multioutput"
    print(f"\n===== Running paradigm: {paradigm_name} =====")
    ensure_dir(os.path.join(out_dir, paradigm_name))

    # Fit per-level encoders on train only (strings)
    level_encoders: Dict[str, LabelEncoder] = {}
    y_train_enc_cols = []
    for lvl in target_levels:
        le = LabelEncoder()
        le.fit(train_df[lvl].astype(str))
        level_encoders[lvl] = le
        y_train_enc_cols.append(le.transform(train_df[lvl].astype(str)))
    # stack to shape (n_samples, n_levels)
    y_train_enc = np.vstack(y_train_enc_cols).T

    vec = clone(vectorizers['tfidf'])
    X_train_vec = vec.fit_transform(X_train_texts)
    X_test_vec = vec.transform(X_test_texts)

    base = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=random_state)
    multi = MultiOutputClassifier(base, n_jobs=-1)
    multi.fit(X_train_vec, y_train_enc)
    y_pred_enc = multi.predict(X_test_vec)  # shape (n_samples, n_levels)

    # decode predictions to label strings per level
    y_pred_levels = []
    for col_idx, lvl in enumerate(target_levels):
        le = level_encoders[lvl]
        y_pred_levels.append(le.inverse_transform(y_pred_enc[:, col_idx]))

    # assemble paths (ensure string cast)
    y_pred_paths = []
    for i in range(len(X_test_texts)):
        path = [str(y_pred_levels[level_idx][i]) for level_idx in range(len(target_levels))]
        y_pred_paths.append(path)
    y_pred_path_strings = ["|".join(p) for p in y_pred_paths]

    rep = classification_report(test_df['path_label'].astype(str), y_pred_path_strings, output_dict=True, zero_division=0)
    hmetrics = hierarchical_metrics(y_test_paths, y_pred_paths, max_depth)
    _save_and_log(paradigm_name, {'vectorizer': vec, 'multi': multi, 'level_encoders': level_encoders}, rep, hmetrics)
    results_summary[paradigm_name] = hmetrics

    # ---------------------- Local Classifier per Node (LCN) ------------------
    paradigm_name = "local_classifier_per_node"
    print(f"\n===== Running paradigm: {paradigm_name} =====")
    ensure_dir(os.path.join(out_dir, paradigm_name))

    vec_template = vectorizers['tfidf']
    base_clf = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=random_state)
    lcn = PerNodeClassifier(tree, vectorizer_template=vec_template, base_clf=base_clf)
    lcn.fit(X_train_texts, y_train_paths)
    y_pred_paths = lcn.predict(X_test_texts, max_depth=max_depth)
    y_pred_path_strings = ["|".join(p) for p in y_pred_paths]

    rep = classification_report(test_df['path_label'].astype(str), y_pred_path_strings, output_dict=True, zero_division=0)
    hmetrics = hierarchical_metrics(y_test_paths, y_pred_paths, max_depth)
    _save_and_log(paradigm_name, lcn, rep, hmetrics)
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
        vec_lvl = clone(vectorizers['tfidf'])
        X_train_vec_lvl = vec_lvl.fit_transform(X_train_texts)
        clf_lvl = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=random_state)
        clf_lvl.fit(X_train_vec_lvl, y_train_lvl)
        level_models[lvl] = {'le': le, 'vec': vec_lvl, 'clf': clf_lvl}

    # Predict each level independently
    y_pred_paths = []
    for text in tqdm(X_test_texts, desc="LCL: predicting"):
        preds = []
        for lvl in target_levels:
            info = level_models[lvl]
            enc = info['clf'].predict(info['vec'].transform([text]))[0]
            pred_label = info['le'].inverse_transform([enc])[0]
            preds.append(str(pred_label))
        y_pred_paths.append(preds)
    y_pred_path_strings = ["|".join(p) for p in y_pred_paths]

    rep = classification_report(test_df['path_label'].astype(str), y_pred_path_strings, output_dict=True, zero_division=0)
    hmetrics = hierarchical_metrics(y_test_paths, y_pred_paths, max_depth)
    _save_and_log(paradigm_name, level_models, rep, hmetrics)
    results_summary[paradigm_name] = hmetrics

    # -------------------------- Top-down cascade ---------------------------
    paradigm_name = "top_down_cascade"
    print(f"\n===== Running paradigm: {paradigm_name} =====")
    ensure_dir(os.path.join(out_dir, paradigm_name))

    # Helper to build a probabilistic classifier robustly
    def _build_probabilistic_clf(X_vec, y_vec, random_state):
        """Return a fitted classifier that supports predict_proba.

        Strategy:
          - if only one class present -> DummyClassifier (predicts constant class with prob 1)
          - else if min per-class count >= 3 -> CalibratedClassifierCV(cv=min(3, min_count))
          - else -> LogisticRegression (multinomial) as fallback (supports predict_proba)
        """
        classes, counts = np.unique(y_vec, return_counts=True)
        min_count = counts.min()
        n_classes = classes.size

        if n_classes == 1:
            # Always predict the single class
            dummy = DummyClassifier(strategy='constant', constant=classes[0])
            dummy.fit(X_vec, y_vec)
            return dummy
        # choose calibration or fallback
        if min_count >= 3:
            # safe to do 3-fold calibration (or fewer if min_count < 3)
            cv_folds = min(3, int(min_count))
            base = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=random_state)
            try:
                clf_cal = CalibratedClassifierCV(base, cv=cv_folds)
                clf_cal.fit(X_vec, y_vec)
                return clf_cal
            except Exception:
                # fallback to LogisticRegression if calibration fails
                lr = LogisticRegression(multi_class='multinomial', max_iter=2000, random_state=random_state)
                lr.fit(X_vec, y_vec)
                return lr
        else:
            # Not enough samples per class for reliable calibration -> use LogisticRegression
            lr = LogisticRegression(multi_class='multinomial', max_iter=2000, random_state=random_state)
            lr.fit(X_vec, y_vec)
            return lr

    # Train per-level probabilistic classifiers for cascade
    cascade_models: Dict[str, Dict] = {}
    for lvl in tqdm(target_levels, desc="Cascade: training levels"):
        le = LabelEncoder()
        le.fit(train_df[lvl].astype(str))
        y_train_lvl = le.transform(train_df[lvl].astype(str))
        vec_lvl = clone(vectorizers['tfidf'])
        X_train_vec_lvl = vec_lvl.fit_transform(X_train_texts)
        clf_prob = _build_probabilistic_clf(X_train_vec_lvl, y_train_lvl, random_state)
        cascade_models[lvl] = {'le': le, 'vec': vec_lvl, 'clf': clf_prob}

    # Predict top-down using allowed children at each step
    y_pred_paths = []
    for text in tqdm(X_test_texts, desc="Cascade: predicting"):
        preds = []
        current_parent = tree['root']
        for lvl in target_levels:
            info = cascade_models[lvl]
            vec = info['vec'].transform([text])
            # get probabilities
            if hasattr(info['clf'], 'predict_proba'):
                probs = info['clf'].predict_proba(vec)[0]
                classes_enc = info['clf'].classes_
            else:
                # fallback: use predict() and assign prob 1 to prediction
                pred_enc = info['clf'].predict(vec)[0]
                classes_enc = np.unique(np.array(info['clf'].classes_)) if hasattr(info['clf'], 'classes_') else np.array([pred_enc])
                probs = np.array([1.0 if c == pred_enc else 0.0 for c in classes_enc])

            label_probs = {}
            for idx, enc_val in enumerate(classes_enc):
                label = info['le'].inverse_transform([int(enc_val)])[0]
                label_probs[str(label)] = float(probs[idx])

            allowed = set(tree['children'].get(current_parent, []))
            if allowed:
                allowed_probs = {lbl: p for lbl, p in label_probs.items() if lbl in allowed}
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
    rep = classification_report(test_df['path_label'].astype(str), y_pred_path_strings, output_dict=True, zero_division=0)
    hmetrics = hierarchical_metrics(y_test_paths, y_pred_paths, max_depth)
    _save_and_log(paradigm_name, cascade_models, rep, hmetrics)
    results_summary[paradigm_name] = hmetrics

    # -------------------------- Path-based probability ---------------------
    paradigm_name = "path_probability_via_levels"
    print(f"\n===== Running paradigm: {paradigm_name} =====")
    ensure_dir(os.path.join(out_dir, paradigm_name))

    # Train per-level probabilistic classifiers (same robust builder)
    level_proba_models: Dict[str, Dict] = {}
    for lvl in tqdm(target_levels, desc="PathProb: training levels"):
        le = LabelEncoder()
        le.fit(train_df[lvl].astype(str))
        y_train_lvl = le.transform(train_df[lvl].astype(str))
        vec_lvl = clone(vectorizers['tfidf'])
        X_train_vec_lvl = vec_lvl.fit_transform(X_train_texts)
        clf_prob = _build_probabilistic_clf(X_train_vec_lvl, y_train_lvl, random_state)
        level_proba_models[lvl] = {'le': le, 'vec': vec_lvl, 'clf': clf_prob}

    # candidate paths = unique training paths (only consider seen train paths)
    candidate_path_strings = sorted(train_df['path_label'].astype(str).unique())
    candidate_paths = [p.split("|") for p in candidate_path_strings]

    def _predict_best_path_by_product(text: str):
        per_level_probs = []
        for lvl in target_levels:
            info = level_proba_models[lvl]
            vec = info['vec'].transform([text])
            if hasattr(info['clf'], 'predict_proba'):
                probs = info['clf'].predict_proba(vec)[0]
                classes_enc = info['clf'].classes_
            else:
                pred_enc = info['clf'].predict(vec)[0]
                classes_enc = np.array([pred_enc])
                probs = np.array([1.0])
            label_probs = {}
            for idx, enc_val in enumerate(classes_enc):
                label = info['le'].inverse_transform([int(enc_val)])[0]
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
    for text in tqdm(X_test_texts, desc="PathProb: predicting"):
        p = _predict_best_path_by_product(text)
        y_pred_paths.append([str(x) for x in p])

    y_pred_path_strings = ["|".join(p) for p in y_pred_paths]
    rep = classification_report(test_df['path_label'].astype(str), y_pred_path_strings, output_dict=True, zero_division=0)
    hmetrics = hierarchical_metrics(y_test_paths, y_pred_paths, max_depth)
    _save_and_log(paradigm_name, level_proba_models, rep, hmetrics)
    results_summary[paradigm_name] = hmetrics

    # -------------------------- Save overall summary -----------------------
    summary_rows = []
    for paradigm, metrics in results_summary.items():
        row = {'paradigm': paradigm}
        row.update(metrics)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, 'results_summary.csv'), sep=";", decimal=",", index=False)

    print("\n✅ Finished all paradigms. Results (CSV + models) saved to:", out_dir)
    return results_summary
