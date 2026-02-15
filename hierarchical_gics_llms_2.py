
"""
Implementation for fine tuning small LLMs locally and testing it on the full hierarchical GICS classification task. This is a more experimental script where I try out different approaches to fine-tuning and inference, and also implement the hierarchical metrics.
"""

import os
from pathlib import Path
import json
import re
import logging
from typing import List, Dict
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    BitsAndBytesConfig
)
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import mlflow
import torch
from torch.utils.data import Dataset

from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

def load_model_pipeline(model_name: str, device: int = -1, use_lora: bool = True):
    """
    Load a quantized LLM pipeline with optional LoRA adapters.
    Automatically fixes embedding/tokenizer mismatches.
    Works for both base and fine-tuned PEFT checkpoints.
    """
    logger.info(f"Loading model pipeline for '{model_name}' (device={device})...")


    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "attn_implementation": "eager",
    }

    # --- Load tokenizer first ---
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # --- Detect if this path is a PEFT fine-tuned model ---
    is_peft_model = (Path(model_name) / "adapter_config.json").exists()

    if is_peft_model:
        # Load PEFT config to get base model
        logger.info("Detected PEFT adapter checkpoint. Loading base model first...")
        peft_config = PeftConfig.from_pretrained(model_name)
        base_model_name = peft_config.base_model_name_or_path

        # Load base model first
        model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)

        # Resize embeddings before loading adapter
        if model.get_input_embeddings().weight.size(0) != len(tokenizer):
            logger.warning(
                f"Resizing embeddings from {model.get_input_embeddings().weight.size(0)} "
                f"to {len(tokenizer)} to match tokenizer."
            )
            model.resize_token_embeddings(len(tokenizer))

        # Now load PEFT adapter
        model = PeftModel.from_pretrained(model, model_name)

    else:
        # Standard model load (no adapters)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        if model.get_input_embeddings().weight.size(0) != len(tokenizer):
            logger.warning(
                f"Resizing embeddings from {model.get_input_embeddings().weight.size(0)} "
                f"to {len(tokenizer)} to match tokenizer."
            )
            model.resize_token_embeddings(len(tokenizer))

        # Attach new LoRA adapter if desired
        if use_lora:
            try:
                lora_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, lora_config)
                logger.info("Attached LoRA adapters for fine-tuning.")
            except Exception as e:
                logger.warning(f"Skipped LoRA attachment: {e}")

    # --- Build pipeline ---
    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )

    return text_gen, model, tokenizer



# --------------------------------------------------------------------------
# GICS hierarchy utilities
# --------------------------------------------------------------------------

def read_gics_hierarchy(gics_csv_path: str, sep=';') -> pd.DataFrame:
    return pd.read_csv(gics_csv_path, sep=sep, dtype=str).fillna('')


def build_canonical_path_from_row(row: pd.Series, label_cols: List[str], sep=' > ') -> str:
    parts = [str(row[c]).strip() for c in label_cols if str(row[c]).strip() != ""]
    return sep.join(parts)


def df_paths_from_labels(df: pd.DataFrame, label_cols: List[str], sep=' > ') -> List[str]:
    return df.apply(lambda r: build_canonical_path_from_row(r, label_cols, sep=sep), axis=1).tolist()


# --------------------------------------------------------------------------
# LLM prompting
# --------------------------------------------------------------------------

def prompt_for_gics_json(llm_pipeline, text: str, max_new_tokens: int = 256) -> str:
    """Ask the LLM for JSON GICS levels for a single 10-K excerpt."""
    prompt = (
        "You are an expert that maps US GAAP 10-K filings to GICS classification.\n"
        "Read the following company's 10-K excerpt and return a JSON exactly matching this format:\n"
        '{ "g-sector": "...", "g-group": "...", "g-ind": "...", "g-subind": "..." }\n'
        "If you cannot decide, try to pick the most specific reasonable label but keep strings short.\n\n"
        "Here is the 10-K text:\n\n"
        "<<<START>>>\n" + text + "\n<<<END>>>\n\nReturn only the JSON object and nothing else."
    )
    out = llm_pipeline(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    return out[0]['generated_text']


def extract_json_from_text(text: str) -> Dict:
    """Extract first JSON object from model output."""
    m = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if not m:
        return {}
    js_text = m.group(0)
    js_text = js_text.replace("'", '"')
    js_text = re.sub(r',\s*}', '}', js_text)
    js_text = re.sub(r',\s*\]', ']', js_text)
    try:
        return json.loads(js_text)
    except Exception:
        try:
            pairs = re.findall(r'"([^"]+)"\s*:\s*"([^"]*)"', js_text)
            return {k: v for k, v in pairs}
        except Exception:
            return {}


def format_predicted_path_from_json(j: Dict, label_cols: List[str], sep=' > ') -> str:
    vals = [str(j.get(k, "") or "").strip() for k in label_cols]
    while vals and vals[-1] == "":
        vals.pop()
    return sep.join([x for x in vals if x != ""])


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def hierarchical_metrics(y_true_paths: List[str], y_pred_paths: List[str], label_cols: List[str], sep=' > '):
    n = len(y_true_paths)
    precs, recs, f1s, common_prefix_lens = [], [], [], []
    max_depth = len(label_cols)

    for t, p in zip(y_true_paths, y_pred_paths):
        t_levels = t.split(sep) if t else []
        p_levels = p.split(sep) if p else []
        match = 0
        for tl, pl in zip(t_levels, p_levels):
            if tl.strip() == pl.strip() and tl.strip() != "":
                match += 1
            else:
                break
        pred_depth = len(p_levels) if len(p_levels) > 0 else 1
        true_depth = len(t_levels) if len(t_levels) > 0 else 1
        precision_i = (match / pred_depth) if pred_depth > 0 else 0
        recall_i = (match / true_depth) if true_depth > 0 else 0
        f1_i = 2 * precision_i * recall_i / (precision_i + recall_i) if (precision_i + recall_i) > 0 else 0
        precs.append(precision_i)
        recs.append(recall_i)
        f1s.append(f1_i)
        common_prefix_lens.append(match / max_depth)

    return {
        'hier_precision_mean': float(np.mean(precs)),
        'hier_recall_mean': float(np.mean(recs)),
        'hier_f1_mean': float(np.mean(f1s)),
        'avg_normalized_common_prefix_len': float(np.mean(common_prefix_lens)),
        'max_depth': max_depth
    }


# --------------------------------------------------------------------------
# Fine-tuning helper
# --------------------------------------------------------------------------

def fine_tune_model(model, tokenizer, df_train, text_col, output_dir, max_length=1024, label_max_length=128):
    """
    Fine-tune a Gemma-style causal LM to output JSON GICS codes from 10-K text.
    df_train must have columns: text_col, gsector, ggroup, gind, gsubind
    """

    os.makedirs(output_dir, exist_ok=True)

    # --- Prompt template ---
    prompt_template = (
        "You are an expert that maps US GAAP 10-K filings to GICS classification.\n"
        "Read the following company's 10-K excerpt and return a JSON exactly matching this format:\n"
        '{{ "gsector": "...", "ggroup": "...", "gind": "...", "gsubind": "..." }}\n'
        "Return only the JSON object with the correct codes and nothing else.\n\n"
        "Here is the 10-K text:\n\n"
        "<<<START>>>\n{text}\n<<<END>>>\n\n"
    )

    # --- Dataset definition ---
    class GICSDataset(Dataset):
        def __init__(self, df, tokenizer, max_length, label_max_length):
            self.df = df.reset_index(drop=True)
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.label_max_length = label_max_length

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            text = str(row[text_col])
            target_json = {
                "gsector": str(row["gsector"]),
                "ggroup": str(row["ggroup"]),
                "gind": str(row["gind"]),
                "gsubind": str(row["gsubind"]),
            }

            # Build prompt and label separately
            prompt = prompt_template.format(text=text)
            label_text = json.dumps(target_json)

            # Tokenize prompt (input) and label separately
            input_enc = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
            label_enc = self.tokenizer(
                label_text,
                truncation=True,
                max_length=self.label_max_length,
                add_special_tokens=False,
            )

            # Concatenate prompt + label for causal LM training
            input_ids = input_enc["input_ids"] + label_enc["input_ids"]
            attention_mask = [1] * len(input_ids)

            # Mask the prompt tokens (loss = -100)
            labels = [-100] * len(input_enc["input_ids"]) + label_enc["input_ids"]

            # Convert to tensors
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

    train_dataset = GICSDataset(df_train, tokenizer, max_length, label_max_length)

    # --- Data collator ---
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    # --- Training args ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        save_strategy="epoch",
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        learning_rate=5e-5,
        bf16=True,
        fp16=False,
        warmup_ratio=0.03,
        report_to="none",
        optim="adamw_torch"
    )

    # --- Trainer setup ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("Starting fine-tuning...")
    trainer.train()
    print(f"Fine-tuning completed. Saving model to {output_dir}")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


# --------------------------------------------------------------------------
# Main run function
# --------------------------------------------------------------------------

def run_hierarchical_experiments_llm(
    df: pd.DataFrame,
    text_col: str,
    target_levels: List[str],
    model_name: str,
    gics_csv_path: str,
    output_dir: str,
    hf_device: int = 0,
    max_new_tokens: int = 256,
    fine_tune: bool = False,
    fine_tune_max_training_length=2500,
    load_tuned_model: str = None,
    test_size: float = 0.2
):
    """Main entrypoint: performs (optional) fine-tuning, then LLM-based hierarchical GICS classification."""
    os.makedirs(output_dir, exist_ok=True)


    # 1) Load model + tokenizer
    llm_pipeline, model, tokenizer = load_model_pipeline(model_name, device=hf_device)

    # Step 1: Sample one representative row per gsubind (these will always be in train)
    representatives = df.groupby('gsubind', group_keys=False).sample(n=1, random_state=42)

    # Step 2: Remove those representatives from the full dataset
    remaining = df.drop(representatives.index)
    df_train, df_test = train_test_split(remaining, test_size=test_size, random_state=42)
    df_train = pd.concat([df_train, representatives], ignore_index=True)

    # 2) Optional fine-tuning or loading of a pre-tuned model
    if load_tuned_model and isinstance(load_tuned_model, str):
        finetuned_dir = os.path.join("fine-tuned_LLMs", re.sub(r'[/:]', '_', load_tuned_model))
        if not os.path.isdir(finetuned_dir):
            raise FileNotFoundError(f"The specified directory for the tuned model does not exist: {finetuned_dir}")

        logger.info(f"Loading specified model from: {finetuned_dir}")

        logger.info(f"Loading checkpoint for potential continued fine-tuning: {finetuned_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            finetuned_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation='eager'
        )
        tokenizer = AutoTokenizer.from_pretrained(finetuned_dir, use_fast=True)
        llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

        if fine_tune:
            logger.info("Continuing fine-tuning from loaded checkpoint (memory-safe)...")
            finetuned_dir_cont = os.path.join("fine-tuned_LLMs", re.sub(r'[/:]', '_', model_name) + "_continued")
            fine_tune_model(model, tokenizer, df_train, text_col, finetuned_dir_cont, max_length=fine_tune_max_training_length)
            logger.info("Reloading newly fine-tuned model...")
            model = AutoModelForCausalLM.from_pretrained(
                finetuned_dir_cont,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
                attn_implementation='eager'
            )
            tokenizer = AutoTokenizer.from_pretrained(finetuned_dir_cont, use_fast=True)
            llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        else:
            logger.info("Loaded model without further fine-tuning.")
        if model.get_input_embeddings().weight.size(0) != len(tokenizer):
            logger.warning("Resizing embeddings after reloading fine-tuned model...")
            model.resize_token_embeddings(len(tokenizer))

    elif fine_tune:
        logger.info("Fine-tuning enabled on base model...")
        finetuned_dir = os.path.join("fine-tuned_LLMs", re.sub(r'[/:]', '_', model_name))

        # --- Option 2: Expand dataset into chunks (VRAM-safe full coverage) ---
        logger.info("Expanding long texts into smaller training chunks...")
        def expand_df_with_chunks(df, tokenizer, text_col, max_tokens=1500):
            rows = []
            for _, r in tqdm(df.iterrows(), total=len(df)):
                tokens = tokenizer.encode(r[text_col], add_special_tokens=False)
                for i in range(0, len(tokens), max_tokens):
                    chunk = tokens[i:i + max_tokens]
                    text_chunk = tokenizer.decode(chunk)
                    row_copy = r.copy()
                    row_copy[text_col] = text_chunk
                    rows.append(row_copy)
            return pd.DataFrame(rows)
        
        df_train_expanded = expand_df_with_chunks(df_train, tokenizer, text_col)
        logger.info(f"Expanded from {len(df_train)} to {len(df_train_expanded)} training samples.")

        # Then call your existing training function
        fine_tune_model(model, tokenizer, df_train_expanded, text_col, finetuned_dir, max_length=fine_tune_max_training_length)

        # Reload fine-tuned model
        logger.info("Reloading fine-tuned model for inference...")
        llm_pipeline, model, tokenizer = load_model_pipeline(finetuned_dir, device=hf_device)

    else:
        logger.info("Fine-tuning disabled. Using base model directly.")

    # 3) Read GICS hierarchy
    gics_df = read_gics_hierarchy(gics_csv_path, sep=';')
    canonical_paths = {
        build_canonical_path_from_row(r, [c for c in gics_df.columns if c in target_levels], sep=' > ')
        for _, r in gics_df.iterrows()
    }
    canonical_paths = {p for p in canonical_paths if p.strip()}

    # 4) Prepare ground truth paths
    y_true_paths = df_test.apply(lambda r: build_canonical_path_from_row(r, target_levels, sep=' > '), axis=1).tolist()

    # 5) Batch inference
    preds_jsons, preds_paths = [], []
    eval_df = df_test
    logger.info(f"Starting batch inference for {len(eval_df)} items...")

    # Step A: Prepare all prompts in a list
    prompt_template = (
        "You are an expert that maps US GAAP 10-K filings to GICS classification.\n"
        "Read the following company's 10-K excerpt and return a JSON exactly matching this format where you replace ... with the correct codes:\n"
        '{{ "gsector": "...", "ggroup": "...", "gind": "...", "gsubind": "..." }}\n'
        # 'An example: {{ "gsector": "20", "ggroup": "2030", "gind": "203020", "gsubind": "20302010" }}\n'
        "Return only the JSON object with the correct codes and nothing else. Do not say anything else. Your answer should begin with {{ and end with }}.\n"
        "You have to respond with a JSON object that contains the four GICS labels and a code number. Do not output anything else.\n"
        "Here is the 10-K text:\n\n"
        "<<<START>>>\n{text}\n<<<END>>>\n\n"
    )
    texts = eval_df[text_col].fillna("").astype(str).tolist()

    # --- Summarize long texts on CPU before sending to LLM (fast + memory-safe) ---
    from nltk.tokenize import sent_tokenize

    def summarize_text_fast(text, max_chars=5000):
        """
        Naive heuristic summarizer to fit context limits (~1k tokens).
        Keeps beginning, middle, and end segments.
        """
        if len(text) <= max_chars:
            return text
        # Simple fast heuristic: take head + tail + middle parts
        third = max_chars // 3
        return text[:third] + "\n...\n" + text[len(text)//2 : len(text)//2 + third] + "\n...\n" + text[-third:]

    # Apply once over all rows (vectorized, CPU only)
    texts = [summarize_text_fast(t, max_chars=10000) for t in texts]
    logger.info("Pre-summarized long inputs to ~4k chars for VRAM safety.")

    prompts = [prompt_template.format(text=t) for t in texts]

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    max_new_tokens = min(max_new_tokens, 128)  # reduce generation length

    raw_outputs = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="LLM Batch Inference (fast)"):
        batch_prompts = prompts[i:i + batch_size]
        
        # Truncate texts to 1024 tokens (or smaller if needed)
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate text
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

        # Slice the output to get only the newly generated tokens
        input_token_len = inputs['input_ids'].shape[1] # Get length of input tokens
        generated_tokens = outputs[:, input_token_len:]
        
        # Decode only the new tokens
        texts_out = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # Decode generated text
        # texts_out = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Wrap outputs in same structure as pipeline for compatibility
        raw_outputs.extend([ [{"generated_text": t}] for t in texts_out ])

    # Step C: Process the results (this part remains the same)
    logger.info("Processing inference results...")
    for output in tqdm(raw_outputs, desc="Processing results"):
        try:
            # The output is a list containing one dictionary
            # print(output)
            generated_text = output[0]['generated_text']
            # print(generated_text)
            j = extract_json_from_text(generated_text)
            print(output)
            preds_jsons.append(j)
            path = format_predicted_path_from_json(j, target_levels, sep=' > ')
            # if path not in canonical_paths:
            #     path = "INVALID"
            preds_paths.append(path)
        except Exception:
            logger.exception("Error during result processing; capturing empty prediction.")
            preds_jsons.append({})
            preds_paths.append("")

    # ----------------------------------------------------------------------
    # FLAT CLASSIFICATION SECTION (clearly labeled)
    # ----------------------------------------------------------------------
    logger.info("=== FLAT CLASSIFICATION METRICS ===")
    unique_paths = sorted(list({p for p in y_true_paths if p.strip() != ""}))
    unique_paths = sorted(list(set(unique_paths) | set([p for p in preds_paths if p.strip() != ""])))
    y_true_for_report = [t if t.strip() != "" else "UNKNOWN" for t in y_true_paths]
    y_pred_for_report = [p if p.strip() != "" else "UNKNOWN" for p in preds_paths]
    labels_for_report = sorted(list(set(y_true_for_report) | set(y_pred_for_report)))

    clf_report = classification_report(
        y_true_for_report, y_pred_for_report, labels=labels_for_report, zero_division=0, output_dict=True
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_for_report, y_pred_for_report, average='macro', zero_division=0
    )

    # 6) Hierarchical metrics
    hier_metrics = hierarchical_metrics(y_true_paths, preds_paths, target_levels, sep=' > ')

    # 7) Save outputs
    out_df = eval_df.copy()
    out_df['_pred_json'] = preds_jsons
    out_df['_pred_path'] = preds_paths
    out_df['_true_path'] = y_true_paths
    predictions_csv = os.path.join(output_dir, "predictions_with_paths.csv")
    out_df.to_csv(predictions_csv, index=False)
    logger.info(f"Wrote predictions to {predictions_csv}")

    # 8) MLflow logging
    mlflow.set_experiment("hierarchical_gics_llm")
    with mlflow.start_run(run_name="global_flat"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("fine_tune", fine_tune)
        mlflow.log_param("text_col", text_col)
        mlflow.log_param("target_levels", ",".join(target_levels))
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("max_new_tokens", max_new_tokens)
        mlflow.log_param("gics_csv_path", gics_csv_path)

        mlflow.log_metric("flat_macro_precision", float(precision_macro))
        mlflow.log_metric("flat_macro_recall", float(recall_macro))
        mlflow.log_metric("flat_macro_f1", float(f1_macro))

        for k, v in hier_metrics.items():
            mlflow.log_metric(k, float(v))

        clf_report_path = os.path.join(output_dir, "classification_report.csv")
        pd.DataFrame(clf_report).to_csv(clf_report_path, index=False)

        mlflow.log_artifact(clf_report_path)
        mlflow.log_artifact(predictions_csv)

        summary_txt = os.path.join(output_dir, "summary.txt")
        with open(summary_txt, "w", encoding="utf-8") as f:
            f.write("Flat macro P/R/F1: {:.4f} / {:.4f} / {:.4f}\n".format(precision_macro, recall_macro, f1_macro))
            f.write("Hierarchical metrics:\n")
            for k, v in hier_metrics.items():
                f.write(f" {k}: {v}\n")
        mlflow.log_artifact(summary_txt)

    logger.info("MLflow logging finished. Run complete.")

    return {
        "classification_report": clf_report,
        "hierarchical_metrics": hier_metrics,
        "predictions_csv": predictions_csv,
        "mlflow_experiment": "hierarchical_gics_llm"
    }
