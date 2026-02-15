"""Test implementation for hierarchical classification of 10-K excerpts using OpenAI's API. Includes data preparation, async API calls, response parsing, metric calculation, and MLflow logging."""
import os
import json
import re
import logging
import asyncio
import time
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from tqdm.asyncio import tqdm_asyncio
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import mlflow
import nest_asyncio

from openai import AsyncOpenAI

# Apply nest_asyncio for Jupyter compatibility
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# 1. Configuration & Setup
# --------------------------------------------------------------------------

# Initialize OpenAI Client
# Ensure OPENAI_API_KEY is set in your environment variables
client = AsyncOpenAI()

MODEL_NAME = "gpt-5"  # Or "gpt-3.5-turbo-0125"
TEMPERATURE = 1.0
prompt_id = "pmpt_6917524d43a881958e50526fd2a7ebe9000c4000310a3bd6"
# --------------------------------------------------------------------------
# 2. GICS Hierarchy & Parsing Utilities (Preserved from Local Pipeline)
# --------------------------------------------------------------------------

def read_gics_hierarchy(gics_csv_path: str, sep=';') -> pd.DataFrame:
    return pd.read_csv(gics_csv_path, sep=sep, dtype=str).fillna('')

def build_canonical_path_from_row(row: pd.Series, label_cols: List[str], sep=' > ') -> str:
    parts = [str(row[c]).strip() for c in label_cols if str(row[c]).strip() != ""]
    return sep.join(parts)

def extract_json_from_text(text: str) -> Dict:
    """Extract first JSON object from model output."""
    # First try standard json load if clean
    print(text, json.loads(text))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Regex fallback
    m = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if not m:
        return {}
    js_text = m.group(0)
    js_text = js_text.replace("'", '"')
    # Cleanup trailing commas
    js_text = re.sub(r',\s*}', '}', js_text)
    js_text = re.sub(r',\s*\]', ']', js_text)
    try:
        return json.loads(js_text)
    except Exception:
        return {}

def format_predicted_path_from_json(j: Dict, label_cols: List[str], sep=' > ') -> str:
    vals = [str(j.get(k, "") or "").strip() for k in label_cols]
    # Remove trailing empty levels
    while vals and vals[-1] == "":
        vals.pop()
    return sep.join([x for x in vals if x != ""])

def summarize_text_fast(text: str, max_chars=10000) -> str:
    """
    Naive heuristic summarizer to reduce token costs and stay within limits.
    Keeps head, middle, and tail.
    """
    if not isinstance(text, str): 
        return ""
    if len(text) <= max_chars:
        return text
    third = max_chars // 3
    return text[:third] + "\n...[SNIP]...\n" + text[len(text)//2 : len(text)//2 + third] + "\n...[SNIP]...\n" + text[-third:]

# --------------------------------------------------------------------------
# 3. Metrics (Preserved from Local Pipeline)
# --------------------------------------------------------------------------

def hierarchical_metrics(y_true_paths: List[str], y_pred_paths: List[str], label_cols: List[str], sep=' > '):
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
# 4. Async OpenAI Inference Logic
# --------------------------------------------------------------------------

async def call_openai_gics(text: str, system_prompt: str) -> str:
    """
    Single async API call using standard Chat Completions.
    Enforces JSON mode for reliability.
    """
    try:
        # response = await client.chat.completions.create(
        #     model=MODEL_NAME,
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": f"<<<START>>>\n{text}\n<<<END>>>"}
        #     ],
        #     temperature=TEMPERATURE,
        #     response_format={"type": "json_object"} 
        # )
        # return response.choices[0].message.content
        response = await client.responses.create(
            model="gpt-5",
            prompt={
                "id": prompt_id,
                "variables":{
                    "text": text,
                },
            },
            # input={"text": row["text"]}
        )
        return response.output_text
    except Exception as e:
        logger.error(f"OpenAI API Error: {e}")
        return "{}"

async def batch_process_dataframe(df: pd.DataFrame, text_col: str, system_prompt: str, 
                                  batch_size: int = 10, batch_pause: float = 0.5) -> List[str]:
    """
    Processes dataframe in batches to respect rate limits.
    """
    results = []
    texts = df[text_col].fillna("").astype(str).tolist()
    
    # Pre-summarize to save tokens/cost
    texts = [summarize_text_fast(t) for t in texts]

    # Create all tasks
    tasks = [call_openai_gics(text, system_prompt) for text in texts]
    
    # Process in chunks
    for i in range(0, len(tasks), batch_size):
        batch_tasks = tasks[i : i + batch_size]
        batch_results = await asyncio.gather(*batch_tasks)
        results.extend(batch_results)
        
        # Rate limit pause
        if i + batch_size < len(tasks):
            time.sleep(batch_pause)
            
    return results

# --------------------------------------------------------------------------
# 5. Main Pipeline / Experiment Runner
# --------------------------------------------------------------------------

def run_openai_gics_experiment(
    df: pd.DataFrame,
    text_col: str,
    target_levels: List[str],  # e.g. ['gsector', 'ggroup', 'gind', 'gsubind']
    gics_csv_path: str,
    output_dir: str,
    test_size: float = 0.2,
    batch_size: int = 5,
    run_name: str = "openai_gics_baseline"
):
    """
    Main entry point. Splits data, runs OpenAI inference, calculates metrics, logs to MLflow.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Data Prep (Preserve Stratified Sampling Logic)
    logger.info("Preparing data...")
    # Sample one representative per gsubind to ensure train set has them (legacy logic, mainly relevant if few shot prompting)
    # if 'gsubind' in df.columns:
    #     representatives = df.groupby('gsubind', group_keys=False).sample(n=1, random_state=42)
    #     remaining = df.drop(representatives.index)
    #     _, df_test = train_test_split(remaining, test_size=test_size, random_state=42)
    #     # We only run inference on the test set for cost reasons in this context
    #     eval_df = df_test.copy()
    # else:
    #     eval_df = df.sample(frac=test_size, random_state=42).copy()
    eval_df = df.sample(frac=test_size, random_state=42).copy()
    logger.info(f"Evaluation set size: {len(eval_df)} rows")

    # 2. Define System Prompt (One-Shot / Zero-Shot Logic)
    # Note: OpenAI JSON mode requires the word "JSON" in the prompt.
    system_prompt = (
        "You are an expert that maps US GAAP 10-K filings to GICS classification.\n"
        "Read the company's 10-K excerpt and return a JSON object exactly matching this format:\n"
        '{ "gsector": "...", "ggroup": "...", "gind": "...", "gsubind": "..." }\n'
        "If you cannot decide, pick the most specific reasonable label.\n"
        "Return ONLY the JSON."
    )

    # 3. Run Async Inference
    logger.info("Starting OpenAI batch inference...")
    loop = asyncio.get_event_loop()
    raw_responses = loop.run_until_complete(
        batch_process_dataframe(eval_df, text_col, system_prompt, batch_size=batch_size)
    )

    # 4. Parse Results
    preds_jsons = []
    preds_paths = []
    
    logger.info("Parsing responses...")
    for resp in raw_responses:
        j = extract_json_from_text(resp)
        preds_jsons.append(j)
        path = format_predicted_path_from_json(j, target_levels, sep=' > ')
        preds_paths.append(path)

    # 5. Calculate Metrics
    # Prepare Ground Truth
    y_true_paths = eval_df.apply(lambda r: build_canonical_path_from_row(r, target_levels, sep=' > '), axis=1).tolist()

    # Flat Metrics
    logger.info("Calculating Flat Metrics...")
    y_true_clean = [t if t.strip() else "UNKNOWN" for t in y_true_paths]
    y_pred_clean = [p if p.strip() else "UNKNOWN" for p in preds_paths]
    
    labels_unique = sorted(list(set(y_true_clean) | set(y_pred_clean)))
    
    clf_report = classification_report(
        y_true_clean, y_pred_clean, labels=labels_unique, zero_division=0, output_dict=True
    )
    prec_mac, rec_mac, f1_mac, _ = precision_recall_fscore_support(
        y_true_clean, y_pred_clean, average='macro', zero_division=0
    )

    # Hierarchical Metrics
    logger.info("Calculating Hierarchical Metrics...")
    hier_metrics = hierarchical_metrics(y_true_paths, preds_paths, target_levels)

    # 6. Save & Log to MLflow
    
    # Add predictions to dataframe
    eval_df['_pred_json'] = [json.dumps(j) for j in preds_jsons]
    eval_df['_pred_path'] = preds_paths
    eval_df['_true_path'] = y_true_paths
    eval_df['raw_response'] = raw_responses
    
    predictions_csv = os.path.join(output_dir, "predictions.csv")
    eval_df.to_csv(predictions_csv, index=False)

    logger.info("Logging to MLflow...")
    mlflow.set_experiment("hierarchical_gics_openai")
    
    with mlflow.start_run(run_name=run_name):
        # Params
        mlflow.log_param("model", MODEL_NAME)
        mlflow.log_param("n_samples", len(eval_df))
        mlflow.log_param("system_prompt", system_prompt[:200] + "...")
        mlflow.log_param("target_levels", ",".join(target_levels))
        
        # Flat Metrics
        mlflow.log_metric("flat_macro_precision", float(prec_mac))
        mlflow.log_metric("flat_macro_recall", float(rec_mac))
        mlflow.log_metric("flat_macro_f1", float(f1_mac))
        
        # Hierarchical Metrics
        for k, v in hier_metrics.items():
            mlflow.log_metric(k, float(v))
            
        # Artifacts
        clf_report_path = os.path.join(output_dir, "classification_report.csv")
        pd.DataFrame(clf_report).transpose().to_csv(clf_report_path)
        
        mlflow.log_artifact(clf_report_path)
        mlflow.log_artifact(predictions_csv)
        
        # Summary Text
        summary_txt = os.path.join(output_dir, "summary.txt")
        with open(summary_txt, "w") as f:
            f.write(f"Model: {MODEL_NAME}\n")
            f.write(f"Flat F1 (Macro): {f1_mac:.4f}\n")
            f.write(f"Hierarchical F1 (Mean): {hier_metrics['hier_f1_mean']:.4f}\n")
        mlflow.log_artifact(summary_txt)

    logger.info("Experiment complete.")
    return eval_df, hier_metrics

# --------------------------------------------------------------------------
# 6. Example Usage Block
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Example dummy data generation if running standalone
    # Replace this with your actual data loading logic
    data = {
        'text': [
            "We create software for banking.", 
            "We drill for oil in the arctic.", 
            "We sell burgers and fries.", 
            "We manufacture semiconductors."
        ] * 5,
        'gsector': ['45', '10', '25', '45'] * 5,
        'ggroup': ['4510', '1010', '2530', '4530'] * 5,
        'gind': ['451020', '101020', '253010', '453010'] * 5,
        'gsubind': ['45102010', '10102020', '25301010', '45301020'] * 5
    }
    df = pd.DataFrame(data)
    
    # Fake hierarchy CSV for testing
    with open("gics_hierarchy.csv", "w") as f:
        f.write("gsector;ggroup;gind;gsubind\n")
    
    # Run the pipeline
    run_openai_gics_experiment(
        df=df,
        text_col='text',
        target_levels=['gsector', 'ggroup', 'gind', 'gsubind'],
        gics_csv_path="gics_hierarchy.csv",
        output_dir="openai_results",
        test_size=0.5, # High split for dummy data
        batch_size=2
    )