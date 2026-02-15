"""Initial Transformer test code using just flat classification. Mostly used for testing the chunking and sliding window approach"""

from typing import List, Optional
import os
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,  # Import TrainerCallback
    DataCollatorWithPadding,
)



def _prepare_chunked_dataset(
    texts: List[str],
    labels: List,
    tokenizer,
    max_length: int = 512,
    stride: int = 128,
    val_fraction: float = 0.1,
    random_state: int = 42,
):
    tokenized_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "doc_id": [],
    }

    for doc_index, (text, label) in tqdm(enumerate(zip(texts, labels)), total=len(texts), desc="Tokenizing"):
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_overflowing_tokens=True,
            stride=stride,
            return_attention_mask=True,
        )
        overflow_count = len(enc["input_ids"]) if isinstance(enc["input_ids"], list) else 1
        for i in range(overflow_count):
            tokenized_inputs["input_ids"].append(enc["input_ids"][i])
            tokenized_inputs["attention_mask"].append(enc["attention_mask"][i])
            tokenized_inputs["labels"].append(label)
            tokenized_inputs["doc_id"].append(doc_index)

    ds = Dataset.from_dict(tokenized_inputs)
    ds = ds.train_test_split(test_size=val_fraction, seed=random_state)
    return ds


def train_transformers_per_column(
    df: pd.DataFrame,
    label_cols: List[str],
    text_col: str = "cleaned_text",
    model_name: str = "allenai/longformer-base-4096",
    output_base_dir: str = "./transformer_models",
    epochs: int = 3,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 2,
    max_length: int = 4096,
    stride: int = 512,
    val_fraction: float = 0.1,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    seed: int = 42,
    overwrite_output_dir: bool = False,
    fp16: Optional[bool] = None,
    report_csv: str = "transformers_classification_reports.csv",
):
    os.makedirs(output_base_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    if fp16 is None:
        fp16 = use_cuda

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    reports = []
    results = {}

    for col in tqdm(label_cols, desc="Training per label column"):
        print(f"\n==== Training for label column: {col} ====")
        labels_series = df[col].astype("category")
        categories = list(labels_series.cat.categories)
        # print(categories)
        label2id = {c: i for i, c in enumerate(categories)}
        id2label = {i: c for c, i in label2id.items()}
        print(label2id, id2label)
        y = labels_series.cat.codes.tolist()
        texts = df[text_col].astype(str).tolist()

        ds = _prepare_chunked_dataset(
            texts,
            y,
            tokenizer,
            max_length=max_length,
            stride=stride,
            val_fraction=val_fraction,
            random_state=seed,
        )

        label_count = len(categories)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=label_count,
            id2label=id2label,
            label2id=label2id,
        )

        data_collator = DataCollatorWithPadding(tokenizer)

        training_args = TrainingArguments(
            output_dir=os.path.join(output_base_dir, col),
            num_train_epochs=epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            fp16=fp16,
            push_to_hub=False,
            logging_steps=50,
            save_total_limit=2,
            overwrite_output_dir=overwrite_output_dir,
        )
        
        # Add the GPU monitoring callback if available and using CUDA
        # callbacks = [GPUStatsCallback()] if GPU_MONITORING_AVAILABLE and use_cuda else []

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            # callbacks=callbacks, # Pass the callback to the Trainer
        )

        if use_cuda:
            print("\nInitial GPU stats before training:")
            # print_gpu_utilization()

        trainer.train()
        
        if use_cuda:
            print("\nFinal GPU stats after training:")
            # print_gpu_utilization()

        output_dir = os.path.join(output_base_dir, col)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        results[col] = output_dir

        # Predictions for sklearn classification report (aggregate by doc_id)
        preds = trainer.predict(ds["test"])
        y_true_chunks = preds.label_ids
        y_pred_chunks = preds.predictions
        doc_ids = ds["test"]["doc_id"]

        df_preds = pd.DataFrame({
            "doc_id": doc_ids,
            "label": y_true_chunks,
            "pred_logits": list(y_pred_chunks),
        })

        agg = (
            df_preds.groupby("doc_id")
            .agg({
                "pred_logits": lambda x: torch.tensor(x.tolist()).mean(0).numpy(),
                "label": "first"
            })
        )

        y_true = agg["label"].values
        y_pred = [logits.argmax() for logits in agg["pred_logits"]]

        report_dict = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(categories))),
            target_names=categories,
            output_dict=True,
        )

        print(report_dict)
        
        for label, metrics in report_dict.items():
            if isinstance(metrics, dict):
                row = {"column": col, "label": label}
                row.update(metrics)
                reports.append(row)

        print(f"Saved model and document-level classification report for {col}")

    reports_df = pd.DataFrame(reports)
    reports_df.to_csv(report_csv, index=False, sep=";")
    print(f"Classification reports saved to {report_csv}")


    return results