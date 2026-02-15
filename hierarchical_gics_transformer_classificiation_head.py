"""Implementation of a hierarchical classification head for GICS classification using a transformer model."""

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tqdm import tqdm
from typing import List, Dict

# ------------------------------- Utilities ---------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def full_path_label(row: pd.Series, levels: List[str]) -> str:
    """Concatenate hierarchical labels into a single path label string."""
    return "|".join([str(row[l]) for l in levels])

def shared_prefix_depth(true_path: List[str], pred_path: List[str]) -> int:
    depth = 0
    for a, b in zip(true_path, pred_path):
        if a == b:
            depth += 1
        else:
            break
    return depth

def hierarchical_metrics(y_true_paths: List[List[str]], y_pred_paths: List[List[str]], max_depth: int) -> Dict:
    common_list = []
    h_prec_list = []
    h_rec_list = []
    h_f1_list = []

    for t, p in zip(y_true_paths, y_pred_paths):
        common = shared_prefix_depth(t, p)
        pred_depth = len(p) if len(p) > 0 else 1
        true_depth = len(t) if len(t) > 0 else 1
        h_prec = common / pred_depth
        h_rec = common / true_depth
        h_f1 = (2 * h_prec * h_rec / (h_prec + h_rec)) if (h_prec + h_rec) > 0 else 0.0
        
        common_list.append(common)
        h_prec_list.append(h_prec)
        h_rec_list.append(h_rec)
        h_f1_list.append(h_f1)

    return {
        "h_precision_micro": float(np.mean(h_prec_list)),
        "h_recall_micro": float(np.mean(h_rec_list)),
        "h_f1_micro": float(np.mean(h_f1_list)),
        "common_depth_mean": float(np.mean(common_list))
    }

# -------------------------- Dataset Class ---------------------------

class ChunkedDocDataset(Dataset):
    """
    Explodes long documents into multiple chunks. 
    During training, every chunk inherits the label of the parent document.
    """
    def __init__(self, texts, labels, doc_indices, tokenizer, max_len=512, stride=128, inference_mode=False):
        self.samples = []
        self.tokenizer = tokenizer
        self.inference_mode = inference_mode
        
        print(f"Tokenizing and chunking {len(texts)} documents...")
        
        for idx, (text, label, doc_idx) in enumerate(zip(texts, labels, doc_indices)):
            # Tokenize the whole document first
            tokens = tokenizer.encode(str(text), add_special_tokens=False)
            
            # Create sliding windows
            start = 0
            while start < len(tokens):
                end = min(start + max_len - 2, len(tokens)) # -2 for [CLS] and [SEP]
                chunk_ids = tokens[start:end]
                
                # We store the inputs needed for the model
                self.samples.append({
                    'input_ids': chunk_ids,
                    'label': label,
                    'doc_idx': doc_idx # To aggregate back later
                })
                
                if end == len(tokens):
                    break
                
                start += (max_len - 2 - stride)
                
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        sample = self.samples[item]
        # Add special tokens on the fly
        input_ids = [self.tokenizer.cls_token_id] + sample['input_ids'] + [self.tokenizer.sep_token_id]
        
        # Pad to max_len (or handle in collate_fn for efficiency, doing simplistic padding here)
        # Note: A proper collate_fn is better for speed, but this is easier to read.
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'doc_idx': sample['doc_idx']
        }

def collate_fn(batch):
    """Pad batch dynamically to max length in this batch"""
    input_ids = [x['input_ids'] for x in batch]
    labels = torch.stack([x['label'] for x in batch])
    doc_idxs = [x['doc_idx'] for x in batch]
    
    # Pad inputs
    padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    # Create attention mask
    mask = (padded != 0).long()
    
    return {
        'input_ids': padded,
        'attention_mask': mask,
        'labels': labels,
        'doc_idxs': doc_idxs
    }

# -------------------------- Training & Evaluation ---------------------------

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate_aggregated(model, dataloader, device, num_classes):
    """
    Run inference on chunks, then aggregate probabilities by Document ID.
    """
    model.eval()
    
    # Storage for aggregation: doc_idx -> list of probability vectors
    doc_probs = {}
    doc_true_labels = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            doc_idxs = batch['doc_idxs']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            for i, doc_id in enumerate(doc_idxs):
                if doc_id not in doc_probs:
                    doc_probs[doc_id] = []
                    doc_true_labels[doc_id] = labels_np[i]
                doc_probs[doc_id].append(probs[i])

    # Aggregate: Mean Pooling of probabilities
    final_preds = []
    final_true = []
    
    # Sort by doc_id to ensure alignment
    sorted_doc_ids = sorted(doc_probs.keys())
    
    for doc_id in sorted_doc_ids:
        # Average probability across all chunks of this document
        avg_prob = np.mean(np.vstack(doc_probs[doc_id]), axis=0)
        pred_label = np.argmax(avg_prob)
        
        final_preds.append(pred_label)
        final_true.append(doc_true_labels[doc_id])
        
    return np.array(final_true), np.array(final_preds)

# --------------------------- Main Runner ---------------------------

def run_finetuning(
    df: pd.DataFrame,
    text_col: str,
    target_levels: List[str],
    out_dir: str,
    model_name: str = "ProsusAI/finbert", # 'sentence-transformers/all-MiniLM-L6-v2',
    batch_size: int = 16, # Lower than embedding batch size because gradients take memory
    epochs: int = 3,
    chunk_max_length: int = 256,
    chunk_stride: int = 50,
    learning_rate: float = 2e-5,
    test_size: float = 0.2,
    random_state: int = 42
):
    ensure_dir(out_dir)
    
    # 1. Prepare Data
    print("Preparing Data...")
    df = df.dropna(subset=target_levels + [text_col]).reset_index(drop=True)
    df['path_label'] = df.apply(lambda r: full_path_label(r, target_levels), axis=1)
    
    # Encode Labels (Global Flat Approach)
    label_encoder = LabelEncoder()
    df['label_enc'] = label_encoder.fit_transform(df['path_label'])
    num_labels = len(label_encoder.classes_)
    
    # Split Documents
    # We pass the index so we can re-aggregate chunks later
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['label_enc'])
    
    # 2. Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 3. Create Chunked Datasets
    # Note: doc_indices are just unique IDs for aggregation. We use the DataFrame index.
    train_dataset = ChunkedDocDataset(
        texts=train_df[text_col].tolist(),
        labels=train_df['label_enc'].tolist(),
        doc_indices=train_df.index.tolist(),
        tokenizer=tokenizer,
        max_len=chunk_max_length,
        stride=chunk_stride
    )
    
    test_dataset = ChunkedDocDataset(
        texts=test_df[text_col].tolist(),
        labels=test_df['label_enc'].tolist(),
        doc_indices=test_df.index.tolist(),
        tokenizer=tokenizer,
        max_len=chunk_max_length,
        stride=chunk_stride
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # Shuffle false for test to keep chunks vaguely together (though aggregation handles it via ID)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 4. Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    # 5. Training Loop
    print(f"Starting training for {epochs} epochs on {device}...")
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}")
        
        # Optional: Run eval every epoch
        y_true, y_pred = evaluate_aggregated(model, test_loader, device, num_labels)
        acc = (y_true == y_pred).mean()
        print(f"Epoch {epoch+1} Validation Accuracy (Doc Level): {acc:.4f}")

    # 6. Final Evaluation & Saving
    print("Final Evaluation...")
    y_true_enc, y_pred_enc = evaluate_aggregated(model, test_loader, device, num_labels)
    
    # Decode back to path strings
    y_true_str = label_encoder.inverse_transform(y_true_enc)
    y_pred_str = label_encoder.inverse_transform(y_pred_enc)
    
    # Convert to list of lists for hierarchical metrics
    y_true_paths = [s.split('|') for s in y_true_str]
    y_pred_paths = [s.split('|') for s in y_pred_str]
    
    # Metrics
    max_depth = len(target_levels)
    h_metrics = hierarchical_metrics(y_true_paths, y_pred_paths, max_depth)
    flat_report = classification_report(y_true_str, y_pred_str, output_dict=True, zero_division=0)
    
    print("\nHIERARCHICAL METRICS:")
    print(h_metrics)
    
    # Save artifacts
    model.save_pretrained(os.path.join(out_dir, "fine_tuned_model"))
    tokenizer.save_pretrained(os.path.join(out_dir, "fine_tuned_model"))
    
    pd.DataFrame(flat_report).transpose().to_csv(os.path.join(out_dir, 'classification_report.csv'))
    pd.DataFrame([h_metrics]).to_csv(os.path.join(out_dir, 'hierarchical_metrics.csv'))
    
    print(f"Saved model and results to {out_dir}")

# -----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--levels", nargs="+", required=True)
    parser.add_argument("--out_dir", default="./gics_finetune_results")
    parser.add_argument("--batch_size", type=int, default=16) 
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    run_finetuning(
        df, 
        text_col=args.text_col, 
        target_levels=args.levels, 
        out_dir=args.out_dir,
        batch_size=args.batch_size, 
        epochs=args.epochs
    )