import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# Define target labels and indices mapping
LABEL_MAP = {
    "needs_reply": 0,
    "fyi": 1,
    "newsletter": 2,
    "cold_outreach": 3,
    "personal": 4,
    "spam": 5
}
id2label = {v: k for k, v in LABEL_MAP.items()}
label2id = LABEL_MAP

def main():
    parser = argparse.ArgumentParser(description="Train multi-class email triage classifier.")
    parser.add_argument('--min-examples-per-class', type=int, default=100, help="Minimum examples per class required to train.")
    parser.add_argument('--epochs', type=int, default=3, help="Number of training epochs.")
    args = parser.parse_args()
    
    # 1. Define Portable Paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATASET_PATH = PROJECT_ROOT / "dataset" / "triage_labeled.csv"
    SAVE_PATH = PROJECT_ROOT / "model" / "saved_model_triage"

    print("=== BERT Email Triage Multi-Class Classifier Training Pipeline ===")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Model save path: {SAVE_PATH}")
    
    if not DATASET_PATH.exists():
        print(f"⚠️ ERROR: Dataset not found at {DATASET_PATH}.")
        print("Please run label bootstrapping and corrections first.")
        sys.exit(1)
        
    # 2. Load and Preprocess Dataset
    print("\nLoading dataset...")
    df = pd.read_csv(DATASET_PATH)
    
    required_cols = {'row_id', 'text', 'label'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_cols}")
        
    # Drop rows with empty texts or labels
    df = df.dropna(subset=['text', 'label'])
    
    # Clean label strings
    df['label'] = df['label'].astype(str).str.strip()
    
    # 3. Class Counts Check
    counts = df['label'].value_counts()
    print("\nClass distribution in labeled dataset:")
    for label_name, count in LABEL_MAP.items():
        c_count = counts.get(label_name, 0)
        print(f"  {label_name}: {c_count}")
        
    # Verify all classes have minimum threshold
    failed_check = False
    for label_name in LABEL_MAP:
        c_count = counts.get(label_name, 0)
        if c_count < args.min_examples_per_class:
            print(f"❌ ERROR: Class '{label_name}' has only {c_count} examples (minimum required is {args.min_examples_per_class}).")
            failed_check = True
            
    if failed_check:
        print("\nAborting training due to insufficient samples per class.")
        print("Please use bootstrap_labels.py and merge_reviewed_labels.py to add more samples.")
        sys.exit(1)
        
    # Map text labels to integers
    df['label_id'] = df['label'].map(LABEL_MAP)
    
    # Remove any row with invalid label mapping
    df = df.dropna(subset=['label_id'])
    df['label_id'] = df['label_id'].astype(int)

    # 4. Split Train/Test Sets
    print("\nSplitting dataset into stratified train and test splits (80/20)...")
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['label_id']
    )
    print(f"Training Samples: {len(train_df)}")
    print(f"Testing Samples: {len(test_df)}")

    # 5. Initialize Pretrained Tokenizer
    print("\nLoading bert-base-uncased tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # 6. Tokenize Datasets
    print("Tokenizing texts...")
    train_encodings = tokenizer(
        train_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=128
    )
    test_encodings = tokenizer(
        test_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=128
    )

    # 7. Define PyTorch Dataset
    class TriageDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels.tolist()

        def __getitem__(self, idx):
            item = {
                key: torch.tensor(val[idx])
                for key, val in self.encodings.items()
            }
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = TriageDataset(train_encodings, train_df['label_id'])
    test_dataset = TriageDataset(test_encodings, test_df['label_id'])

    # 8. Initialize Pretrained BERT Sequence Classifier for 6 Labels
    print("Loading pretrained bert-base-uncased model for 6 labels...")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=6,
        id2label=id2label,
        label2id=label2id
    )

    # 9. Define Metrics Computation
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        accuracy = accuracy_score(labels, preds)
        return {
            "accuracy": accuracy
        }

    # 10. Configure Training Arguments
    training_args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "model" / "results_triage"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        load_best_model_at_end=True,
        report_to="none"
    )

    # 11. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # 12. Run Fine-Tuning
    confirm = input("\nAre you sure you want to run the training script? This takes time on CPU. (y/N): ")
    if confirm.lower() == 'y':
        print("\nStarting training loop...")
        trainer.train()
        
        # 13. Evaluate and Save Row IDs first
        print("\nSaving test split row IDs to prevent evaluation leakage...")
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
        test_ids_path = SAVE_PATH / "test_row_ids.txt"
        with open(test_ids_path, 'w', encoding='utf-8') as f:
            for rid in test_df['row_id']:
                f.write(f"{rid}\n")
        print(f"Saved {len(test_df)} test split row IDs to {test_ids_path}")
        
        print("\nRunning evaluation & generating classification report...")
        predictions_output = trainer.predict(test_dataset)
        preds = np.argmax(predictions_output.predictions, axis=1)
        labels = predictions_output.label_ids
        
        target_names = [id2label[i] for i in range(6)]
        report = classification_report(labels, preds, target_names=target_names)
        print("\n" + "=" * 60)
        print("               EVALUATION RESULTS (TEST SPLIT)            ")
        print("=" * 60)
        print(report)
        print("=" * 60)

        # 14. Save Fine-tuned Model and Tokenizer
        print(f"\nSaving fine-tuned model and tokenizer to {SAVE_PATH}...")
        model.save_pretrained(SAVE_PATH)
        tokenizer.save_pretrained(SAVE_PATH)
        print("Model saved successfully!")
    else:
        print("Training execution skipped.")

if __name__ == "__main__":
    main()
