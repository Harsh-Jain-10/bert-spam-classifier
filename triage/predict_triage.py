import os
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from transformers import BertTokenizer, BertForSequenceClassification

# Define Portable Paths relative to this script location (which is now in triage/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "saved_model_triage"
WEIGHTS_FILE = MODEL_PATH / "model.safetensors"
DATASET_PATH = PROJECT_ROOT / "dataset" / "triage_labeled.csv"
TEST_IDS_PATH = MODEL_PATH / "test_row_ids.txt"

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

def check_model_weights():
    """
    Verifies that the fine-tuned model weights file exists locally.
    """
    if not WEIGHTS_FILE.exists():
        print("=" * 80)
        print("⚠️  ERROR: Fine-tuned triage model weights file is missing!")
        print("=" * 80)
        print(f"Expected location: {WEIGHTS_FILE}")
        print("\nExplanation:")
        print("- Fine-tuned BERT weights are excluded from Git.")
        print("- Please run 'triage/train_triage.py' to train and save the model.")
        print("=" * 80)
        return False
    return True

def predict_triage(message, tokenizer, model, device):
    """
    Performs inference on a single email message text.
    Returns the predicted label, confidence, and dictionary of all class probabilities.
    """
    encoding = tokenizer(
        message,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    
    # Move tensors to the selected device
    encoding = {key: value.to(device) for key, value in encoding.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model(**encoding)
        
    probabilities = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class]
    
    scores = {id2label[i]: float(probabilities[i]) for i in range(6)}
    predicted_label = id2label[predicted_class]
    
    return predicted_label, confidence, scores

class TriageTestDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index):
        item = {
            key: torch.tensor(value[index])
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index])
        return item

    def __len__(self):
        return len(self.labels)

def evaluate_model(tokenizer, model, device):
    """
    Loads dataset/triage_labeled.csv, filters it by row_ids saved in
    saved_model_triage/test_row_ids.txt, runs batch inference,
    and displays metrics.
    """
    if not DATASET_PATH.exists():
        print(f"⚠️  Error: Dataset file not found at {DATASET_PATH}.")
        return
        
    if not TEST_IDS_PATH.exists():
        print(f"⚠️  Error: Test split IDs file not found at {TEST_IDS_PATH}.")
        print("Please ensure you run train_triage.py to completion first.")
        return
        
    print(f"\nLoading test split row IDs from {TEST_IDS_PATH.name}...")
    with open(TEST_IDS_PATH, 'r', encoding='utf-8') as f:
        test_row_ids = set(line.strip() for line in f if line.strip())
        
    print(f"Loading and preprocessing dataset from {DATASET_PATH.name}...")
    df = pd.read_csv(DATASET_PATH)
    
    # Filter by row_ids from the test split
    df['row_id_str'] = df['row_id'].astype(str)
    df_test = df[df['row_id_str'].isin(test_row_ids)].copy()
    
    if df_test.empty:
        print("⚠️  Error: No matching test rows found in dataset. The dataset may have changed or been cleared.")
        return
        
    print(f"Loaded {len(df_test)} test split samples (no leakage).")
    
    # Map label names to class IDs
    df_test['label_id'] = df_test['label'].map(LABEL_MAP)
    df_test = df_test.dropna(subset=['label_id'])
    test_labels = df_test['label_id'].astype(int).tolist()
    test_texts = df_test['text'].tolist()
    
    print("Tokenizing test split...")
    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=128
    )
    
    test_dataset = TriageTestDataset(test_encodings, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    predicted_labels = []
    print("Running batch inference on device...")
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            predicted_labels.extend(predictions.cpu().numpy())
            
    acc = accuracy_score(test_labels, predicted_labels)
    cm = confusion_matrix(test_labels, predicted_labels)
    
    target_names = [id2label[i] for i in range(6)]
    report = classification_report(test_labels, predicted_labels, target_names=target_names)
    
    print("\n" + "=" * 60)
    print("           TRIAGE EVALUATION RESULTS           ")
    print("=" * 60)
    print(f"Calculated Test Accuracy: {acc * 100:.6f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Inference and evaluation script for BERT multi-class Email Triage Classifier.")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run batch evaluation on the held-out test split using saved row_ids and display metrics."
    )
    args = parser.parse_args()

    # 1. Model Weights Pre-check
    if not check_model_weights():
        return

    # 2. Select Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 3. Load Tokenizer and Saved Fine-Tuned Model
    print("Loading tokenizer and fine-tuned triage BERT model...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    print("Loaded successfully!")

    # 4. Determine execution mode
    if args.evaluate:
        evaluate_model(tokenizer, model, device)
    else:
        # Interactive mode
        print("\n=== Email Triage Predictor (Interactive Mode) ===")
        print("Type 'exit' or 'quit' to stop.")
        while True:
            try:
                message = input("\nEnter Email Message (subject + body or text): ").strip()
                if not message:
                    continue
                if message.lower() in ["exit", "quit"]:
                    print("Exiting Predictor. Goodbye!")
                    break
                
                prediction, confidence, scores = predict_triage(message, tokenizer, model, device)
                print(f"\nPrediction: {prediction} (Confidence: {confidence * 100:.2f}%)")
                print("All Class Scores:")
                for label_name, score in scores.items():
                    print(f"  - {label_name}: {score * 100:.2f}%")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting Predictor. Goodbye!")
                break

if __name__ == "__main__":
    main()
