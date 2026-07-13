import os
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from transformers import BertTokenizer, BertForSequenceClassification

# 1. Define Portable Paths relative to this script location
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model" / "saved_model"
WEIGHTS_FILE = MODEL_PATH / "model.safetensors"
DATASET_PATH = PROJECT_ROOT / "dataset" / "spam.csv"

def check_model_weights():
    """
    Verifies that the fine-tuned model weights file exists locally.
    Provides a helpful, beginner-friendly message if the weights are missing.
    """
    if not WEIGHTS_FILE.exists():
        print("=" * 80)
        print("⚠️  ERROR: Fine-tuned model weights file is missing!")
        print("=" * 80)
        print(f"Expected location: {WEIGHTS_FILE}")
        print("\nExplanation:")
        print("- The fine-tuned BERT model weights ('model.safetensors') are approximately 438 MB.")
        print("- This file is intentionally excluded from Git tracking because it exceeds GitHub's")
        print("  standard 100 MB per-file tracking limit.")
        print("- Running predictions or evaluation requires these fine-tuned weights.")
        print("\nHow to resolve:")
        print("1. If you have trained the model locally, ensure 'train.py' was run to completion")
        print("   and weights saved into 'model/saved_model/'.")
        print("2. If cloning the repository, obtain 'model.safetensors' from your model source")
        print("   and place it inside the 'model/saved_model/' directory.")
        print("=" * 80)
        return False
    return True

def predict_spam(message, tokenizer, model, device):
    """
    Performs inference on a single SMS message.
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
        
    probabilities = torch.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item() * 100
    
    prediction = "SPAM" if predicted_class == 1 else "HAM"
    return prediction, confidence

class SpamTestDataset(Dataset):
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
    Loads dataset/spam.csv, reconstructs the test split, runs batch inference,
    and displays the accuracy, confusion matrix, and classification report.
    """
    if not DATASET_PATH.exists():
        print(f"⚠️  Error: Dataset file not found at {DATASET_PATH}.")
        return
        
    print("\nLoading and preprocessing dataset for evaluation...")
    df = pd.read_csv(DATASET_PATH, encoding="ISO-8859-1")
    df = df[["v1", "v2"]]
    df.columns = ["label", "text"]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    
    print("Reconstructing original 80/20 train/test split...")
    # Recreate the exact split used during training
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=0.2,
        random_state=42
    )
    print(f"Testing Samples: {len(test_texts)}")
    
    print("Tokenizing test split...")
    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=128
    )
    
    test_dataset = SpamTestDataset(test_encodings, test_labels)
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
    report = classification_report(test_labels, predicted_labels, target_names=["HAM", "SPAM"])
    
    print("\n" + "=" * 40)
    print("           EVALUATION RESULTS           ")
    print("=" * 40)
    print(f"Calculated Test Accuracy: {acc * 100:.6f}%")
    print(f"Expected Test Accuracy  : ~99.372197%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print("=" * 40)

def main():
    parser = argparse.ArgumentParser(description="Inference and evaluation script for BERT SMS Spam Classifier.")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run batch evaluation on the held-out test split and display metrics."
    )
    args = parser.parse_args()

    # 1. Model Weights Pre-check
    if not check_model_weights():
        return

    # 2. Select Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 3. Load Tokenizer and Saved Fine-Tuned Model
    print("Loading tokenizer and fine-tuned BERT model...")
    # Load tokenizer directly from pretrained model hub to avoid JSONDecodeError
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    print("Loaded successfully!")

    # 4. Determine execution mode
    if args.evaluate:
        evaluate_model(tokenizer, model, device)
    else:
        # Interactive mode
        print("\n=== SMS Spam Predictor (Interactive Mode) ===")
        print("Type 'exit' or 'quit' to stop.")
        while True:
            try:
                message = input("\nEnter SMS Message: ").strip()
                if not message:
                    continue
                if message.lower() in ["exit", "quit"]:
                    print("Exiting Predictor. Goodbye!")
                    break
                
                prediction, confidence = predict_spam(message, tokenizer, model, device)
                print(f"Prediction: {prediction}")
                print(f"Confidence: {confidence:.2f}%")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting Predictor. Goodbye!")
                break

if __name__ == "__main__":
    main()
