import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

def main():
    # 1. Define Portable Paths
    PROJECT_ROOT = Path(__file__).resolve().parent
    DATASET_PATH = PROJECT_ROOT / "dataset" / "spam.csv"
    SAVE_PATH = PROJECT_ROOT / "model" / "saved_model"

    print("=== BERT SMS Spam Classifier Training Pipeline ===")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Model save path: {SAVE_PATH}")
    print("\nNOTE: Training BERT from scratch/fine-tuning is computationally expensive.")
    print("In the project environment, training 3 epochs took approximately 73 minutes on CPU.")

    # 2. Load and Preprocess Dataset
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Please ensure spam.csv is in the dataset/ folder.")
    
    print("\nLoading and cleaning dataset...")
    df = pd.read_csv(DATASET_PATH, encoding="ISO-8859-1")
    
    # Keep only labels (v1) and text (v2)
    df = df[["v1", "v2"]]
    df.columns = ["label", "text"]
    
    # Map labels: ham -> 0, spam -> 1
    df["label"] = df["label"].map({
        "ham": 0,
        "spam": 1
    })

    # 3. Split Train/Test Sets
    print("Splitting dataset into train and test splits...")
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42
    )
    print(f"Training Samples: {len(train_texts)}")
    print(f"Testing Samples: {len(test_texts)}")

    # 4. Initialize Pretrained Tokenizer
    print("\nLoading bert-base-uncased tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # 5. Tokenize Datasets
    print("Tokenizing texts...")
    train_encodings = tokenizer(
        train_texts.tolist(),
        truncation=True,
        padding=True,
        max_length=128
    )
    test_encodings = tokenizer(
        test_texts.tolist(),
        truncation=True,
        padding=True,
        max_length=128
    )

    # 6. Define PyTorch Dataset
    class SpamDataset(Dataset):
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

    train_dataset = SpamDataset(train_encodings, train_labels)
    test_dataset = SpamDataset(test_encodings, test_labels)

    # 7. Initialize Pretrained BERT Sequence Classifier
    print("Loading pretrained bert-base-uncased model...")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )

    # 8. Define Metrics Computation
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        accuracy = accuracy_score(labels, preds)
        return {
            "accuracy": accuracy
        }

    # 9. Configure Training Arguments
    training_args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "model" / "results"),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        load_best_model_at_end=True,
        report_to="none"
    )

    # 10. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # 11. Run Fine-Tuning (Call included but execution warning displayed in main)
    # WARNING: Do not execute main() if you only want to inspect code.
    confirm = input("\nAre you sure you want to run the training script? This takes ~73 minutes. (y/N): ")
    if confirm.lower() == 'y':
        print("\nStarting training loop...")
        trainer.train()
        
        # 12. Evaluate Model
        print("Running final evaluation...")
        results = trainer.evaluate()
        print(f"Evaluation Results: {results}")

        # 13. Save Fine-tuned Model and Tokenizer
        print(f"Saving fine-tuned model and tokenizer to {SAVE_PATH}...")
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(SAVE_PATH)
        tokenizer.save_pretrained(SAVE_PATH)
        print("Model saved successfully!")
    else:
        print("Training execution skipped.")

if __name__ == "__main__":
    main()
