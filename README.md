# BERT SMS Spam Classifier

This project fine-tunes a pretrained **BERT (Bidirectional Encoder Representations from Transformers)** model to classify SMS messages as either **HAM** (normal messages) or **SPAM** (unsolicited or malicious messages).

> [!IMPORTANT]
> **Architecture Note**: This project fine-tunes a pretrained BERT Transformer model (`bert-base-uncased`) using Hugging Face's `BertForSequenceClassification` API. It **does not** implement the Transformer architecture from scratch.

---

## Project Overview

The classifier is trained on the popular **SMS Spam Collection dataset** using Python, PyTorch, and Hugging Face Transformers. 

### Key Features
* **Transformer-based Classification**: Leverages pretrained semantic embeddings from `bert-base-uncased` for sequence classification.
* **Production-style Scripts**: Replaced Jupyter-based prediction workflows with portable Python scripts (`train.py` and `predict.py`).
* **Interactive Prediction**: Execute interactive predictions inside the terminal using a simple command shell.
* **Optional Batch Evaluation**: Evaluate the fine-tuned model on the held-out test split, reporting Accuracy, Confusion Matrix, and a full Classification Report.
* **Ignored Weights Safety**: Pre-inference check to notify users if large weights are missing before Hugging Face raises long tracebacks.

---

## Tech Stack

* **Programming Language**: Python 3.12+
* **Deep Learning Framework**: PyTorch
* **NLP & Transformers**: Hugging Face Transformers, Accelerate
* **Machine Learning & Data Science**: Scikit-Learn, Pandas, NumPy

---

## Dataset Information

The model uses the **SMS Spam Collection dataset**, which contains 5,572 English SMS messages.

### Data Preprocessing & Label Mapping
The raw dataset (`dataset/spam.csv`) is loaded using `ISO-8859-1` encoding. During cleaning:
1. Columns `v1` (label) and `v2` (text) are retained.
2. Columns are renamed to `label` and `text`.
3. Labels are mapped as follows:
   * **HAM** $\rightarrow$ `0` (4,825 samples)
   * **SPAM** $\rightarrow$ `1` (747 samples)

### Train/Test Split
* **Train Set**: 4,457 samples (80%)
* **Test Set (Held-Out)**: 1,115 samples (20%)
* **Split Configuration**: `test_size=0.2`, `random_state=42`

---

## Project Structure

```
bert-spam-classifier/
│
├── dataset/
│   └── spam.csv                  # Raw SMS spam dataset
│
├── model/
│   ├── results/                  # Training checkpoints (ignored in git)
│   └── saved_model/              # Fine-tuned model checkpoints & configs
│       ├── config.json           # Model configuration
│       ├── model.safetensors     # Saved model weights (~438 MB, ignored in git)
│       ├── special_tokens_map.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── vocab.txt             # Vocabulary list for BERT
│   │
│   └── train.ipynb               # Historical training notebook with preserved outputs
│
├── train.py                      # Script containing the reproducible training pipeline
├── predict.py                    # Main script for prediction and evaluation
│
├── .gitignore                    # Git exclusions (ignores Pycache, venv, model weights)
├── LICENSE                       # Repository LICENSE file
├── README.md                     # Project documentation
└── requirements.txt              # Pinned Python package dependencies
```

---

## Installation & Setup

> [!WARNING]
> **Missing Model Weights Warning**: 
> * The fine-tuned BERT model weights (`model/saved_model/model.safetensors`, ~438 MB) are **intentionally excluded** from this Git repository. This is because they exceed GitHub's standard 100 MB per-file tracking limit.
> * As the trained weights are not currently hosted on a public repository (e.g., Hugging Face Hub), the prediction script `predict.py` is **not immediately runnable after cloning** unless you train the model locally or obtain and place the fine-tuned `model.safetensors` file into the `model/saved_model/` directory.

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd bert-spam-classifier
   ```

2. **Set Up a Virtual Environment** (Recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run the Scripts

### 1. Model Training (`train.py`)
Run the training pipeline script from the project root:
```bash
python train.py
```
* **Note**: Training fine-tunes BERT and is computationally expensive. The original training loop took approximately **73 minutes** on CPU.
* To view the historical output logs and epoch-by-epoch training details, open `model/train.ipynb`.

### 2. Predict Spam Interactively (`predict.py`)
Run the interactive command-line interface:
```bash
python predict.py
```
* **Interactive Prompt**: The script prompts `Enter SMS Message:` and outputs classifications instantly:
  ```text
  Enter SMS Message: Congratulations! You have won 50000 rupees. Click here to claim your prize.
  Prediction: SPAM
  Confidence: 99.97%
  ```

### 3. Evaluate Metrics (`predict.py --evaluate`)
To run a detailed batch evaluation on the held-out test split:
```bash
python predict.py --evaluate
```
* This script splits the dataset, tokenizes the test messages, and outputs the final metrics.

---

## Model Performance

The final validation metrics on the held-out test split (1,115 samples) calculated from the saved model:

* **Evaluation Loss**: `0.04068`
* **Test Accuracy**: **99.37%** (1,108 out of 1,115 samples correctly classified)

### Confusion Matrix
```
[[963   2]  <- [True HAM, False SPAM]
 [  5 145]] <- [False HAM, True SPAM]
```

### Classification Report
```
              precision    recall  f1-score   support

         HAM       0.99      1.00      1.00       965
        SPAM       0.99      0.97      0.98       150

    accuracy                           0.99      1115
   macro avg       0.99      0.98      0.99      1115
weighted avg       0.99      0.99      0.99      1115
```

---

## Known Model Limitations

> [!WARNING]
> While the model performs exceptionally well on the test dataset (99.37% accuracy), it is **not production-ready** for general email/message spam filtering and exhibits specific limitations:

1. **Domain Mismatch (SMS vs. Email)**:
   * The training dataset represents SMS messages, which are typically short, direct, and contain informal abbreviations (e.g., *u, dun, lar*).
   * Longer formal emails, newsletters, or rich-text promotional mailings represent a different distribution. Testing the model on longer formal recruitment emails (e.g. from *InternsElite*) inviting users to *"JOIN WHATSAPP GROUP"* or *"Click on the Link"* yields **false positives** (classifying legitimate formal communication as spam).
2. **Text-Only Heuristic**:
   * The classifier makes decisions **solely** based on text patterns.
   * Unlike production-grade filters, it does not analyze:
     * **Sender Identity**: Sender email address, domain registration date, or reputation scores.
     * **Domain Authentication**: Validation of SPF, DKIM, or DMARC records.
     * **URL Reputation**: Blacklists or verification of embedded links.
     * **History**: User relationship, registration context, or past conversation threads.
3. **Imbalanced Training**:
   * The SMS Spam dataset has a massive imbalance (86% HAM vs. 14% SPAM). While accuracy is high, the model can struggle on highly ambiguous borderline promotional text.

---

## Future Improvements

* **Domain Adaptation**: Fine-tune the BERT model on a larger and more modern dataset containing email corpuses (e.g., Enron Spam dataset).
* **Multi-modal Signals**: Integrate metadata parameters (sender verification status, domain age, link destination safety checks) alongside text classification.
* **Model Distillation**: Distill the BERT model into a smaller, faster model (like DistilBERT or MobileBERT) for low-latency edge deployments.
