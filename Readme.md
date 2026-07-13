# BERT SMS Spam Classifier

This project fine-tunes a pretrained **BERT (Bidirectional Encoder Representations from Transformers)** model to classify SMS messages as either **HAM** (normal messages) or **SPAM** (unsolicited or malicious messages).

> [!IMPORTANT]
> **Architecture Note**: This project fine-tunes a pretrained BERT Transformer model (`bert-base-uncased`) using Hugging Face's `BertForSequenceClassification` API. It **does not** implement the Transformer architecture from scratch.

---

## Project Overview

The classifier is trained on the popular **SMS Spam Collection dataset** using Python, PyTorch, and Hugging Face Transformers. 

### Key Features
* **Transformer-based Classification**: Leverages pretrained semantic embeddings from `bert-base-uncased` for sequence classification.
* **Beginner-Friendly Notebooks**: Clear step-by-step guides for training (`train.ipynb`) and prediction/evaluation (`predict.ipynb`).
* **High Performance**: Achieves **99.37%** validation accuracy on the held-out test split of the SMS Spam dataset.
* **Detailed Evaluation**: Computes classification metrics, including the **Confusion Matrix** and **Classification Report** (Precision, Recall, F1-Score).
* **Interactive Inference**: Allows testing custom inputs interactively inside the prediction notebook.

---

## Tech Stack

* **Programming Language**: Python 3.12+
* **Deep Learning Framework**: PyTorch
* **NLP & Transformers**: Hugging Face Transformers (Tokenizer & Trainer APIs), Accelerate
* **Machine Learning & Data Science**: Scikit-Learn, Pandas, NumPy
* **Environment**: Jupyter Notebook / VS Code

---

## Dataset Information

The model uses the **SMS Spam Collection dataset**, which contains 5,572 English SMS messages.

### Data Preprocessing & Label Mapping
The raw dataset (`dataset/spam.csv`) contains columns (`v1`, `v2`, and several unnamed columns). During cleaning:
1. Only columns `v1` (label) and `v2` (text) are retained.
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
│   ├── saved_model/              # Fine-tuned model checkpoints & configs
│   │   ├── config.json           # Model configuration
│   │   ├── model.safetensors     # Saved model weights (~438 MB, ignored in git)
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   ├── tokenizer.json
│   │   └── vocab.txt             # Vocabulary list for BERT
│   │
│   ├── train.ipynb               # Notebook showing the fine-tuning workflow
│   └── predict.ipynb             # Notebook for inference and test set evaluation
│
├── .gitignore                    # Git exclusions (ignores Pycache, venv, large weights)
├── Readme.md                     # Project documentation
└── requirements.txt              # Pinned Python package dependencies
```

---

## Installation & Setup

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

4. **Verify Environment**:
   Ensure you have access to a Jupyter kernel:
   ```bash
   python -m ipykernel install --user --name=venv --display-name "Python (venv)"
   ```

---

## How to Run the Notebooks

### 1. Model Training (`model/train.ipynb`)
Open `model/train.ipynb` in VS Code or Jupyter Notebook. 
* **Do not rerun the training cell (`trainer.train()`)** unless you explicitly wish to overwrite the model.
* The fine-tuned weights took **73 minutes** to train on CPU and are saved in `model/saved_model`.

### 2. Predict & Evaluate (`model/predict.ipynb`)
Open `model/predict.ipynb` to evaluate the saved model and test predictions.
* **Inference Workaround**: Due to a known issue loading the local tokenizer configuration (which triggers a `JSONDecodeError`), the notebook loads the tokenizer directly from `bert-base-uncased` while loading model weights from the local `model/saved_model/` directory:
  ```python
  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
  model = BertForSequenceClassification.from_pretrained("model/saved_model")
  ```
* Run all cells to generate prediction results, display the confusion matrix, and view the classification report.

---

## Model Performance

The final validation metrics on the held-out test split (1,115 samples) are:

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

### Prediction Examples
* **Input**: *"Congratulations! You have won 50000 rupees. Click here to claim your prize."*
  * **Prediction**: `SPAM` (Confidence: `99.97%`)
* **Input**: *"Hey Harsh, are you coming to college tomorrow?"*
  * **Prediction**: `HAM` (Confidence: `99.99%`)

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
