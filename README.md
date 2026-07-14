<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0F2027,100:2C5364&height=180&section=header&text=BERT%20Spam%20Classifier&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=35" width="100%"/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&pause=1000&color=61DBFB&background=00000000&center=true&vCenter=true&width=600&lines=Fine-tuned+BERT+for+SMS+Spam+Detection;99.37%25+Test+Accuracy;Binary+%2B+6-Class+Email+Triage+Extension" alt="Typing SVG" />

[![Python](https://img.shields.io/badge/PYTHON-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=FFD43B&labelColor=1a1a1a)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PYTORCH-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white&labelColor=EE4C2C)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗_TRANSFORMERS-FFD21E?style=for-the-badge&labelColor=FFD21E&color=FFD21E&logoColor=black)](https://huggingface.co/)
[![License](https://img.shields.io/badge/LICENSE-MIT-1a1a1a?style=for-the-badge&labelColor=1a1a1a&color=4CAF50)](LICENSE)

</div>

---

## 🚀 Overview

This project fine-tunes a pretrained **BERT** (`bert-base-uncased`) model — via Hugging Face's `BertForSequenceClassification` — to classify SMS messages as **HAM** or **SPAM**, and extends into a **6-class email triage system** powered by Gmail + Groq.

> ⚠️ **Note:** This fine-tunes a pretrained Transformer; it does **not** implement the Transformer architecture from scratch.

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 Transformer-based | Leverages pretrained `bert-base-uncased` semantic embeddings |
| ⚙️ Production Scripts | Portable `train.py` / `predict.py` — no notebook dependency |
| 💬 Interactive CLI | Real-time predictions from the terminal |
| 📊 Batch Evaluation | Accuracy, confusion matrix, full classification report |
| 🛡️ Safety Check | Warns on missing weights before Hugging Face throws long tracebacks |

## 🏆 Performance

<div align="center">

| Metric | Score |
|---|---|
| **Test Accuracy** | 🟢 **99.37%** (1,108 / 1,115) |
| **Eval Loss** | 0.0407 |
| **SPAM Precision / Recall** | 0.99 / 0.97 |
| **HAM Precision / Recall** | 0.99 / 1.00 |

</div>

<details>
<summary>📈 Full classification report & confusion matrix</summary>

```
Confusion Matrix:
[[963   2]  <- [True HAM, False SPAM]
 [  5 145]] <- [False HAM, True SPAM]

              precision    recall  f1-score   support
         HAM       0.99      1.00      1.00       965
        SPAM       0.99      0.97      0.98       150
    accuracy                           0.99      1115
```

</details>

## 🛠️ Tech Stack

`Python 3.12+` · `PyTorch` · `Hugging Face Transformers` · `Accelerate` · `Scikit-Learn` · `Pandas` · `NumPy`

## 📂 Dataset

**SMS Spam Collection** — 5,572 messages (`ISO-8859-1` encoded), mapped as:

- HAM → `0` (4,825 samples) &nbsp;|&nbsp; SPAM → `1` (747 samples)
- Split: **80/20** (`test_size=0.2`, `random_state=42`) → 4,457 train / 1,115 test

<details>
<summary>📁 Full project structure</summary>

```
bert-spam-classifier/
│
├── dataset/
│   └── spam.csv
│
├── model/
│   ├── results/                  # training checkpoints (gitignored)
│   └── saved_model/              # fine-tuned model + configs
│       ├── config.json
│       ├── model.safetensors     # ~438 MB (gitignored)
│       ├── special_tokens_map.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── vocab.txt
│   └── train.ipynb               # historical training notebook
│
├── train.py
├── predict.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

</details>

## ⚡ Quickstart

```bash
git clone <repository-url>
cd bert-spam-classifier

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> ⚠️ **Model weights not included** (`model.safetensors`, ~438 MB exceeds GitHub's 100 MB limit). Train locally via `train.py` or supply your own fine-tuned weights before running `predict.py`.

### Train
```bash
python train.py
```
*(~73 min on CPU — see `model/train.ipynb` for full epoch logs)*

### Predict
```bash
python predict.py
```
```text
Enter SMS Message: Congratulations! You have won 50000 rupees. Click here to claim your prize.
Prediction: SPAM   |   Confidence: 99.97%
```

### Evaluate
```bash
python predict.py --evaluate
```

## ⚠️ Known Limitations

- **Domain mismatch** — trained on short informal SMS text; produces false positives on longer, formal emails (e.g. recruitment mail with links/WhatsApp CTAs).
- **Text-only heuristic** — no sender reputation, SPF/DKIM/DMARC checks, or URL reputation signals.
- **Class imbalance** — 86% HAM vs 14% SPAM in training data.

## 🔮 Roadmap

- [ ] Domain adaptation on email corpora (e.g. Enron Spam)
- [ ] Multi-modal signals (sender verification, domain age, link safety)
- [ ] Distillation to DistilBERT / MobileBERT for edge deployment

---

## 📬 Multi-Class Email Triage Extension

A 6-class extension of the binary classifier, wired directly into Gmail.

**Classes:** `needs_reply` · `fyi` · `newsletter` · `cold_outreach` · `personal` · `spam`

<details>
<summary>🔄 View full pipeline & data flow</summary>

```
[Gmail API] ──(fetch_emails.py)──> [emails.csv]
                                          │
                                          ▼
                              (bootstrap_labels.py)
                             [Groq: llama-3.1-8b-instant]
                                          │
        ┌─────────── confidence ≥ 0.7 ───┴─── confidence < 0.7 / sampled ──────┐
        ▼                                                                      ▼
[triage_labeled.csv]                                            [triage_sample_review.csv]
        ▲                                                                      │
        └───────────── (merge_reviewed_labels.py) ◄──────────── [Manual Review]
                                          │
                                          ▼
                                (train_triage.py)
                                          │
                                          ▼
                          [saved_model_triage/] ◄──── (predict_triage.py)
                                          │
                                          ▼
                                   (serve.py — FastAPI)
                                          │
                                          ▼
                              (gmail_watcher.py — daemon)
```

**Pipeline steps:**
1. **Gmail Setup** — enable Gmail API, create OAuth Desktop credentials → `credentials.json`
2. **Fetch:** `python triage/fetch_emails.py --limit 150 --query "category:primary"`
3. **Bootstrap labels (Groq):** `python triage/bootstrap_labels.py --min-confidence 0.7 --sample-review 10`
4. **Human-in-the-loop correction:** edit `triage_sample_review.csv` → `python triage/merge_reviewed_labels.py`
5. **Train:** `python triage/train_triage.py --min-examples-per-class 100 --epochs 3`
6. **Predict/Evaluate:** `python triage/predict_triage.py [--evaluate]`
7. **Serve:** `uvicorn triage.serve:app --reload` → `POST /classify`
8. **Automate:** `python triage/gmail_watcher.py --interval 5 --dry-run`

</details>

---

<div align="center">

### 👨‍💻 Author

**Harsh Jain**

<sub>Data Science & AI Undergraduate · Python · Data Analytics</sub>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:2C5364,100:0F2027&height=3&width=100%" width="100%"/>

</div>
