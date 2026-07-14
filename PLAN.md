# Multi-Class Email Triage Classifier Plan (Revised v5)

This document outlines the final plan to extend the binary spam/ham BERT classifier into a multi-class email triage classifier with 6 labels: `needs_reply`, `fyi`, `newsletter`, `cold_outreach`, `personal`, `spam`.

---

## 1. Data Flow Overview
We will implement an end-to-end data pipeline from raw email fetching to model training, as follows:

```
[Gmail API] ──(fetch_emails.py with clean_text() & deduplication)──> [dataset/emails.csv (gmail_message_id, subject, body)]
                                                                               │
                                                                               ▼
                                                                     (bootstrap_labels.py)
                                                                [Groq API: llama-3.1-8b-instant]
                                                                               │
                 ┌─────────────────────────────┴─────────────────────────────┐
                 ▼ (confidence >= --min-confidence)                         ▼ (confidence < --min-confidence or sampled)
    [dataset/triage_labeled.csv]                               [dataset/triage_sample_review.csv]
(row_id, gmail_message_id, text, label,                      (row_id, gmail_message_id, text, label,
 confidence, source)                                          confidence, source, reviewed=False)
                 ▲                                                           │
                 │                                                           ▼
                 │                                                    [Manual Review]
                 │                                            (User sets label & reviewed=True)
                 │                                                           │
                 └─────────────── (merge_reviewed_labels.py) ◄───────────────┘
                                               │
                                               ▼
                                      (train_triage.py) ───> [model/saved_model_triage/test_row_ids.txt]
                                               │                                         │
                                               ▼                                         ▼
                                  [model/saved_model_triage/]                  (predict_triage.py --evaluate)
```

---

## 2. Updated Components

### Shared Utility (`text_utils.py`)
- **`clean_text(text)`**: A shared text preprocessing function.
  - Strips HTML tags.
  - Strips quoted reply chains (e.g., matching pattern `"On ... wrote:"` and everything below it).
  - Strips common email signature dividers (e.g., `"--"`, `"-- "`, `"Regards,"`, `"Best regards,"` followed by lines).
- **`generate_text_hash(text)`**: Helper function that computes a stable MD5 or SHA256 hash of the cleaned text to act as a unique, stable `row_id`.

### Phase 1.5 — Raw Email Extraction (`fetch_emails.py`)
- **Purpose**: Fetch recent emails from Gmail to construct our local training corpus.
- **Implementation**:
  - Uses Gmail API (OAuth2 Client flow) via `credentials.json` and a local gitignored `token.json`.
  - Queries recent messages. For each message:
    - Extracts the Gmail message ID (`gmail_message_id`), `subject`, and raw `body`.
    - Sanitizes the body using `clean_text()`.
  - **Deduplication**:
    - If `dataset/emails.csv` already exists, reads existing IDs from the `gmail_message_id` column.
    - Only appends messages whose `gmail_message_id` is not already present in the CSV.
  - Saves columns: `gmail_message_id`, `subject`, `body` to `dataset/emails.csv`.
  - Privacy precaution: Adds `dataset/emails.csv` to `.gitignore`.

### Phase 2 — Label Bootstrapping (`bootstrap_labels.py`)
- **Input**: `dataset/emails.csv`.
- **Row ID Generation**:
  - For each email, we combine the subject and cleaned body into a text block: `Subject: {subject}\n\nBody: {body}`.
  - Generates a stable unique hash of this combined text as the `row_id`.
  - The Gmail Message ID `gmail_message_id` is preserved as a separate column.
- **API Integration**:
  - Calls Groq Cloud API using the official `groq` Python library (reads `GROQ_API_KEY` from the environment).
  - Uses the free-tier instruction-following model `llama-3.1-8b-instant`.
  - Employs Groq's tool-calling / function-calling parameters to enforce structured JSON output.
- **Confidence Thresholding**:
  - Prompts Llama to classify the text into one of the 6 classes, returning `label` and `confidence` score.
  - Accepts a `--min-confidence` CLI flag (default: `0.7`).
  - Emails with confidence $\ge$ `--min-confidence` are saved directly to `dataset/triage_labeled.csv` (columns: `row_id`, `gmail_message_id`, `text`, `label`, `confidence`, `source="llm_bootstrap"`).
  - Emails with confidence $<$ `--min-confidence` are appended to `dataset/triage_sample_review.csv` (columns: `row_id`, `gmail_message_id`, `text`, `label`, `confidence`, `source="llm_low_confidence"`, `reviewed=False`).
- **Sample Review Trigger**:
  - If the `--sample-review` flag is specified, the script also selects a sample of $N$ random rows per label from those that passed the threshold.
  - **Exclusion check**: Excludes any rows that already have `source == "manual_override"` in `dataset/triage_labeled.csv` to avoid re-sampling previously-reviewed rows.
  - These rows are appended to `dataset/triage_sample_review.csv` with `reviewed=False`.
  - **File Collision & Deduping**: When writing/appending to `dataset/triage_sample_review.csv`, the script will read existing entries first, append the new low-confidence/sampled rows, and deduplicate the list of rows based on `row_id`.

### Phase 2.5 — Closing the Review Loop (`merge_reviewed_labels.py`)
- **Purpose**: Merge reviewed/corrected labels back into the main training set.
- **Implementation**:
  - Reads `dataset/triage_sample_review.csv`.
  - Reads `dataset/triage_labeled.csv`.
  - Identifies rows in `dataset/triage_sample_review.csv` where the column `reviewed` is `True`.
  - Matches these reviewed rows against `dataset/triage_labeled.csv` using the stable `row_id` column.
  - For each match:
    - Overwrites/updates the row in `dataset/triage_labeled.csv` with the manual classification `label` and sets `source="manual_override"`.
    - If the row does not exist in `dataset/triage_labeled.csv`, it is inserted as a new row.
  - Deletes the merged rows (where `reviewed` is `True`) from `dataset/triage_sample_review.csv`, keeping other pending unreviewed rows intact.

### Phase 3 — Multi-class Training (`train_triage.py`)
- **Pre-Training Checks**:
  - Counts examples per class in the final post-correction `dataset/triage_labeled.csv`.
  - Aborts with an error if any class has fewer than a minimum threshold (default: 100, configurable via `--min-examples-per-class` CLI flag).
  - Prints the per-class counts before proceeding to fine-tuning.
- **Implementation**:
  - Adapt `train.py` but use `BertForSequenceClassification` with `num_labels=6`.
  - Map labels to IDs:
    `needs_reply` (0), `fyi` (1), `newsletter` (2), `cold_outreach` (3), `personal` (4), `spam` (5).
  - Save `id2label` and `label2id` in the model configuration.
  - Perform stratified train/test split (`test_size=0.2`, `random_state=42`).
  - **Save Test Split split Row IDs**: Save the `row_id` values of the test split to `model/saved_model_triage/test_row_ids.txt` to eliminate evaluation data leakage.
  - Save model checkpoints to `model/saved_model_triage/`.
  - Print class-wise precision, recall, and F1-score classification report after training.

### Phase 4 — Prediction Script (`predict_triage.py`)
- **Input**: Text input or `--evaluate` batch testing.
- **`--evaluate` Mode**:
  - Loads the test split row IDs from `model/saved_model_triage/test_row_ids.txt`.
  - Reads `dataset/triage_labeled.csv` and filters it to include only rows with `row_id` matching those in `test_row_ids.txt`.
  - Runs batch evaluation on this exact held-out set and outputs precision/recall/F1 metrics.
- **Interactive Mode**: Returns predicted class and confidence scores for all 6 classes.

### Phase 5 — Serving Endpoint (`serve.py`)
- Minimal FastAPI server with `POST /classify` returning class probabilities and prediction label.

### Phase 6 — Gmail Polling Watcher (`gmail_watcher.py`)
- **Loop Avoidance / Label Filtering**:
  - Gmail API query will explicitly exclude messages that already have any of the 6 triage labels (`needs_reply`, `fyi`, `newsletter`, `cold_outreach`, `personal`, `spam`) applied.
  - Example Gmail search query: `is:unread -label:needs_reply -label:fyi -label:newsletter -label:cold_outreach -label:personal -label:spam`.
  - Runs on an APScheduler interval (default 5 min).

---

## 3. Privacy & Repository Safety
The following files containing actual emails or credentials will be excluded from the repository via `.gitignore`:
- `dataset/emails.csv`
- `dataset/triage_labeled.csv`
- `dataset/triage_sample_review.csv`
- `credentials.json`
- `token.json`

We will commit example templates instead:
- `dataset/emails.csv.example`
- `dataset/triage_labeled.csv.example`
- `credentials.json.example`
