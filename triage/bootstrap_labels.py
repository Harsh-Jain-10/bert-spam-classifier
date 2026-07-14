import os
import csv
import json
import argparse
from pathlib import Path
import pandas as pd
from groq import Groq
from text_utils import generate_text_hash, clean_text

def classify_email_via_llm(client, email_text: str) -> tuple:
    """
    Calls Groq API using tool calling to classify the email text.
    Returns (label, confidence_score).
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "classify_email",
                "description": "Submits the classification label and confidence score for an email.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "enum": ["needs_reply", "fyi", "newsletter", "cold_outreach", "personal", "spam"],
                            "description": "The category of the email."
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence score between 0.0 and 1.0 indicating classifier certainty."
                        }
                    },
                    "required": ["label", "confidence"]
                }
            }
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": f"Please classify this email into one of the 6 categories: needs_reply, fyi, newsletter, cold_outreach, personal, spam.\n\nEmail Text:\n{email_text}"
                }
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "classify_email"}},
            temperature=0.0
        )
        
        message = response.choices[0].message
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            result = json.loads(tool_call.function.arguments)
            return result["label"], float(result["confidence"])
            
        return "spam", 0.0
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return "spam", 0.0 # Default fallback on API error

def main():
    parser = argparse.ArgumentParser(description="Bootstrap multi-class labels for raw emails using Groq API.")
    parser.add_argument('--input', type=str, default='dataset/emails.csv', help="Path to raw emails CSV.")
    parser.add_argument('--min-confidence', type=float, default=0.7, help="Minimum confidence threshold for auto-labeling.")
    parser.add_argument('--sample-review', type=int, default=0, help="Number of random rows per class to export for manual review.")
    args = parser.parse_args()
    
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    INPUT_PATH = PROJECT_ROOT / args.input
    LABELED_PATH = PROJECT_ROOT / "dataset" / "triage_labeled.csv"
    REVIEW_PATH = PROJECT_ROOT / "dataset" / "triage_sample_review.csv"
    
    # Check for GROQ_API_KEY
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("=" * 80)
        print("⚠️  ERROR: GROQ_API_KEY environment variable is not set!")
        print("=" * 80)
        print("Please set the environment variable and try again.")
        print("Example (Windows PowerShell): $env:GROQ_API_KEY='your-key'")
        print("=" * 80)
        return
        
    if not INPUT_PATH.exists():
        print(f"⚠️  ERROR: Input file not found at {INPUT_PATH}.")
        print("Please run 'triage/fetch_emails.py' first or provide a valid CSV input.")
        return
        
    print(f"Loading raw emails from {INPUT_PATH}...")
    df_emails = pd.read_csv(INPUT_PATH)
    required_cols = {'gmail_message_id', 'subject', 'body'}
    if not required_cols.issubset(df_emails.columns):
        raise ValueError(f"Input CSV must contain columns: {required_cols}")
        
    # Read existing labeled row_ids and review row_ids to skip duplicates
    existing_labeled_ids = set()
    df_existing_labeled = pd.DataFrame()
    if LABELED_PATH.exists():
        df_existing_labeled = pd.read_csv(LABELED_PATH)
        if 'row_id' in df_existing_labeled.columns:
            existing_labeled_ids = set(df_existing_labeled['row_id'].astype(str))
            
    existing_review_ids = set()
    df_existing_review = pd.DataFrame()
    if REVIEW_PATH.exists():
        df_existing_review = pd.read_csv(REVIEW_PATH)
        if 'row_id' in df_existing_review.columns:
            existing_review_ids = set(df_existing_review['row_id'].astype(str))
            
    print(f"Skipping emails already in labeled ({len(existing_labeled_ids)}) or review ({len(existing_review_ids)}) files.")
    
    # Process emails and compute hashes
    emails_to_process = []
    for _, row in df_emails.iterrows():
        gmail_id = str(row['gmail_message_id'])
        subject = str(row['subject']) if pd.notna(row['subject']) else ""
        body = str(row['body']) if pd.notna(row['body']) else ""
        
        # Combine subject and body
        combined_text = f"Subject: {subject}\n\nBody: {body}"
        row_id = generate_text_hash(combined_text)
        
        if row_id in existing_labeled_ids or row_id in existing_review_ids:
            continue
            
        emails_to_process.append({
            'row_id': row_id,
            'gmail_message_id': gmail_id,
            'text': combined_text
        })
        
    print(f"Total new emails to classify: {len(emails_to_process)}")
    if not emails_to_process:
        print("No new emails to bootstrap.")
    else:
        # Initialize Groq Client
        client = Groq(api_key=api_key)
        
        new_labeled = []
        new_low_confidence = []
        
        for idx, email in enumerate(emails_to_process):
            print(f"[{idx+1}/{len(emails_to_process)}] Classifying email {email['gmail_message_id']}...")
            label, confidence = classify_email_via_llm(client, email['text'])
            
            email_entry = {
                'row_id': email['row_id'],
                'gmail_message_id': email['gmail_message_id'],
                'text': email['text'],
                'label': label,
                'confidence': confidence,
            }
            
            if confidence >= args.min_confidence:
                email_entry['source'] = 'llm_bootstrap'
                new_labeled.append(email_entry)
                print(f"  -> High Confidence ({confidence:.2f}): {label}")
            else:
                email_entry['source'] = 'llm_low_confidence'
                email_entry['reviewed'] = False
                new_low_confidence.append(email_entry)
                print(f"  -> Low Confidence ({confidence:.2f}): Routed to review as {label}")
                
        # Write high confidence labels to triage_labeled.csv
        if new_labeled:
            df_new_labeled = pd.DataFrame(new_labeled)
            if not df_existing_labeled.empty:
                df_combined_labeled = pd.concat([df_existing_labeled, df_new_labeled], ignore_index=True)
            else:
                df_combined_labeled = df_new_labeled
            df_combined_labeled.to_csv(LABELED_PATH, index=False)
            print(f"Appended {len(new_labeled)} high-confidence items to {LABELED_PATH}")
            # Refresh existing labeled dataset for sampling later
            df_existing_labeled = df_combined_labeled
            
        # Write low confidence labels to triage_sample_review.csv
        if new_low_confidence:
            df_new_low = pd.DataFrame(new_low_confidence)
            if not df_existing_review.empty:
                df_combined_review = pd.concat([df_existing_review, df_new_low], ignore_index=True)
            else:
                df_combined_review = df_new_low
            # Deduplicate by row_id
            df_combined_review.drop_duplicates(subset=['row_id'], keep='first', inplace=True)
            df_combined_review.to_csv(REVIEW_PATH, index=False)
            print(f"Appended/deduplicated {len(new_low_confidence)} low-confidence items in {REVIEW_PATH}")
            df_existing_review = df_combined_review
            
    # Sample Review Export
    if args.sample_review > 0:
        if df_existing_labeled.empty:
            print("No high-confidence labels available to sample from.")
            return
            
        print(f"Exporting sample review (up to {args.sample_review} samples per class)...")
        # Exclude already manual_override rows
        df_pool = df_existing_labeled[df_existing_labeled['source'] != 'manual_override']
        
        sampled_rows = []
        for class_name in df_pool['label'].unique():
            df_class = df_pool[df_pool['label'] == class_name]
            sample_size = min(len(df_class), args.sample_review)
            if sample_size > 0:
                df_sample = df_class.sample(n=sample_size, random_state=42)
                for _, row in df_sample.iterrows():
                    sampled_rows.append({
                        'row_id': row['row_id'],
                        'gmail_message_id': row['gmail_message_id'],
                        'text': row['text'],
                        'label': row['label'],
                        'confidence': row['confidence'],
                        'source': 'sampled_for_review',
                        'reviewed': False
                    })
                    
        if sampled_rows:
            df_new_samples = pd.DataFrame(sampled_rows)
            # Merge with existing reviews
            if not df_existing_review.empty:
                # Make sure reviewed column exists
                if 'reviewed' not in df_existing_review.columns:
                    df_existing_review['reviewed'] = False
                df_final_review = pd.concat([df_existing_review, df_new_samples], ignore_index=True)
            else:
                df_final_review = df_new_samples
                
            # Deduplicate by row_id (keeping the first occurrence)
            df_final_review.drop_duplicates(subset=['row_id'], keep='first', inplace=True)
            df_final_review.to_csv(REVIEW_PATH, index=False)
            print(f"Exported sampled review rows. Total review queue size in {REVIEW_PATH}: {len(df_final_review)}")

if __name__ == "__main__":
    main()
