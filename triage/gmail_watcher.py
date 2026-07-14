import os
import argparse
import requests
from pathlib import Path
from apscheduler.schedulers.blocking import BlockingScheduler
from googleapiclient.errors import HttpError
from fetch_emails import get_gmail_service, extract_body
from text_utils import clean_text

# FastAPI endpoint
CLASSIFY_URL = "http://localhost:8000/classify"

# Valid triage labels
TRIAGE_LABELS = ["needs_reply", "fyi", "newsletter", "cold_outreach", "personal", "spam"]

# Cache to avoid listing labels repeatedly
LABEL_IDS_CACHE = {}

def get_or_create_label_id(service, label_name):
    """
    Returns the Gmail label ID for a given label name.
    If the label does not exist, it is created.
    """
    if label_name in LABEL_IDS_CACHE:
        return LABEL_IDS_CACHE[label_name]
        
    try:
        results = service.users().labels().list(userId='me').execute()
        labels = results.get('labels', [])
        
        for label in labels:
            if label['name'].lower() == label_name.lower():
                LABEL_IDS_CACHE[label_name] = label['id']
                return label['id']
                
        # Create label if not found
        print(f"Label '{label_name}' not found on Gmail. Creating label...")
        label_body = {
            'name': label_name,
            'labelListVisibility': 'labelShow',
            'messageListVisibility': 'labelShow'
        }
        created_label = service.users().labels().create(userId='me', body=label_body).execute()
        label_id = created_label['id']
        LABEL_IDS_CACHE[label_name] = label_id
        return label_id
    except HttpError as error:
        print(f"Error retrieving/creating label '{label_name}': {error}")
        return None

def poll_and_classify_emails(service, dry_run):
    """
    Polls Gmail for new/unread emails that have not yet been categorized.
    Classifies them using FastAPI and applies the appropriate Gmail label.
    """
    print("\n--- Starting Gmail Poll Cycle ---")
    
    # Gmail query: matches unread emails that DO NOT have any of the 6 triage labels applied
    exclude_labels_query = " ".join([f"-label:{lbl}" for lbl in TRIAGE_LABELS])
    query = f"is:unread {exclude_labels_query}"
    
    try:
        messages_result = service.users().messages().list(userId='me', q=query).execute()
        messages = messages_result.get('messages', [])
        
        if not messages:
            print("No new unread uncategorized emails found.")
            return
            
        print(f"Found {len(messages)} new email(s) to process.")
        
        for idx, msg_info in enumerate(messages):
            msg_id = msg_info['id']
            try:
                # 1. Fetch full message details
                msg = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
                payload = msg.get('payload', {})
                
                # Extract Subject
                headers = payload.get('headers', [])
                subject = ""
                for header in headers:
                    if header.get('name') == 'Subject':
                        subject = header.get('value')
                        break
                        
                # Extract and clean body
                raw_body = extract_body(payload)
                cleaned_body = clean_text(raw_body)
                
                # Format to match classifier input schema
                combined_text = f"Subject: {subject}\n\nBody: {cleaned_body}"
                
                # 2. Call FastAPI endpoint
                try:
                    response = requests.post(CLASSIFY_URL, json={"text": combined_text}, timeout=10)
                    response.raise_for_status()
                    result = response.json()
                    predicted_label = result["label"]
                    confidence = result["confidence"]
                    print(f"[{idx+1}/{len(messages)}] Message {msg_id} (Subject: '{subject}') classified as '{predicted_label}' with confidence {confidence * 100:.1f}%")
                except requests.exceptions.RequestException as e:
                    print(f"[{idx+1}/{len(messages)}] ⚠️ Classifier API error for msg {msg_id}: {e}")
                    print("Ensure that serve.py is running via: uvicorn triage.serve:app --reload")
                    continue
                
                # 3. Apply Gmail Label (if not dry run)
                if dry_run:
                    print(f"  [DRY RUN] Would apply label '{predicted_label}' to message {msg_id}")
                else:
                    label_id = get_or_create_label_id(service, predicted_label)
                    if label_id:
                        service.users().messages().modify(
                            userId='me',
                            id=msg_id,
                            body={'addLabelIds': [label_id]}
                        ).execute()
                        print(f"  Applied label '{predicted_label}' successfully.")
                    else:
                        print(f"  ⚠️ Skipping label modification: Could not resolve label ID for '{predicted_label}'")
                        
            except Exception as ex:
                print(f"Error processing message {msg_id}: {ex}")
                
    except HttpError as error:
        print(f"An error occurred during Gmail API polling: {error}")
    print("--- Gmail Poll Cycle Complete ---")

def main():
    parser = argparse.ArgumentParser(description="Gmail Poll Watcher for Email Triage.")
    parser.add_argument('--interval', type=int, default=5, help="Polling interval in minutes (default: 5).")
    parser.add_argument('--dry-run', action='store_true', help="Log prediction results without applying labels in Gmail.")
    args = parser.parse_args()
    
    print("=" * 80)
    print("🤖 STARTING GMAIL EMAIL TRIAGE WATCHER DAEMON")
    print(f"  - Polling interval: {args.interval} minute(s)")
    print(f"  - Dry-run mode: {'ON' if args.dry_run else 'OFF'}")
    print("=" * 80)
    
    # 1. Initialize service and check auth on start
    try:
        service = get_gmail_service()
    except Exception as e:
        print(f"Authentication failed: {e}")
        return
        
    # 2. Setup Scheduler
    scheduler = BlockingScheduler()
    
    # Run once immediately on start
    poll_and_classify_emails(service, args.dry_run)
    
    # Schedule recurring job
    scheduler.add_job(
        poll_and_classify_emails,
        'interval',
        args=[service, args.dry_run],
        minutes=args.interval,
        id='gmail_triage_watcher'
    )
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\nStopping Gmail Email Triage Watcher. Goodbye!")

if __name__ == "__main__":
    main()
