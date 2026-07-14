import os
import csv
import base64
import argparse
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from text_utils import clean_text

# Define scopes required to read and modify labels
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.modify'
]

def get_gmail_service():
    """
    Handles Gmail API authentication and returns the service object.
    Uses token.json if it exists, otherwise performs the InstalledAppFlow
    using credentials.json and saves the credentials to token.json.
    """
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    TOKEN_PATH = PROJECT_ROOT / "token.json"
    CREDENTIALS_PATH = PROJECT_ROOT / "credentials.json"
    
    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
        
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing expired Gmail API credentials...")
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"Error refreshing credentials: {e}")
                creds = None
                
        if not creds:
            if not CREDENTIALS_PATH.exists():
                print("=" * 80)
                print("⚠️  ERROR: credentials.json is missing!")
                print("=" * 80)
                print(f"Expected location: {CREDENTIALS_PATH}")
                print("\nHow to resolve:")
                print("1. Set up a Desktop Application client in your Google Cloud Console.")
                print("2. Download the client secret JSON file.")
                print("3. Rename it to 'credentials.json' and place it in the project root.")
                print("4. Make sure 'credentials.json' is added to '.gitignore'.")
                print("=" * 80)
                raise FileNotFoundError("credentials.json not found.")
                
            print("Starting authentication flow... Please log in via the browser window.")
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
            creds = flow.run_local_server(port=0)
            
        with open(TOKEN_PATH, 'w') as token_file:
            token_file.write(creds.to_json())
            print(f"Saved authentication token to {TOKEN_PATH}")
            
    return build('gmail', 'v1', credentials=creds)

def extract_body(payload) -> str:
    """
    Recursively extracts the plain text or HTML body from Gmail message payload.
    """
    body = ""
    if 'parts' in payload:
        for part in payload['parts']:
            mime_type = part.get('mimeType')
            body_data = part.get('body', {}).get('data')
            if mime_type == 'text/plain' and body_data:
                body += base64.urlsafe_b64decode(body_data).decode('utf-8', errors='ignore')
            elif mime_type == 'text/html' and body_data and not body:
                # Use HTML body as fallback if text/plain is not present
                body += base64.urlsafe_b64decode(body_data).decode('utf-8', errors='ignore')
            elif 'parts' in part:
                body += extract_body(part)
    else:
        body_data = payload.get('body', {}).get('data')
        if body_data:
            body += base64.urlsafe_b64decode(body_data).decode('utf-8', errors='ignore')
            
    return body

def load_existing_ids(csv_path: Path) -> set:
    """
    Loads all existing gmail_message_id values from the CSV.
    """
    ids = set()
    if csv_path.exists():
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and 'gmail_message_id' in reader.fieldnames:
                for row in reader:
                    ids.add(row['gmail_message_id'])
    return ids

def main():
    parser = argparse.ArgumentParser(description="Fetch recent emails from Gmail and save to dataset/emails.csv.")
    parser.add_argument('--limit', type=int, default=150, help="Maximum number of emails to fetch.")
    parser.add_argument('--query', type=str, default="category:primary", help="Gmail search query filter.")
    args = parser.parse_args()
    
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATASET_DIR = PROJECT_ROOT / "dataset"
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    CSV_PATH = DATASET_DIR / "emails.csv"
    
    print(f"Initializing Gmail Service with query='{args.query}' and limit={args.limit}...")
    try:
        service = get_gmail_service()
    except Exception as e:
        print(f"Authentication failed: {e}")
        return
        
    existing_ids = load_existing_ids(CSV_PATH)
    print(f"Found {len(existing_ids)} existing messages in {CSV_PATH.name}")
    
    try:
        # List recent messages matching query
        print("Fetching message list from Gmail...")
        messages_result = service.users().messages().list(
            userId='me',
            q=args.query,
            maxResults=min(args.limit, 500)
        ).execute()
        
        messages = messages_result.get('messages', [])
        if not messages:
            print("No messages found matching the query.")
            return
            
        print(f"Found {len(messages)} messages on Gmail. Filtering duplicates...")
        
        new_messages = [msg for msg in messages if msg['id'] not in existing_ids]
        print(f"Total new messages to fetch details for: {len(new_messages)}")
        
        if not new_messages:
            print("All messages are already present in the dataset.")
            return
            
        # Prepare CSV file
        file_is_new = not CSV_PATH.exists()
        with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['gmail_message_id', 'subject', 'body'])
            if file_is_new:
                writer.writeheader()
                
            fetched_count = 0
            for idx, msg_info in enumerate(new_messages):
                msg_id = msg_info['id']
                if fetched_count >= args.limit:
                    break
                try:
                    # Retrieve the full message details
                    print(f"[{idx+1}/{len(new_messages)}] Fetching message {msg_id}...")
                    msg = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
                    payload = msg.get('payload', {})
                    
                    # Extract headers
                    headers = payload.get('headers', [])
                    subject = ""
                    for header in headers:
                        if header.get('name') == 'Subject':
                            subject = header.get('value')
                            break
                            
                    # Extract body and clean it
                    raw_body = extract_body(payload)
                    cleaned_body = clean_text(raw_body)
                    
                    # Write row to CSV
                    writer.writerow({
                        'gmail_message_id': msg_id,
                        'subject': subject,
                        'body': cleaned_body
                    })
                    fetched_count += 1
                except Exception as ex:
                    print(f"Failed to fetch details for message {msg_id}: {ex}")
                    
        print(f"\nSuccessfully fetched and appended {fetched_count} new emails to {CSV_PATH}")
        
    except HttpError as error:
        print(f"An error occurred while listing or fetching messages: {error}")

if __name__ == "__main__":
    main()
