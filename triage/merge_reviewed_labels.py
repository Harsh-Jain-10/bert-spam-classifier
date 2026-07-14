import argparse
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Merge manually reviewed emails into the main triage labeled dataset.")
    parser.add_argument('--labeled-csv', type=str, default='dataset/triage_labeled.csv', help="Path to main labeled CSV.")
    parser.add_argument('--review-csv', type=str, default='dataset/triage_sample_review.csv', help="Path to reviewed samples CSV.")
    args = parser.parse_args()
    
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    LABELED_PATH = PROJECT_ROOT / args.labeled_csv
    REVIEW_PATH = PROJECT_ROOT / args.review_csv
    
    if not REVIEW_PATH.exists():
        print(f"No review file found at {REVIEW_PATH}. Nothing to merge.")
        return
        
    print(f"Reading reviews from {REVIEW_PATH}...")
    df_review = pd.read_csv(REVIEW_PATH)
    
    if df_review.empty:
        print("Review file is empty. Nothing to merge.")
        return
        
    # Check if 'reviewed' column exists
    if 'reviewed' not in df_review.columns:
        print("⚠️ Warning: 'reviewed' column not found in review CSV. All rows will be treated as unreviewed.")
        return
        
    # Parse reviewed column as boolean
    # Handle string values like "True", "true", "1"
    df_review['reviewed_bool'] = df_review['reviewed'].apply(
        lambda x: str(x).strip().lower() in ['true', '1', 'yes', 't'] if pd.notna(x) else False
    )
    
    reviewed_rows = df_review[df_review['reviewed_bool'] == True].copy()
    pending_rows = df_review[df_review['reviewed_bool'] == False].copy()
    
    # Drop temp boolean column
    reviewed_rows.drop(columns=['reviewed_bool'], inplace=True)
    pending_rows.drop(columns=['reviewed_bool'], inplace=True)
    
    if reviewed_rows.empty:
        print("No rows found with reviewed=True. Please edit the CSV to mark reviewed rows first.")
        return
        
    print(f"Found {len(reviewed_rows)} reviewed rows to merge.")
    
    # Load or initialize the main labeled dataset
    df_labeled = pd.DataFrame()
    if LABELED_PATH.exists():
        df_labeled = pd.read_csv(LABELED_PATH)
        print(f"Loaded existing labeled dataset with {len(df_labeled)} rows.")
    else:
        print(f"Creating new labeled dataset file at {LABELED_PATH}...")
        
    # Prepare merged rows
    # For reviewed rows, we update source to 'manual_override'
    reviewed_rows['source'] = 'manual_override'
    # Drop the 'reviewed' column before merging to labeled since labeled dataset doesn't need it
    if 'reviewed' in reviewed_rows.columns:
        reviewed_rows.drop(columns=['reviewed'], inplace=True)
        
    # Merge/upsert logic
    if not df_labeled.empty:
        # Create indexed dictionaries/dataframes to update existing rows
        df_labeled.set_index('row_id', inplace=True)
        reviewed_rows.set_index('row_id', inplace=True)
        
        # Update existing records
        df_labeled.update(reviewed_rows)
        
        # Find completely new records that don't exist in labeled yet
        new_indices = reviewed_rows.index.difference(df_labeled.index)
        if len(new_indices) > 0:
            df_new_records = reviewed_rows.loc[new_indices]
            df_labeled = pd.concat([df_labeled, df_new_records])
            
        df_labeled.reset_index(inplace=True)
        # Restore index in reviewed_rows for later logging
        reviewed_rows.reset_index(inplace=True)
    else:
        df_labeled = reviewed_rows
        
    # Save the updated labeled dataset
    df_labeled.to_csv(LABELED_PATH, index=False)
    print(f"Successfully merged reviewed rows. Main dataset now has {len(df_labeled)} items.")
    
    # Save remaining pending unreviewed rows back to review file
    pending_rows.to_csv(REVIEW_PATH, index=False)
    print(f"Updated {REVIEW_PATH.name}: {len(pending_rows)} pending rows remaining.")

if __name__ == "__main__":
    main()
