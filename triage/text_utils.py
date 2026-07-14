import re
import hashlib

def clean_text(text: str) -> str:
    """
    Cleans raw email body content by:
    1. Stripping HTML tags.
    2. Removing quoted reply chains ("On ... wrote:" and similar markers).
    3. Stripping common signature blocks.
    4. Normalizing whitespace.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Strip HTML tags
    cleaned = re.sub(r'<[^>]+>', ' ', text)
    
    # 2. Remove reply chains
    # Common email reply patterns:
    # - "On Mon, Oct 23, 2023 at 10:00 AM User <user@example.com> wrote:"
    # - "-----Original Message-----"
    # - "From: user@example.com" (usually indicates forwarded/reply block start)
    reply_patterns = [
        r'(?i)\bOn\s+.*\s+wrote\s*:\s*',
        r'(?i)-+\s*Original Message\s*-+',
        r'(?i)^From:\s+\S+',
        r'(?i)^---+\s*Forwarded message\s*---+'
    ]
    
    for pattern in reply_patterns:
        # Split text on pattern and keep the first part (pre-reply)
        parts = re.split(pattern, cleaned, maxsplit=1, flags=re.MULTILINE)
        if len(parts) > 1:
            cleaned = parts[0]
            
    # 3. Strip common signature blocks
    # Common signature starts: "--", "-- ", "Regards", "Best regards", "Sincerely", "Thanks", "Thank you"
    # Let's search for these near the bottom of the email.
    # We can split on signature delimiters if they occur towards the end of the text.
    sig_patterns = [
        r'\r?\n--\s*\r?\n',           # Standard signature delimiter "--"
        r'\r?\n---\s*\r?\n',          # "---" signature delimiter
        r'(?i)\r?\nRegards,\s*\r?\n',
        r'(?i)\r?\nBest\s+regards,\s*\r?\n',
        r'(?i)\r?\nSincerely,\s*\r?\n',
        r'(?i)\r?\nWarm\s+regards,\s*\r?\n',
        r'(?i)\r?\nThanks,\s*\r?\n',
        r'(?i)\r?\nThank\s+you,\s*\r?\n'
    ]
    
    for pattern in sig_patterns:
        parts = re.split(pattern, cleaned, flags=re.MULTILINE)
        # Only treat it as signature if it splits the text and the signature part is relatively short
        # (e.g. less than 30% of the total email or last 300 characters, to prevent false positives in body)
        if len(parts) > 1:
            potential_sig = parts[-1]
            if len(potential_sig) < 300 or len(potential_sig) < (0.3 * len(cleaned)):
                cleaned = "".join(parts[:-1]) # Keep everything except the last part
    
    # 4. Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

def generate_text_hash(text: str) -> str:
    """
    Computes a stable SHA-256 hash of the cleaned text to act as a unique, stable row_id.
    """
    cleaned = clean_text(text)
    return hashlib.sha256(cleaned.encode('utf-8')).hexdigest()
