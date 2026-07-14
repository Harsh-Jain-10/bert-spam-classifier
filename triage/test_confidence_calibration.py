import os
import sys
import json
import pandas as pd
from pathlib import Path
from groq import Groq
from bootstrap_labels import classify_email_via_llm

def get_calibration_emails():
    """
    Returns 30 synthetic emails: 10 clear cases, 10 borderline/ambiguous cases, and 10 spam/junk.
    """
    return [
        # Clear Cases (10)
        {"text": "Subject: Urgently need your reply on Q3 goals\n\nBody: Hi team, please look at the Q3 goals document and send me your feedback by tomorrow end of day. I need to submit it to the board.", "label": "needs_reply"},
        {"text": "Subject: Re: Meeting schedule update\n\nBody: Sure, that time works for me. See you tomorrow at 2 PM.", "label": "needs_reply"},
        {"text": "Subject: Receipt for your order #12839\n\nBody: Thank you for your purchase. Here is the receipt for your order of $49.99 on Amazon.", "label": "fyi"},
        {"text": "Subject: Weekly newsletter: Tech trends 2026\n\nBody: Welcome to our weekly digest. Here are the top 5 articles on AI and web development from this week.", "label": "newsletter"},
        {"text": "Subject: The Daily Product Manager Newsletter\n\nBody: Tips on product roadmap prioritization, stakeholder management, and design tokens.", "label": "newsletter"},
        {"text": "Subject: Partnership Proposal with your company\n\nBody: Dear Founder, I saw your product online and wanted to know if you'd be open to a partnership call next Tuesday.", "label": "cold_outreach"},
        {"text": "Subject: We are hiring - Software engineer roles\n\nBody: Hi Harsh, I am a recruiter at TechCorp. I came across your GitHub profile and would love to chat about open roles.", "label": "cold_outreach"},
        {"text": "Subject: Dinner this Sunday?\n\nBody: Hey, it's Mom. Just wanted to see if you are coming over for dinner this Sunday around 6 PM.", "label": "personal"},
        {"text": "Subject: Photos from our last trip\n\nBody: Hey! Here are the photos from our weekend trip to the mountains. Let me know which ones you like.", "label": "personal"},
        {"text": "Subject: CONGRATULATIONS! YOU HAVE WON A $1000 GIFT CARD!\n\nBody: CLAIM YOUR FREE GIFT CARD NOW BY CLICKING ON THIS LINK BEFORE IT EXPIRES!!!", "label": "spam"},
        
        # Borderline/Ambiguous Cases (10)
        {"text": "Subject: Notes from the sync\n\nBody: Hey, here are my quick notes from the meeting. Take a look when you have a moment.", "label": "fyi"}, # Borderline fyi/needs_reply
        {"text": "Subject: check this out\n\nBody: Hey, found this article online. Thought you might find it interesting.", "label": "personal"}, # Borderline personal/newsletter/fyi
        {"text": "Subject: Invoice overdue: Reminder #2\n\nBody: Your invoice #987 is now 15 days overdue. Please process the payment as soon as possible.", "label": "needs_reply"}, # Borderline fyi/needs_reply
        {"text": "Subject: Quick question about your code\n\nBody: Hi, I saw your repo. Can you explain why you used this loss function on line 45?", "label": "needs_reply"}, # Borderline personal/needs_reply/cold_outreach
        {"text": "Subject: New login detected on your account\n\nBody: We detected a new login from a Chrome browser on Windows. If this wasn't you, please change your password.", "label": "fyi"}, # Borderline fyi/needs_reply
        {"text": "Subject: Update on our terms of service\n\nBody: We are updating our privacy policy and terms of service starting next month. No action is required from you.", "label": "fyi"}, # Borderline fyi/newsletter
        {"text": "Subject: Are you looking to scale your engineering team?\n\nBody: Hi, we build offshore software teams. Let me know if you need React or Python devs.", "label": "cold_outreach"}, # Borderline cold_outreach/spam
        {"text": "Subject: Congratulations on your work anniversary!\n\nBody: LinkedIn: Wish John a happy work anniversary at TechCorp.", "label": "fyi"}, # Borderline fyi/personal
        {"text": "Subject: Feedback on your submission\n\nBody: Hi, your paper has been reviewed. The comments are attached. Let us know if you have questions.", "label": "fyi"}, # Borderline fyi/needs_reply
        {"text": "Subject: Subscribe to premium for more content\n\nBody: You have read 3 out of 5 free articles this month. Upgrade to our premium subscription for unlimited access.", "label": "newsletter"}, # Borderline newsletter/spam
        
        # Spam/Junk/Promotional (10)
        {"text": "Subject: Save 50% on Web Hosting services\n\nBody: Host your site starting at only $1/month. 24/7 support and free SSL. Limited time deal!", "label": "spam"},
        {"text": "Subject: Get cash in 24 hours\n\nBody: Need emergency loans? Fast approval, no credit checks. Apply now online.", "label": "spam"},
        {"text": "Subject: Lower your mortgage payments now\n\nBody: You qualify for a lower refinance rate. Click to compare current lender rates and save.", "label": "spam"},
        {"text": "Subject: Best replica watches online\n\nBody: High quality Swiss replica watches at affordable prices. Rolex, Omega, Breitling. Fast shipping.", "label": "spam"},
        {"text": "Subject: Increase your Google search rankings fast\n\nBody: We guarantee first page rankings on Google for your website. Reply for a free SEO audit.", "label": "spam"},
        {"text": "Subject: Earn $500/day working from home\n\nBody: Start earning part-time. No experience required. Register today for our training webinar.", "label": "spam"},
        {"text": "Subject: Stop snoring tonight with this device\n\nBody: Millions of people sleep better with our anti-snore nasal dilator. Order now and get 30% off.", "label": "spam"},
        {"text": "Subject: Stock alert: Buy XYZ before it explodes\n\nBody: Hot stock tip! XYZ is trading at $0.05 and is expected to reach $2.00 next week. Buy now!", "label": "spam"},
        {"text": "Subject: Your Amazon account has been suspended\n\nBody: Alert: Unusual activity detected. Please verify your billing details by clicking here to reactivate.", "label": "spam"},
        {"text": "Subject: Special discount just for you\n\nBody: Use promo code SAVE20 at checkout for 20% off all items in our online shop.", "label": "spam"}
    ]

def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    # Check for GROQ_API_KEY
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("=" * 80)
        print("⚠️  ERROR: GROQ_API_KEY environment variable is not set!")
        print("=" * 80)
        print("Please set the environment variable and run this script again.")
        print("Example (Windows PowerShell): $env:GROQ_API_KEY='your-key'")
        print("=" * 80)
        sys.exit(1)
        
    print("Initializing Groq Client...")
    client = Groq(api_key=api_key)
    
    emails = get_calibration_emails()
    print(f"Loaded {len(emails)} calibration test emails.")
    
    results = []
    
    print("\nStarting zero-shot classification via Llama-3.1-8b-instant on Groq...")
    for idx, item in enumerate(emails):
        print(f"[{idx+1}/{len(emails)}] Classifying...")
        pred_label, confidence = classify_email_via_llm(client, item['text'])
        results.append({
            'index': idx + 1,
            'true_label': item['label'],
            'predicted_label': pred_label,
            'confidence': confidence,
            'correct': (item['label'] == pred_label),
            'text_snippet': item['text'][:50].replace('\n', ' ') + "..."
        })
        
    df_results = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("             CALIBRATION ANALYSIS RESULTS            ")
    print("=" * 60)
    
    # 1. Summary Metrics
    print(f"Total Emails Tested: {len(df_results)}")
    print(f"Directional Accuracy (vs. synthetic true label): {df_results['correct'].mean() * 100:.1f}%")
    print(f"Mean Confidence: {df_results['confidence'].mean():.3f}")
    print(f"Median Confidence: {df_results['confidence'].median():.3f}")
    print(f"Min Confidence: {df_results['confidence'].min():.3f}")
    print(f"Max Confidence: {df_results['confidence'].max():.3f}")
    
    # 2. Score Distribution Bins
    print("\nConfidence Score Distribution Bins:")
    bins = [0.0, 0.5, 0.7, 0.9, 1.0]
    bin_labels = ["0.0 - 0.5 (Low)", "0.5 - 0.7 (Medium-Low)", "0.7 - 0.9 (Medium-High)", "0.9 - 1.0 (High)"]
    df_results['bin'] = pd.cut(df_results['confidence'], bins=bins, labels=bin_labels, include_lowest=True)
    bin_counts = df_results['bin'].value_counts().sort_index()
    for bin_name, count in bin_counts.items():
        percentage = (count / len(df_results)) * 100
        print(f"  {bin_name}: {count} ({percentage:.1f}%)")
        
    # 3. Check for Clustering Near 1.0
    high_conf_pct = (df_results['confidence'] >= 0.95).mean() * 100
    print(f"\nPercentage of predictions with confidence >= 0.95: {high_conf_pct:.1f}%")
    
    if high_conf_pct > 85.0:
        print("\n⚠️ WARNING: Confidence is heavily clustered near 1.0 (over-confidence).")
        print("This suggests Llama-3.1-8b-instant's self-reported confidence is poorly calibrated.")
        print("You may want to adjust --min-confidence to a higher value (e.g. 0.9 or 0.95),")
        print("or use a fixed fraction of items for manual review instead of a hard threshold.")
    else:
        print("\n✅ Confidence scores show variation. The confidence threshold approach is likely suitable.")
        
    # 4. Detailed output
    print("\nDetailed Calibration Test Results:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_results[['index', 'true_label', 'predicted_label', 'confidence', 'correct', 'text_snippet']].to_string(index=False))

if __name__ == "__main__":
    main()
