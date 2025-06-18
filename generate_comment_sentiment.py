import requests
import random
import re
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import datetime as dt
import os

# Set dates for today
now = dt.datetime.now()

# check and download punkt tokenizer
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Create a list of random user agents
user_agent_list = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64)...',
]
user_agent = random.choice(user_agent_list)
header = {'User-Agent': user_agent}

# load notice list to scrape
input_csv_path = "notices_to_scrape.csv"
notices_df = pd.read_csv(input_csv_path)
notice_ids = notices_df['notice_id'].dropna().tolist()

# prep lists to store new entries
aggregated_rows = []
detailed_rows = []

# sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# helper to summarize comment text
def summarize_text(text, num_sentences=2):
    sentences = sent_tokenize(text)
    return " ".join(sentences[:num_sentences]) if sentences else ""

# start loop per notice
for notice_id in notice_ids:
    print(f"Processing {notice_id}")
    notice_slug = notice_id.lower().replace("regulatory notice ", "").replace("-", "")
    url = f"https://www.finra.org/rules-guidance/notices/{notice_slug}"

    try:
        req = requests.get(url, headers=header, timeout=20)
        soup = BeautifulSoup(req.text, 'html.parser')

        # attempt to find comments section
        comments_section = soup.find("div", class_="view-comments")
        if not comments_section:
            print(f"No comments found for {notice_id}")
            continue

        comment_blocks = comments_section.find_all("div", class_="comment")
        if not comment_blocks:
            print(f"No individual comments for {notice_id}")
            continue

        comments = []
        for block in comment_blocks:
            commenter = block.find("div", class_="comment-author")
            comment_date = block.find("div", class_="comment-date")
            comment_body = block.find("div", class_="comment-body")

            commenter_text = commenter.get_text(strip=True) if commenter else "Unknown"
            date_text = comment_date.get_text(strip=True) if comment_date else ""
            body_text = comment_body.get_text(strip=True) if comment_body else ""

            score = analyzer.polarity_scores(body_text)
            summary = summarize_text(body_text)

            detailed_rows.append({
                "notice_id": notice_id,
                "commenter": commenter_text,
                "date": date_text,
                "comment_text": body_text,
                "summary": summary,
                "score": score['compound'],
                "pos": score['pos'],
                "neg": score['neg']
            })

            comments.append(body_text)

        if comments:
            combined = " ".join(comments)
            avg_score = sum(analyzer.polarity_scores(c)['compound'] for c in comments) / len(comments)
            avg_pos = sum(analyzer.polarity_scores(c)['pos'] for c in comments) / len(comments)
            avg_neg = sum(analyzer.polarity_scores(c)['neg'] for c in comments) / len(comments)
            summary = summarize_text(combined, 3)

            aggregated_rows.append({
                "notice_id": notice_id,
                "num_comments": len(comments),
                "avg_score": round(avg_score, 3),
                "avg_pos": round(avg_pos, 3),
                "avg_neg": round(avg_neg, 3),
                "content_summary": summary
            })

    except requests.exceptions.RequestException as e:
        print(f"Request error for {notice_id}: {e}")
    except Exception as e:
        print(f"Error parsing {notice_id}: {e}")

# convert to DataFrames
agg_df = pd.DataFrame(aggregated_rows)
detail_df = pd.DataFrame(detailed_rows)

# save output files
output_dir = "finra_comments_output"
os.makedirs(output_dir, exist_ok=True)
agg_df.to_csv(os.path.join(output_dir, "comments_aggregated.csv"), index=False)
detail_df.to_csv(os.path.join(output_dir, "comments_detailed.csv"), index=False)

print(f"Saved {len(agg_df)} summaries and {len(detail_df)} comments.")
