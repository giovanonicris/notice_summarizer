import os
import re
import io
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import traceback
from nltk import download
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pdfminer.high_level import extract_text
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

## SET UP
# download nltk tokenizer
download('punkt')

# sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def textrank_summary(text, num_sentences=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)

def extract_text_from_pdf_url(url):
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return extract_text(io.BytesIO(response.content))
    except Exception as e:
        print(f"Failed to extract PDF from {url}: {e}")
        return ""

# load Notice list
input_csv_path = "notice_selection.csv"
notices_df = pd.read_csv(input_csv_path)
notice_ids = notices_df['notice_title'].dropna().tolist()

aggregated_rows = []
detailed_rows = []

for notice_id in notice_ids:
    print(f"\nProcessing {notice_id}")
    try:
        slug = notice_id.lower().replace("regulatory notice ", "").strip()
        url = f"https://www.finra.org/rules-guidance/notices/{slug}"

        res = requests.get(url, timeout=15)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        comment_table = soup.select_one("div.table-responsive table")
        if not comment_table:
            print(f"No comment table for {notice_id}")
            continue

        rows = comment_table.select("tbody tr")
        comments = []

        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 2:
                continue

            date_text = cols[0].get_text(strip=True)
            a_tag = cols[1].find("a")
            commenter_text = a_tag.get_text(strip=True) if a_tag else cols[1].get_text(strip=True)
            href = a_tag["href"] if a_tag and "href" in a_tag.attrs else None

            if not href:
                continue

            if href.startswith("/"):
                href = "https://www.finra.org" + href

            if href.lower().endswith(".pdf"):
                comment_text = extract_text_from_pdf_url(href)
            else:
                try:
                    comment_res = requests.get(href, timeout=15)
                    comment_res.raise_for_status()
                    comment_soup = BeautifulSoup(comment_res.text, "html.parser")

                    # aggressive strategy for web content - grab ALL visible text blocks inside <main> that are not nav/footer
                    main_content = comment_soup.select_one("main")
                    if main_content:
                        all_paragraphs = main_content.find_all("p")
                        comment_text = "\n".join(p.get_text(strip=True) for p in all_paragraphs if p.get_text(strip=True))
                    else:
                        comment_text = ""

                    if len(comment_text) < 100 or comment_text.startswith("For the Public"):
                        print(f"Skipping generic or short content from {href}")
                        continue

                except Exception as e:
                    print(f"Failed to extract HTML from {href}: {e}")
                    continue
            score = analyzer.polarity_scores(comment_text)
            summary = textrank_summary(comment_text)
            detailed_rows.append({
                "notice_id": notice_id,
                "notice_id_slug": slug,
                "comment_id": href.split("/")[-1],
                "comment_link": href,
                "commenter": commenter_text,
                "date": date_text,
                "comment_text": comment_text,
                "summary": summary,
                "score": score['compound'],
                "pos": score['pos'],
                "neg": score['neg']
            })

            comments.append(comment_text)

        if comments:
            combined = " ".join(comments)
            avg_score = sum(analyzer.polarity_scores(c)['compound'] for c in comments) / len(comments)
            avg_pos = sum(analyzer.polarity_scores(c)['pos'] for c in comments) / len(comments)
            avg_neg = sum(analyzer.polarity_scores(c)['neg'] for c in comments) / len(comments)
            summary = textrank_summary(combined, 3)

            aggregated_rows.append({
                "notice_id": notice_id,
                "num_comments": len(comments),
                "avg_score": round(avg_score, 3),
                "avg_pos": round(avg_pos, 3),
                "avg_neg": round(avg_neg, 3),
                "content_summary": summary
            })

    except Exception as e:
        print(f"ERROR!! Failed to process {notice_id}:")
        traceback.print_exc()

# save output!!
pd.DataFrame(aggregated_rows).to_csv("comments_aggregated.csv", index=False)
pd.DataFrame(detailed_rows).to_csv("comments_detailed.csv", index=False)

print(f"\nSaved {len(aggregated_rows)} summaries and {len(detailed_rows)} comments.")