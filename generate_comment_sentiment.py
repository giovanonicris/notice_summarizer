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
