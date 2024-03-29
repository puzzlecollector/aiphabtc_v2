from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse, JsonResponse
from ..models import Question, Answer, Comment, Board, Vote, VotingOption
from django.utils import timezone
from ..forms import QuestionForm, AnswerForm, CommentForm
from django.core.paginator import Paginator
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Q, Count
from django.conf import settings
from django.views.decorators.http import require_POST, require_http_methods
import requests
import pyupbit
import ccxt
import time
import yfinance as yf
import pytz
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from dateutil import parser
from transformers import AutoModelForSequenceClassification, AlbertTokenizer, AutoModel, AutoConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import openai
from statsmodels.tsa.arima.model import ARIMA
import pickle
import joblib
from sklearn.neural_network import MLPRegressor
from xgboost import XGBClassifier
import lightgbm as lgb
from common.models import PointTokenTransaction
from django.db.models import Q
from django.core.cache import cache
import backoff
import faiss
import json
import threading

# prepare model and tokenizer
tokenizer = AlbertTokenizer.from_pretrained("aiphabtc/kr-cryptodeberta")
embedding_model = AutoModel.from_pretrained("aiphabtc/kr-cryptodeberta")
config = AutoConfig.from_pretrained("aiphabtc/kr-cryptodeberta")
device = torch.device("cpu") # default to CPU

# read candidate texts
with open('aiphabtc/candidate_texts_0325.pkl', 'rb') as f:
    candidate_texts = pickle.load(f)

# read published dates for candidate texts
with open('aiphabtc/published_datetimes_0325.pkl', 'rb') as f:
    published_datetimes = pickle.load(f)

# get faiss index
index = faiss.read_index('aiphabtc/coinness_faiss_index_0325.index')

# get chart data
chart_df = pd.read_csv("aiphabtc/upbit_chart_data_0325.csv")

def inner_product_to_percentage(inner_product):
    return (inner_product + 1) / 2 * 100

def get_query_embedding(query):
    encoded_query = tokenizer(query, max_length=512, padding="max_length", truncation=True, return_tensors="pt").to(device)
    embedding_model.to(device)
    embedding_model.eval()
    with torch.no_grad():
      query_embedding = embedding_model(**encoded_query)[0][:, 0, :]
      query_embedding = query_embedding.numpy()
    return query_embedding

def get_relevant_chart_segment1d(chart_df, datestr):
    chart_df.set_index(pd.DatetimeIndex(chart_df.index), inplace=True)
    df1d_idx = -1
    cur_date = chart_df.index.values  # Ensure this column contains date and time
    news_datestr = datetime.strptime(datestr, "%Y-%m-%d %H:%M:%S")
    for i in range(len(cur_date) - 1):
        # Convert numpy.datetime64 to string and then to datetime
        current_date_str = cur_date[i].astype(str)
        current_date_str = current_date_str.split('T')[0] + ' ' + current_date_str.split('T')[1].split('.')[0]
        current_date = datetime.strptime(current_date_str, "%Y-%m-%d %H:%M:%S")
        next_date_str = cur_date[i + 1].astype(str)
        next_date_str = next_date_str.split('T')[0] + ' ' + next_date_str.split('T')[1].split('.')[0]
        next_date = datetime.strptime(next_date_str, "%Y-%m-%d %H:%M:%S")
        if news_datestr >= current_date and news_datestr < next_date:
            df1d_idx = i
            break
    return df1d_idx

def search_news(request):
    if request.method == "POST":
        data = json.loads(request.body)
        query = data.get("news_text")
        topk = int(data.get("top_k", 5))
        topk = max(5, min(topk, 20))
        # call the functions to perform the search
        query_embedding = get_query_embedding(query) 
        faiss.normalize_L2(query_embedding)
        distances, indices = index.search(query_embedding, 1000)
        results = []
        for i in range(topk):
            text = candidate_texts[indices[0][i]]
            text = text.replace("\n", "<br>")
            similarity = round(inner_product_to_percentage(distances[0][i]), 3)
            date = published_datetimes[indices[0][i]]
            relevant_chart_idx_1d = get_relevant_chart_segment1d(chart_df, date)
            relevant_chart_segment_1d = chart_df.iloc[relevant_chart_idx_1d:relevant_chart_idx_1d + 30] # relevant chart data for the next 1 month

            chart_data_1d = {
                'x': relevant_chart_segment_1d.index.tolist(),
                'y': relevant_chart_segment_1d['close'].tolist(),
            }

            results.append({
                "text": text,
                "similarity": similarity,
                "date": date,
                "chart_data_1d": chart_data_1d,
            })


        # Return the results as Json
        return JsonResponse({"results": results})

def pattern_matching_views(request):
    return render(request, "aiphabtc/pattern_matching.html", {})