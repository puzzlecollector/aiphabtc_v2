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
from tqdm import tqdm
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
from annoy import AnnoyIndex # alternative to faiss-cpu
import json
import gc

# prepare model and tokenizer
tokenizer = AlbertTokenizer.from_pretrained("aiphabtc/kr-cryptodeberta")
embedding_model = AutoModel.from_pretrained("aiphabtc/kr-cryptodeberta")
device = torch.device("cpu") # default to CPU

# read candidate texts
with open('aiphabtc/candidate_texts_0325.pkl', 'rb') as f:
    candidate_texts = pickle.load(f)

# read published dates for candidate texts
with open('aiphabtc/published_datetimes_0325.pkl', 'rb') as f:
    published_datetimes = pickle.load(f)

# get chart data
chart_df = pd.read_csv("aiphabtc/chart_data.csv", encoding='utf-8')

chart_df_30m = pd.read_csv("aiphabtc/chart_data_30m.csv", encoding="utf-8")

def inner_product_to_percentage(inner_product):
    return (inner_product + 1) / 2 * 100

# get annoy index
u = AnnoyIndex(768, 'angular')
try:
    # Load the saved index
    u.load('aiphabtc/coinness_annoy_30_ubuntu.index')  # Path to saved index
except Exception as e:
    print(f"Failed to load annoy index")

def get_query_embedding(query):
    encoded_query = tokenizer(query, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    embedding_model.eval()
    with torch.no_grad():
      query_embedding = embedding_model(**encoded_query)[0][:, 0, :]
      query_embedding = query_embedding.numpy()
    return query_embedding

def get_relevant_chart_segment30m(chart_df_30m, datestr):
    df1d_idx = -1
    cur_date = chart_df_30m["dates"].values  # Ensure this column contains date and time
    news_datestr = datetime.strptime(datestr, "%Y-%m-%d %H:%M:%S")
    for i in range(len(cur_date) - 1):
        # Convert numpy.datetime64 to string and then to datetime
        current_date_str = cur_date[i]
        # current_date_str = current_date_str.split('T')[0] + ' ' + current_date_str.split('T')[1].split('.')[0]
        current_date = datetime.strptime(current_date_str, "%Y-%m-%d %H:%M:%S")
        next_date_str = cur_date[i + 1]
        # next_date_str = next_date_str.split('T')[0] + ' ' + next_date_str.split('T')[1].split('.')[0]
        next_date = datetime.strptime(next_date_str, "%Y-%m-%d %H:%M:%S")
        if news_datestr >= current_date and news_datestr < next_date:
            df1d_idx = i
            break
    return df1d_idx


def get_relevant_chart_segment1d(chart_df, datestr):
    df1d_idx = -1
    cur_date = chart_df["dates"].values  # Ensure this column contains date and time
    news_datestr = datetime.strptime(datestr, "%Y-%m-%d %H:%M:%S")
    for i in range(len(cur_date) - 1):
        # Convert numpy.datetime64 to string and then to datetime
        current_date_str = cur_date[i]
        # current_date_str = current_date_str.split('T')[0] + ' ' + current_date_str.split('T')[1].split('.')[0]
        current_date = datetime.strptime(current_date_str, "%Y-%m-%d %H:%M:%S")
        next_date_str = cur_date[i + 1]
        # next_date_str = next_date_str.split('T')[0] + ' ' + next_date_str.split('T')[1].split('.')[0]
        next_date = datetime.strptime(next_date_str, "%Y-%m-%d %H:%M:%S")
        if news_datestr >= current_date and news_datestr < next_date:
            df1d_idx = i
            break
    return df1d_idx

def distance_to_percentage(distances):
    # Convert squared Euclidean distances to cosine similarity percentages
    percentages = [(1 - (d**2) / 2.0) * 50 + 50 for d in distances]
    return percentages


# use annoy instead of faiss for fast vector similarity search
def search_news(request):
    if request.method == "POST":
        data = json.loads(request.body)
        query = data.get("news_text")
        topk = int(data.get("top_k", 5))
        topk = max(5, min(topk, 20))
        # call the functions to perform the search
        query_embedding = get_query_embedding(query)
        query_embedding = query_embedding.reshape((768))
        indices, distances = u.get_nns_by_vector(query_embedding, 30, include_distances=True)  # Finds the 30 nearest neighbors
        percentages = distance_to_percentage(distances)
        results = []
        for i in range(topk):
            text = candidate_texts[indices[i]]
            text = text.replace("\n", "<br>")
            similarity = round(percentages[i], 3)
            date = published_datetimes[indices[i]]

            relevant_chart_idx_30m = get_relevant_chart_segment30m(chart_df_30m, date)
            relevant_chart_segment30m = chart_df_30m.iloc[relevant_chart_idx_30m:relevant_chart_idx_30m + 48] # relevant chart data for the next day

            relevant_chart_idx_1d = get_relevant_chart_segment1d(chart_df, date)
            relevant_chart_segment_1d = chart_df.iloc[relevant_chart_idx_1d:relevant_chart_idx_1d + 30] # relevant chart data for the next 1 month

            chart_data_30m = {
                'x': relevant_chart_segment30m["dates"].tolist(),
                'y': relevant_chart_segment30m['close'].tolist(),
            }

            chart_data_1d = {
                'x': relevant_chart_segment_1d["dates"].tolist(),
                'y': relevant_chart_segment_1d['close'].tolist(),
            }

            results.append({
                "text": text,
                "similarity": similarity,
                "date": date,
                "chart_data_30m": chart_data_30m,
                "chart_data_1d": chart_data_1d,
            })

        # Return the results as Json
        return JsonResponse({"results": results})


def dtw_similarity(ts_a, ts_b, d=lambda x, y: abs(x - y)):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param ts_a: Numpy array of the first time series
    :param ts_b: Numpy array of the second time series
    :param d: Distance function
    :return: The DTW distance between the two time series
    """
    # Create cost matrix
    ts_a, ts_b = np.array(ts_a), np.array(ts_b)
    M, N = len(ts_a), len(ts_b)
    cost = np.zeros((M, N))

    # Initialize the first row and column
    cost[0, 0] = d(ts_a[0], ts_b[0])
    for i in range(1, M):
        cost[i, 0] = cost[i - 1, 0] + d(ts_a[i], ts_b[0])

    for j in range(1, N):
        cost[0, j] = cost[0, j - 1] + d(ts_a[0], ts_b[j])

    # Populate rest of the cost matrix
    for i in range(1, M):
        for j in range(1, N):
            choices = cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]
            cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])
    return cost[-1, -1]


def search_chart_pattern(request, chart_type):
    if request.method == "POST":
        data = json.loads(request.body)
        topk = int(data.get("top_k", 5))
        topk = max(5, min(topk, 20))

        # Determine chart data source based on chart_type
        if chart_type == "1d":
            btc_krw = pyupbit.get_ohlcv("KRW-BTC", interval="day")
            btc_krw["dates"] = btc_krw.index
            btc_krw = btc_krw.iloc[-21:]
        elif chart_type == "30m":
            btc_krw = pyupbit.get_ohlcv("KRW-BTC", interval="minute30")
            btc_krw["dates"] = btc_krw.index
            btc_krw = btc_krw.iloc[-7:]
        else:
            return JsonResponse({"error": "Invalid chart type"}, status=400)

        current_pattern = btc_krw["close"].values
        current_datetime = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in btc_krw["dates"]]

        # Determine historical chart data source based on chart_type
        if chart_type == "1d":
            historical_data = chart_df["close"].values[:-30]
            historical_date = chart_df["dates"].values[:-30]
        elif chart_type == "30m":
            historical_data = chart_df_30m["close"].values
            historical_date = chart_df_30m["dates"].values
        else:
            return JsonResponse({"error": "Invalid chart type"}, status=400)

        similarities = []
        for i in tqdm(range(len(historical_data) - len(current_pattern))):
            past_pattern = historical_data[i:i + len(current_pattern)]
            similarity = dtw_similarity(current_pattern, past_pattern)
            similarities.append((i, similarity))

        top_k_similar = sorted(similarities, key=lambda x: x[1])[:topk]

        results = []
        date_format = "%Y-%m-%d %H:%M:%S"
        for i in range(topk):
            cur_idx, _ = top_k_similar[i]
            if chart_type == "1d":
                add = 21
            elif chart_type == "30m":
                add = 7
            sim_chart_start = historical_data[cur_idx:cur_idx + add].tolist()
            sim_chart_end = historical_data[cur_idx + add:cur_idx + add + add].tolist()
            date_start = [pd.Timestamp(ts).strftime(date_format) for ts in historical_date[cur_idx:cur_idx + add]]
            date_end = [pd.Timestamp(ts).strftime(date_format) for ts in historical_date[cur_idx + add:cur_idx + add + add]]

            results.append({
                "chart_data_start": sim_chart_start,
                "chart_data_end": sim_chart_end,
                "date_start": date_start,
                "date_end": date_end,
            })

        return JsonResponse({
            'results': results,
            'current_pattern': current_pattern.tolist(),
            'current_datetime': current_datetime,
            'chart_type': chart_type  # Indicate whether the response is for 1d or 30m data
        })


def get_current_chart_pattern_helper(chart_type):
    if chart_type == "1d":
        btc_krw = pyupbit.get_ohlcv("KRW-BTC", interval="day")
        btc_krw["dates"] = btc_krw.index
        btc_krw = btc_krw.iloc[-21:]
    elif chart_type == "30m":
        btc_krw = pyupbit.get_ohlcv("KRW-BTC", interval="minute30")
        btc_krw["dates"] = btc_krw.index
        btc_krw = btc_krw.iloc[-7:]
    else:
        return JsonResponse({"error": "Invalid chart type"}, status=400)

    current_pattern = btc_krw["close"].values
    current_datetime = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in btc_krw["dates"]]

    pattern = {
        'dates': current_datetime,
        'close': current_pattern.tolist(),
    }
    return pattern

def get_current_chart_pattern(request):
    # Fetch the most recent patterns for both 1d and 30m charts
    current_pattern_1d = get_current_chart_pattern_helper('1d')  # Implement this function
    current_pattern_30m = get_current_chart_pattern_helper('30m')  # And this one
    return JsonResponse({
        'current_pattern_1d': current_pattern_1d,
        'current_pattern_30m': current_pattern_30m,
    })


def pattern_matching_views(request):
    return render(request, "aiphabtc/pattern_matching.html", {})