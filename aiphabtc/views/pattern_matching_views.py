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
import pickle
import joblib
from sklearn.neural_network import MLPRegressor
from xgboost import XGBClassifier
from common.models import PointTokenTransaction
from django.db.models import Q
from django.core.cache import cache
import backoff
from annoy import AnnoyIndex # alternative to faiss-cpu
import json
import gc
import math
import logging

# define functions to get ohlcv from MEXC
def fetch_with_retries(url, params, retries=5, delays=5):
    for i in range(retries):
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                logging.error(f"Failed to fetch data from {url}. Status code: {response.status_code}")
        except Exception as e:
            logging.error(f"Attempt {i+1}: Error fetching data from {url}: {e}")
            time.sleep(delays)
    raise Exception(f"Failed to fetch data fater {retries} attempts")

def get_mexc_ohlcv(symbol="BTC_USDT", interval="1d", limit=21):
    url = f"https://www.mexc.com/open/api/v2/market/kline"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    try:
        data = fetch_with_retries(url, params)
        df = pd.DataFrame(data["data"], columns=["timestamp", "open", "high", "low", "close", "volume", "QuoteAssetVolume"])
        df.drop(columns={"QuoteAssetVolume"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        today_utc = datetime.utcnow().date()
        df = df[df["timestamp"].dt.date < today_utc]
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df
    except Exception as e:
        logging.error(f"Error in get_mexc_ohlcv: {e}")
        raise


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
chart_df = pd.read_csv("aiphabtc/kr_1d_truncated_0925.csv", encoding='utf-8') # BTC-KRW dataframe
chart_df_30m = pd.read_csv("aiphabtc/kr_30m_truncated_0925.csv", encoding="utf-8") # BTC-KRW dataframe

# get usdt chart data
chart_df_usdt = pd.read_csv("aiphabtc/usd_1d_truncated.csv", encoding="utf-8") # BTC-USDT dataframe
chart_df_usdt_30m = pd.read_csv("aiphabtc/usd_30m_truncated.csv", encoding="utf-8") # BTC-USDT dataframe

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


def get_relevant_chart_segment30m(chart_df_30m_cur, datestr):
    df1d_idx = -1
    cur_date = chart_df_30m_cur["dates"].values  # Ensure this column contains date and time
    # Handle both timezone-aware and naive dates
    if '+' in datestr or 'Z' in datestr:  # Check if it's a timezone-aware string
        news_datestr = datetime.strptime(datestr, "%Y-%m-%d %H:%M:%S%z")
    else:
        news_datestr = datetime.strptime(datestr, "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.UTC)

    for i in range(len(cur_date) - 1):
        # Convert numpy.datetime64 to string and then to datetime
        current_date_str = str(cur_date[i])
        next_date_str = str(cur_date[i + 1])

        # Convert to datetime with timezone awareness
        if '+' in current_date_str or 'Z' in current_date_str:
            current_date = datetime.strptime(current_date_str, "%Y-%m-%d %H:%M:%S%z")
            next_date = datetime.strptime(next_date_str, "%Y-%m-%d %H:%M:%S%z")
        else:
            current_date = datetime.strptime(current_date_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.UTC)
            next_date = datetime.strptime(next_date_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.UTC)

        if news_datestr >= current_date and news_datestr < next_date:
            df1d_idx = i
            break
    return df1d_idx


def get_relevant_chart_segment1d(chart_df_cur, datestr):
    df1d_idx = -1
    cur_date = chart_df_cur["dates"].values  # Ensure this column contains date and time

    # Handle both timezone-aware and naive dates
    if '+' in datestr or 'Z' in datestr:
        news_datestr = datetime.strptime(datestr, "%Y-%m-%d %H:%M:%S%z")
    else:
        news_datestr = datetime.strptime(datestr, "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.UTC)

    for i in range(len(cur_date) - 1):
        # Convert numpy.datetime64 to string and then to datetime
        current_date_str = str(cur_date[i])
        next_date_str = str(cur_date[i + 1])

        # Convert to datetime with timezone awareness
        if '+' in current_date_str or 'Z' in current_date_str:
            current_date = datetime.strptime(current_date_str, "%Y-%m-%d %H:%M:%S%z")
            next_date = datetime.strptime(next_date_str, "%Y-%m-%d %H:%M:%S%z")
        else:
            current_date = datetime.strptime(current_date_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.UTC)
            next_date = datetime.strptime(next_date_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.UTC)

        if news_datestr >= current_date and news_datestr < next_date:
            df1d_idx = i
            break
    return df1d_idx


def distance_to_percentage(distances):
    # Convert squared Euclidean distances to cosine similarity percentages
    percentages = [(1 - (d**2) / 2.0) * 50 + 50 for d in distances]
    return percentages


# Translation function
def translate_text_deepl(text, target_lang="EN", source_lang="KO"):
    api_url = "https://api.deepl.com/v2/translate"
    api_key = settings.DEEPL_API_KEY  # Replace with your actual DeepL API key

    params = {
        "auth_key": api_key,
        "text": text,
        "target_lang": target_lang,
        "source_lang": source_lang
    }

    try:
        response = requests.post(api_url, data=params)
        if response.status_code == 200:
            translation_result = response.json()
            translated_text = translation_result['translations'][0]['text']
            return translated_text
        else:
            return f"Error: {response.status_code}, {response.text}"
    except requests.RequestException as e:
        return f"Error: {str(e)}"


# use annoy instead of faiss for fast vector similarity search
def search_news(request):
    if request.method == "POST":
        data = json.loads(request.body)
        query = data.get("news_text")
        topk = int(data.get("top_k", 5))
        chart_type = data.get("chart_type", "KRW")  # Get the chart type from the request
        topk = max(5, min(topk, 20))

        # Assign the appropriate dataframe based on chart_type
        if chart_type == "USDT":
            print("Using USDT chart data")
            relevant_chart_df_30m = chart_df_usdt_30m
            relevant_chart_df_1d = chart_df_usdt
        elif chart_type == "KRW":
            print("Using KRW chart data")
            relevant_chart_df_30m = chart_df_30m
            relevant_chart_df_1d = chart_df

        # Detect if the query is in English, and translate it to Korean if necessary
        lang = data.get("language", "ko")
        if lang == "en":
            query = translate_text_deepl(query, target_lang="KO", source_lang="EN")

        # Perform the search using Annoy
        query_embedding = get_query_embedding(query)
        query_embedding = query_embedding.reshape((768))
        indices, distances = u.get_nns_by_vector(query_embedding, 30, include_distances=True)  # Find 30 nearest neighbors
        percentages = distance_to_percentage(distances)
        results = []

        for i in range(topk):
            text = candidate_texts[indices[i]]

            # Translate the result back to English if the request was for English
            if lang == "en":
                text = translate_text_deepl(text, target_lang="EN", source_lang="KO")

            similarity = round(percentages[i], 3)
            date = published_datetimes[indices[i]]

            # Use the relevant dataframe based on chart type for getting chart data
            relevant_chart_idx_30m = get_relevant_chart_segment30m(relevant_chart_df_30m, date)
            relevant_chart_segment30m = relevant_chart_df_30m.iloc[relevant_chart_idx_30m:relevant_chart_idx_30m + 48]

            relevant_chart_idx_1d = get_relevant_chart_segment1d(relevant_chart_df_1d, date)
            relevant_chart_segment_1d = relevant_chart_df_1d.iloc[relevant_chart_idx_1d:relevant_chart_idx_1d + 30]

            # Prepare chart data for both 30m and 1d intervals
            cur_chart_data_30m = {
                'x': relevant_chart_segment30m["dates"].tolist(),
                'y': relevant_chart_segment30m['close'].tolist(),
            }

            cur_chart_data_1d = {
                'x': relevant_chart_segment_1d["dates"].tolist(),
                'y': relevant_chart_segment_1d['close'].tolist(),
            }

            results.append({
                "text": text.replace("\n", "<br>"),  # Format text for HTML
                "similarity": similarity,
                "date": date,
                "chart_data_30m": cur_chart_data_30m,
                "chart_data_1d": cur_chart_data_1d,
            })

        # Return the results as JSON
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
        chart_currency = data.get("chart_currency", "KRW")  # Get the chart currency (KRW or USDT)
        topk = max(5, min(topk, 20))

        # Determine the correct chart data source based on chart_currency and chart_type
        if chart_currency == "KRW":
            if chart_type == "1d":
                btc_chart = pyupbit.get_ohlcv("KRW-BTC", interval="day")
                btc_chart["dates"] = btc_chart.index
                btc_chart = btc_chart.iloc[-21:]
            elif chart_type == "30m":
                btc_chart = pyupbit.get_ohlcv("KRW-BTC", interval="minute30")
                btc_chart["dates"] = btc_chart.index
                btc_chart = btc_chart.iloc[-7:]
            else:
                return JsonResponse({"error": "Invalid chart type"}, status=400)

            # Use the historical KRW chart dataframes
            if chart_type == "1d":
                historical_data = chart_df["close"].values[:-30]
                historical_date = chart_df["dates"].values[:-30]
            elif chart_type == "30m":
                historical_data = chart_df_30m["close"].values
                historical_date = chart_df_30m["dates"].values
            else:
                return JsonResponse({"error": "Invalid chart type"}, status=400)

        elif chart_currency == "USDT":
            if chart_type == "1d":
                btc_chart = pyupbit.get_ohlcv("USDT-BTC", interval="day")
                btc_chart["dates"] = btc_chart.index
                btc_chart = btc_chart.iloc[-21:]
            elif chart_type == "30m":
                btc_chart = pyupbit.get_ohlcv("USDT-BTC", interval="minute30")
                btc_chart["dates"] = btc_chart.index
                btc_chart = btc_chart.iloc[-7:]
            else:
                return JsonResponse({"error": "Invalid chart type"}, status=400)

            # Use the historical USDT chart dataframes
            if chart_type == "1d":
                historical_data = chart_df_usdt["close"].values[:-30]
                historical_date = chart_df_usdt["dates"].values[:-30]
            elif chart_type == "30m":
                historical_data = chart_df_usdt_30m["close"].values
                historical_date = chart_df_usdt_30m["dates"].values
            else:
                return JsonResponse({"error": "Invalid chart type"}, status=400)

        else:
            return JsonResponse({"error": "Invalid chart currency"}, status=400)

        # Extract the current pattern and datetime from the latest data
        current_pattern = btc_chart["close"].values
        current_datetime = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in btc_chart["dates"]]

        # Perform Dynamic Time Warping (DTW) to match patterns with historical data
        similarities = []
        for i in tqdm(range(len(historical_data) - len(current_pattern))):
            past_pattern = historical_data[i:i + len(current_pattern)]
            similarity = dtw_similarity(current_pattern, past_pattern)
            similarities.append((i, similarity))

        top_k_similar = sorted(similarities, key=lambda x: x[1])[:topk]

        # Build results
        results = []
        date_format = "%Y-%m-%d %H:%M:%S"
        for i in range(topk):
            cur_idx, _ = top_k_similar[i]
            add = 21 if chart_type == "1d" else 7
            sim_chart_start = historical_data[cur_idx:cur_idx + add].tolist()
            sim_chart_end = historical_data[cur_idx + add:cur_idx + add + add].tolist()
            date_start = [pd.Timestamp(ts).strftime(date_format) for ts in historical_date[cur_idx:cur_idx + add]]
            date_end = [pd.Timestamp(ts).strftime(date_format) for ts in
                        historical_date[cur_idx + add:cur_idx + add + add]]

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
            'chart_type': chart_type,  # Indicate whether the response is for 1d or 30m data
            'chart_currency': chart_currency  # Include chart currency
        })

def get_current_chart_pattern_helper(chart_currency, chart_type):
    if chart_currency == "KRW":
        if chart_type == "1d":
            btc_chart = pyupbit.get_ohlcv("KRW-BTC", interval="day")
            btc_chart["dates"] = btc_chart.index
            btc_chart = btc_chart.iloc[-21:]
        elif chart_type == "30m":
            btc_chart = pyupbit.get_ohlcv("KRW-BTC", interval="minute30")
            btc_chart["dates"] = btc_chart.index
            btc_chart = btc_chart.iloc[-7:]
        else:
            return JsonResponse({"error": "Invalid chart type"}, status=400)
    elif chart_currency == "USDT":
        if chart_type == "1d":
            btc_chart = get_mexc_ohlcv(symbol="BTC_USDT", interval="1d", limit=21)
            btc_chart.rename(columns={"timestamp":"dates"}, inplace=True)
        elif chart_type == "30m":
            btc_chart = get_mexc_ohlcv(symbol="BTC_USDT", interval="30m", limit=21)
            btc_chart.rename(columns={"timestamp":"dates"}, inplace=True)
            btc_chart = btc_chart.iloc[-7:]
        else:
            return JsonResponse({"error": "Invalid chart type"}, status=400)

    current_pattern = btc_chart["close"].values

    current_datetime = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in btc_chart["dates"]]

    pattern = {
        'dates': current_datetime,
        'close': current_pattern.tolist(),
    }
    return pattern

def get_current_chart_pattern(request):
    if request.method == "GET":
        chart_currency = request.GET.get("chart_currency", "KRW")  # Get chart currency from request
        chart_type = request.GET.get("chart_type", "1d")  # Get chart type from request

        # Fetch the most recent patterns for the given chart type and currency
        current_pattern = get_current_chart_pattern_helper(chart_currency, chart_type)
        return JsonResponse({
            'current_pattern': current_pattern,
            'chart_currency': chart_currency,
            'chart_type': chart_type
        })
    else:
        return JsonResponse({"error": "Invalid request method"}, status=405)


def pattern_matching_views(request):
    return render(request, "aiphabtc/pattern_matching.html", {})