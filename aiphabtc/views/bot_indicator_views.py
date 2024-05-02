from django.shortcuts import render
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
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from dateutil import parser
from transformers import AutoModelForSequenceClassification, AlbertTokenizer
import torch
import torch.nn as nn
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
import asyncio

def get_predictions_arima(btc_sequence, p=1, d=1, q=1, steps_ahead=1):
    try:
        # Differencing
        btc_diff = np.diff(btc_sequence, n=d)
        # Fit ARIMA model
        model = ARIMA(btc_diff, order=(p, 0, q))
        fitted_model = model.fit()
        # Forecast
        forecast_diff = fitted_model.forecast(steps=steps_ahead)
        # Invert differencing
        forecast = [btc_sequence[-1]]
        for diff in forecast_diff:
            forecast.append(forecast[-1] + diff)
        return forecast[1:][0]
    except Exception as e:
        print(f"Model fitting failed: {str(e)}")
        return np.zeros((steps_ahead,))


def get_predictions_mlp(test_input):
    with open('aiphabtc/mlp_regressor.pkl', 'rb') as model_file:
        loaded_mlp = pickle.load(model_file)
    prediction = loaded_mlp.predict(test_input)
    return prediction


def get_predictions_elasticnet(test_input):
    with open("aiphabtc/elastic_net.pkl", "rb") as model_file:
        loaded_elasticnet = pickle.load(model_file)
    prediction = loaded_elasticnet.predict(test_input)
    return prediction


def preprocess_function(chart_df):
    days, months = [], []
    for dt in tqdm(chart_df.index):
        day = pd.to_datetime(dt).day
        month = pd.to_datetime(dt).month
        days.append(day)
        months.append(month)
    chart_df["day"] = days
    chart_df["months"] = months

    delta = chart_df["close"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    chart_df['RSI'] = 100 - (100 / (1 + rs))

    chart_df['SMA_20'] = chart_df['close'].rolling(window=20).mean()
    chart_df['STD_20'] = chart_df['close'].rolling(window=20).std()
    chart_df['Upper_Bollinger'] = chart_df['SMA_20'] + (chart_df['STD_20'] * 2)
    chart_df['Lower_Bollinger'] = chart_df['SMA_20'] - (chart_df['STD_20'] * 2)
    short_ema = chart_df['close'].ewm(span=12, adjust=False).mean()
    long_ema = chart_df['close'].ewm(span=26, adjust=False).mean()
    chart_df['MACD'] = short_ema - long_ema
    chart_df['Signal'] = chart_df['MACD'].ewm(span=9, adjust=False).mean()
    low_14 = chart_df['low'].rolling(window=14).min()
    high_14 = chart_df['high'].rolling(window=14).max()
    chart_df['%K'] = 100 * ((chart_df['close'] - low_14) / (high_14 - low_14))
    chart_df['%D'] = chart_df['%K'].rolling(window=3).mean()

    for l in tqdm(range(1, 4), position=0, leave=True):
        for col in ["high", "low", "volume"]:
            val = chart_df[col].values
            val_ret = [None for _ in range(l)]
            for i in range(l, len(val)):
                if val[i - l] == 0:
                    ret = 1
                else:
                    ret = val[i] / val[i - l]
                val_ret.append(ret)
            chart_df["{}_change_{}".format(col, l)] = val_ret

    chart_df.dropna(inplace=True)
    return chart_df


def get_predictions_xgboost(test_input):
    loaded_model = XGBClassifier()
    loaded_model.load_model("aiphabtc/xgb_clf_mainlanding")
    xgb_prob = loaded_model.predict_proba(test_input)[0]
    return xgb_prob[0] * 100.0, xgb_prob[1] * 100.0  # short, long


def get_predictions_lightgbm(test_input):
    test_lgb = lgb.Booster(model_file="aiphabtc/lightgbm_model.txt")
    lgb_prob = test_lgb.predict(test_input, num_iteration=test_lgb.best_iteration)[0]  # long probability
    short = 1 - lgb_prob
    return short * 100.0, lgb_prob * 100.0


def get_predictions_rf(test_input):
    with open("aiphabtc/rf_model.pkl", "rb") as file:
        loaded_rf = pickle.load(file)
    rf_prob = loaded_rf.predict_proba(test_input)[0]
    short, long = rf_prob[0], rf_prob[1]
    return short * 100.0, long * 100.0


def should_update_prediction():
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst)
    last_update = cache.get('last_prediction_update', now - timedelta(days=1))
    # Check if it's past 9 AM and if the last update was before today's 9 AM
    today_9am = now.replace(hour=9, minute=0, second=0, microsecond=0)
    if last_update < today_9am and now >= today_9am:
        return True
    return False

def trading_bot_indicator(request):
    if should_update_prediction() or not cache.get('predictions'):
        print("calculating as we cannot use previously cached value")
        df = pyupbit.get_ohlcv("KRW-BTC", interval="day")
        previous_btc_close = df["close"].values[-2]
        preprocessed_df = preprocess_function(df)
        clf_test_input = preprocessed_df.iloc[-2].values.reshape((1, -1))

        # ARIMA prediction
        btc_sequence = df["close"].values[:-1]
        arima_prediction = get_predictions_arima(btc_sequence)
        arima_percentage_change = (arima_prediction - previous_btc_close) / previous_btc_close * 100.0

        # MLP prediction
        mlp_test_input = df[["open", "high", "low", "close", "volume"]].iloc[-2].values.reshape((1, -1))
        mlp_prediction = get_predictions_mlp(mlp_test_input)
        mlp_percentage_change = (mlp_prediction - previous_btc_close) / previous_btc_close * 100.0

        # ElasticNet prediction
        elasticnet_test_input = df[["open", "high", "low", "close", "volume"]].iloc[-2].values.reshape((1, -1))
        elasticnet_prediction = get_predictions_elasticnet(elasticnet_test_input)
        elasticnet_percentage_change = (elasticnet_prediction - previous_btc_close) / previous_btc_close * 100.0

        # XGBoost prediction
        xgb_short, xgb_long = get_predictions_xgboost(clf_test_input)

        # LightGBM prediction
        lgb_short, lgb_long = get_predictions_lightgbm(clf_test_input)

        # RandomForest prediction
        rf_short, rf_long = get_predictions_rf(clf_test_input)

        predictions = {
            "arima_prediction": arima_prediction,
            "arima_percentage_change": arima_percentage_change,
            "mlp_prediction": mlp_prediction,
            "mlp_percentage_change": mlp_percentage_change,
            "elasticnet_prediction": elasticnet_prediction,
            "elasticnet_percentage_change": elasticnet_percentage_change,
            "xgb_short": xgb_short,
            "xgb_long": xgb_long,
            "lgb_short": lgb_short,
            "lgb_long": lgb_long,
            "rf_short": rf_short,
            "rf_long": rf_long,
        }
        cache.set('predictions', predictions, 86400)

    prediction_contexts = cache.get('predictions')

    return render(request, 'aiphabtc/bot_indicator.html', prediction_contexts)
