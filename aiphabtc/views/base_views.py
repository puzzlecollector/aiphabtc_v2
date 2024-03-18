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


def loading(request): 
    return render(request, "loading.html")

# return percentage
def latest_voting_data(request):
    options = VotingOption.objects.annotate(vote_count=Count('votes'))
    total_votes = sum(option.vote_count for option in options)
    data = {
        'labels': [option.name for option in options],
        'data': [(option.vote_count / total_votes * 100) if total_votes > 0 else 0 for option in options],
    }
    return JsonResponse(data)

# it says bitget, but we are using coinbase data
# rule: korean exchange - upbit, american exchange - coinbase
def get_kimchi_data():
    coinbasepro = ccxt.coinbasepro()
    data = {}
    seoul_timezone = pytz.timezone("Asia/Seoul")
    current_time_seoul = datetime.now(seoul_timezone)
    data["current_time"] = current_time_seoul.strftime("%Y-%m-%d %H:%M:%S")
    USDKRW = yf.Ticker("USDKRW=X")
    history = USDKRW.history(period="1d")
    data["now_usd_krw"] = history["Close"].iloc[0]
    data["now_upbit_price"] = pyupbit.get_current_price("KRW-BTC")
    data["now_bitget_price"] = coinbasepro.fetch_ticker("BTC/USDT")["close"]
    data["kp"] = round((data["now_upbit_price"] * 100 / (data["now_bitget_price"] * data["now_usd_krw"])) - 100, 3)
    return data

# for coinness data scraping
def get_articles(headers, url):
    news_req = requests.get(url, headers=headers)
    soup = BeautifulSoup(news_req.content, "html.parser")
    # Extracting title
    title = soup.find("h1", {"class": "view_top_title noselect"}).text.strip()
    # Finding the specific <div>
    article_content_div = soup.find('div', class_='article_content', itemprop='articleBody')
    content = ""  # Initialize content as empty string
    # Check if the div was found
    if article_content_div:
        # Extracting text from all <p> tags within the <div>
        p_tags = article_content_div.find_all('p')
        for p in p_tags:
            content += p.get_text(strip=True) + " "  # Appending each <p> content with a space for readability

        # Optionally, remove specific unwanted text
        unwanted_text = "이 광고는 쿠팡 파트너스 활동의 일환으로, 이에 따른 일정액의 수수료를 제공받습니다."
        content = content.replace(unwanted_text, "").strip()
    else:
        content = "No content found in the specified structure."
    return title, content

def scrape_tokenpost():
    all_titles, all_contents, all_full_times = [], [], []
    for i in tqdm(range(1, 2), desc="Scraping content from tokenpost"):
        try:
            links = []
            url = f"https://www.tokenpost.kr/coinness?page={i}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            news_req = requests.get(url, headers=headers)
            soup = BeautifulSoup(news_req.content, "html.parser")
            elems = soup.find_all("div", {"class": "list_left_item"})
            for e in elems:
                article_elems = e.find_all("div", {"class": "list_item_text"})
                for article in article_elems:
                    title_link = article.find("a", href=True)
                    if title_link and '/article-' in title_link['href']:
                        full_link = 'https://www.tokenpost.kr' + title_link['href']
                        # Find the date element in the parent of the article
                        date_elem = article.parent.find("span", {"class": "day"})
                        news_date = parser.parse(date_elem.text)
                        links.append(full_link)
                        all_full_times.append(news_date)
                    if len(all_full_times) > 4:
                        break
            for link in links:
                try:
                    title, content = get_articles(headers, link)
                    all_titles.append(title)
                    all_contents.append(content)
                except Exception as e:
                    print(f"Error while scraping news content: {e}")
        except Exception as e:
            print(f"Error while scraping page {i}: {e}")
        time.sleep(0.1)

    if len(all_titles) == 0 and len(all_full_times) == 5:
        for k in range(5):
            all_titles.append('')
            all_contents.append('')

    return pd.DataFrame({'titles': all_titles, 'contents': all_contents, 'datetimes': all_full_times})

def get_sentiment_scores(df):
    titles = df["titles"].values
    contents = df["contents"].values
    tokenizer = AlbertTokenizer.from_pretrained("aiphabtc/kr-cryptodeberta")
    sentiment_model = AutoModelForSequenceClassification.from_pretrained("aiphabtc/kr-cryptodeberta")
    scores = np.zeros(3)
    for i in range(len(titles)):
        encoded_inputs = tokenizer(str(titles[i]), str(contents[i]), max_length=512, padding="max_length",
                                   truncation=True, return_tensors="pt")
        with torch.no_grad():
            sentiment = sentiment_model(**encoded_inputs).logits
            sentiment = nn.Softmax(dim=1)(sentiment)
            sentiment = sentiment.detach().cpu().numpy()[0]
        scores += sentiment
    scores /= int(df.shape[0])
    print(scores)
    return scores  # average scores

def get_news_and_sentiment(request):
    # Your news scraping and sentiment analysis logic here
    df = scrape_tokenpost()
    scraped_data = df.to_dict(orient="records")  # converts DataFrame to list of dicts
    avg_sentiment_scores = get_sentiment_scores(df)
    avg_sentiment_scores_percentage = [round(score * 100, 2) for score in avg_sentiment_scores]
    sentiment_labels = ['호재', '악재', '중립']
    # Prepare your context data for JsonResponse
    data = {
        "scraped_data": scraped_data,
        "avg_sentiment_scores_percentage": avg_sentiment_scores_percentage,
        "sentiment_labels": sentiment_labels,
    }
    return JsonResponse(data)

def get_technical_indicators(timeframe="day"):
    df = pyupbit.get_ohlcv("KRW-BTC", interval=timeframe)
    df["SMA_50"] = df["close"].rolling(window=50).mean()
    df["EMA_50"] = df["close"].ewm(span=50, adjust=False).mean()
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['STD_20'] = df['close'].rolling(window=20).std()
    df['Upper_Bollinger'] = df['SMA_20'] + (df['STD_20'] * 2)
    df['Lower_Bollinger'] = df['SMA_20'] - (df['STD_20'] * 2)
    short_ema = df['close'].ewm(span=12, adjust=False).mean()
    long_ema = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = short_ema - long_ema
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['%K'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    df['%D'] = df['%K'].rolling(window=3).mean()
    # get last seven rows
    sample = df.iloc[-7:, 1:]
    sample_str = sample.to_string(index=False)
    data = {"output_str": sample_str}
    return data

def fetch_ai_technical1d(request):
    technical_data = get_technical_indicators(timeframe="day")
    technical_output = technical_data["output_str"]
    message = ("다음과 같은 일봉 비트코인 데이터가 주어졌을때:\n\n"
               "{}\n\n"
               "비트코인 가격 추세를 분석하고 총평을 해줘."
               ).format(technical_output)
    openai.api_key = settings.OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"user", "content":message}
        ]
    )
    chat_message = response["choices"][0]["message"]["content"]
    return JsonResponse({"chat_message": chat_message})

@login_required(login_url="common:login")
@require_http_methods(["POST"])  # Ensures that this view can only be accessed with a POST request
def submit_sentiment_vote(request):
    # Check if the request is an AJAX request
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        option_id = request.POST.get("sentimentVoteOption")
        try:
            selected_option = VotingOption.objects.get(id=option_id)
            Vote.objects.create(vote_option=selected_option)
            # Return a success message in JSON format
            return JsonResponse({"message": "투표해주셔서 감사합니다", "status": "success"})
        except VotingOption.DoesNotExist:
            return JsonResponse({"message": "선택한 옵션이 존재하지 않습니다", "status": "error"})
    # Fallback for non-AJAX requests if necessary
    return redirect("index")

def calculate_vote_percentages(voting_options):
    total_votes = sum(option.vote_count for option in voting_options)
    if total_votes == 0:
        return [(option, 0) for option in voting_options] # avoid division by zero
    return [(option, (option.vote_count / total_votes) * 100) for option in voting_options]


def get_correlation():
    pearson = spearman = kendall = -100
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        # Attempt to fetch NAS100 data
        NAS100 = yf.download('^NDX', start=start_date, end=end_date)['Close']

        # Attempt to fetch Bitcoin data
        df = pyupbit.get_ohlcv("KRW-BTC", count=30, interval="day")
        btc_close = df["close"].values

        # Ensure both datasets were fetched successfully
        if len(NAS100) > 0 and len(btc_close) > 0:
            min_length = min(len(btc_close), len(NAS100))  # Match their length

            # Align data
            btc_close_aligned = btc_close[-min_length:]
            NAS100_aligned = NAS100[-min_length:]

            data_aligned = pd.DataFrame({"btc": btc_close_aligned, "nas": NAS100_aligned})

            # Calculate correlations
            pearson_corr = data_aligned.corr(method='pearson')
            spearman_corr = data_aligned.corr(method='spearman')
            kendall_corr = data_aligned.corr(method='kendall')

            pearson = pearson_corr.iloc[0, 1]
            spearman = spearman_corr.iloc[0, 1]
            kendall = kendall_corr.iloc[0, 1]
            return pearson, spearman, kendall
    except Exception as e:
        # Handle any exception by logging or printing error message
        print(f"Error occurred: {e}")
    return pearson, spearman, kendall

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
    return xgb_prob[0]*100.0, xgb_prob[1]*100.0 # short, long

def get_predictions_lightgbm(test_input):
    test_lgb = lgb.Booster(model_file="aiphabtc/lightgbm_model.txt")
    lgb_prob = test_lgb.predict(test_input, num_iteration=test_lgb.best_iteration)[0] # long probability
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

def index(request):
    boards = Board.objects.all()
    board_posts = {}
    for board in boards:
        # Fetch the top 3 posts for each board
        posts = Question.objects.filter(board=board).order_by('-create_date')[:3]
        board_posts[board] = posts

    url_fng = "https://api.alternative.me/fng/?limit=7&date_format=kr"
    response_fng = requests.get(url_fng)
    data_fng = response_fng.json().get('data', [])

    url_global = "https://api.coinlore.net/api/global/"
    response_global = requests.get(url_global)
    data_global = response_global.json()

    kimchi_data = get_kimchi_data()

    sentiment_voting_options = VotingOption.objects.annotate(vote_count=Count('votes'))
    sentiment_votes_with_percentages = calculate_vote_percentages(sentiment_voting_options)
    sentiment_data = {
        "labels": [option.name for option in sentiment_voting_options],
        "data": [percentage for _, percentage in sentiment_votes_with_percentages]
    }

    pearson, spearman, kendall = get_correlation()

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
        cache.set('last_prediction_update', datetime.now(pytz.timezone('Asia/Seoul')))

    prediction_contexts = cache.get('predictions')
    context = {
        "board_posts": board_posts,
        "data_fng": data_fng,
        "data_global": data_global,
        "kimchi_data": kimchi_data,
        "sentiment_voting_options": sentiment_voting_options,
        "sentiment_data": sentiment_data,
        "pearson": pearson,
        "spearman": spearman,
        "kendall": kendall,
    }
    # merge context and prediction_contexts then return it as context
    context = {**context, **prediction_contexts}
    return render(request, 'index.html', context)


def index_orig(request, board_name="free_board"):
    page = request.GET.get('page', '1')
    kw = request.GET.get('kw', '')
    so = request.GET.get("so", "recent")

    # Initialize the query for all questions or filter by board if board_name is given
    if board_name:
        board = get_object_or_404(Board, name=board_name)
        question_list = Question.objects.filter(board=board)
    else:
        board = None
        question_list = Question.objects.all()

    # Apply filtering based on 'so' and 'kw'
    if so == "recommend":
        question_list = question_list.annotate(num_voter=Count('voter')).order_by('-num_voter', '-create_date')
    elif so == "popular":
        question_list = question_list.annotate(num_answer=Count("answer")).order_by("-num_answer", "-create_date")
    else:
        question_list = question_list.order_by("-create_date")

    if kw:
        question_list = question_list.filter(
            Q(subject__icontains=kw) |
            Q(content__icontains=kw) |
            Q(author__username__icontains=kw) |
            Q(answer__author__username__icontains=kw)
        ).distinct()

    paginator = Paginator(question_list, 10)
    page_obj = paginator.get_page(page)
    
    context = {
        "board": board,  # Include the board in context
        "question_list": page_obj,
        'page': page,
        'kw': kw,
        'so': so
    }
    return render(request, 'aiphabtc/question_list.html', context)

def detail(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    context = {"question": question}
    return render(request, 'aiphabtc/question_detail.html', context)

def community_guideline(request):
    return render(request, "guidelines.html", {})

# for perceptive board
def get_current_price(request, ticker):
    try:
        price = pyupbit.get_current_price(ticker)
        return JsonResponse({'price': price})
    except Exception as e:
        # Handle errors or the case where the price cannot be fetched
        return JsonResponse({'error': str(e)}, status=400)

def search_results(request):
    query = request.GET.get('q')
    if query:
        questions = Question.objects.filter(Q(subject__icontains=query) | Q(content__icontains=query))
        answers = Answer.objects.filter(content__icontains=query)
        comments = Comment.objects.filter(content__icontains=query)
    else:
        questions = Answer.objects.none()
        answers = Answer.objects.none()
        comments = Comment.objects.none()

    context = {
        'query': query,
        'questions': questions,
        'answers': answers,
        'comments': comments,
    }
    return render(request, 'aiphabtc/search_results.html', context)