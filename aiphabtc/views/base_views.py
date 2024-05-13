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
from django.templatetags.static import static
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
from telethon import TelegramClient, errors


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
    # Check cache first
    data = cache.get('kimchi_data')
    if data is not None:
        print("Cached data exists")
        return data

    print("No current cached kimchi premium data")
    data = {}
    seoul_timezone = pytz.timezone("Asia/Seoul")
    current_time_seoul = datetime.now(seoul_timezone)
    data["current_time"] = current_time_seoul.strftime("%Y-%m-%d %H:%M:%S")

    # Initialize API clients
    coinbasepro = ccxt.coinbasepro()

    # Fetch data using external APIs
    try:
        # Fetch USD to KRW exchange rate
        USDKRW = yf.Ticker("USDKRW=X")
        history = USDKRW.history(period="1d")
        data["now_usd_krw"] = history["Close"].iloc[0]

        # Fetch Bitcoin prices from Upbit and Coinbase
        data["now_upbit_price"] = pyupbit.get_current_price("KRW-BTC")
        data["now_bitget_price"] = coinbasepro.fetch_ticker("BTC/USDT")["close"]

        # Calculate the Kimchi Premium if all values are available
        if None not in (data["now_usd_krw"], data["now_upbit_price"], data["now_bitget_price"]):
            data["kp"] = round((data["now_upbit_price"] * 100 / (data["now_bitget_price"] * data["now_usd_krw"])) - 100, 3)
        else:
            data["kp"] = "Data unavailable"

    except Exception as e:
        print(f"Failed to fetch kimchi data due to: {e}")
        data = {"error": "Failed to fetch data"}

    # Cache the data for 1 hour if fetched successfully
    cache.set('kimchi_data', data, 3600)
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


# Define a function to give up on after a certain number of retries
# or only retry on certain status codes
@backoff.on_exception(backoff.expo,
                      requests.exceptions.RequestException,
                      max_tries=8,
                      giveup=lambda e: e.response is not None and e.response.status_code < 500)
def fetch_with_retry(url, headers):
    print(f"Trying {url}")
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Will trigger retry on 4xx and 5xx errors
    return response.json()


def scrape_coinness_xhr():
    url = 'https://api.coinness.com/feed/v1/news'
    headers = {'User-Agent': 'Mozilla/5.0'}
    titles, contents, datetimes_arr = [], [], []

    try:
        news_data = fetch_with_retry(url, headers)

        # Loop through each news item in the response
        for news_item in news_data:
            title = news_item.get('title')
            content = news_item.get('content')
            publish_at = news_item.get('publishAt')

            titles.append(title)
            contents.append(content)
            datetimes_arr.append(publish_at)

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e.response.status_code}")
        return pd.DataFrame()  # Return an empty DataFrame on failure
    except requests.exceptions.RequestException as e:
        print("Networking error, giving up.")
        return pd.DataFrame()  # Return an empty DataFrame on failure

    return pd.DataFrame({'titles': titles, 'contents': contents, 'datetimes': datetimes_arr})


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
    df = scrape_coinness_xhr()
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

def get_technical_indicators_1m(timeframe="month"):
    df = pyupbit.get_ohlcv("KRW-BTC", interval=timeframe)
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
    # get last six rows
    sample = df.iloc[-6:, 1:]
    sample_str = sample.to_string(index=False)
    data = {"output_str": sample_str}
    return data

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

def fetch_ai_technical1m(request):
    technical_data = get_technical_indicators_1m(timeframe="month")
    technical_output = technical_data["output_str"]
    message = ("다음과 같은 월봉 비트코인 데이터가 주어졌을때:\n\n"
               "{}\n\n"
               "비트코인 가격 추세를 분석하고 총평을 해줘."
               ).format(technical_output)
    openai.api_key = settings.OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": message}
        ]
    )
    chat_message = response["choices"][0]["message"]["content"]
    return JsonResponse({"chat_message": chat_message})


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
            {"role": "user", "content": message}
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
        return [(option, 0) for option in voting_options]  # avoid division by zero
    return [(option, (option.vote_count / total_votes) * 100) for option in voting_options]


def should_update_prediction():
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst)
    last_update = cache.get('last_prediction_update', now - timedelta(days=1))
    # Check if it's past 9 AM and if the last update was before today's 9 AM
    today_9am = now.replace(hour=9, minute=0, second=0, microsecond=0)
    if last_update < today_9am and now >= today_9am:
        return True
    return False


api_id = settings.TELEGRAM_ID
api_hash = settings.TELEGRAM_HASH

# Function to get messages from a specified Telegram channel
async def get_telegram_messages(client, channel, limit=3):
    try:
        messages = await client.get_messages(channel, limit=limit)
        return [(msg.date, msg.message) for msg in messages]
    except errors.RPCError as e:
        print(f"An error occurred while fetching messages from {channel}: {e}")
        return []

async def return_telegram_messages(api_id, api_hash):
    channels = ['@crypto_gazua', '@shrimp_notice', '@whaleliq', '@whalealertkorean']
    seoul_tz = pytz.timezone('Asia/Seoul')

    async with TelegramClient('/home/ubuntu/venvs/anon_prod.session', api_id, api_hash) as client:
        tasks = [get_telegram_messages(client, channel) for channel in channels]
        messages_lists = await asyncio.gather(*tasks)

    results = {}
    for channel, messages in zip(channels, messages_lists):
        processed_texts = [text + "\n" + date.astimezone(seoul_tz).strftime("%Y-%m-%d %H:%M:%S %Z") for date, text in messages]
        results[channel] = processed_texts

    return results

def get_telegram_messages_sync(api_id, api_hash):
    results = asyncio.run(return_telegram_messages(api_id, api_hash))
    return results


def fetch_fng_data():
    try:
        url_fng = "https://api.alternative.me/fng/?limit=5&date_format=kr"
        response_fng = requests.get(url_fng)
        data_fng = response_fng.json().get('data', [])
    except Exception as e:
        print(f"Failed to fetch FNG data: {e}")
        data_fng = []
    return data_fng

def fetch_global_data():
    try:
        url_global = "https://api.coinlore.net/api/global/"
        response_global = requests.get(url_global)
        data_global = response_global.json()
    except Exception as e:
        print(f"Failed to fetch global data: {e}")
        data_global = {}
    return data_global

def index(request):
    boards = Board.objects.all()
    board_posts = {}
    for board in boards:
        # Fetch the top 3 posts for each board
        posts = Question.objects.filter(board=board).order_by('-create_date')[:4]
        board_posts[board] = posts

    # Get top 4 popular questions
    popular_posts = Question.objects.annotate(
        num_answer=Count("answer"),
        num_voter=Count('voter')
    ).order_by("-num_answer", "-num_voter", "-create_date")[:10]

    kimchi_data = get_kimchi_data()
    print(kimchi_data)

    sentiment_voting_options = VotingOption.objects.annotate(vote_count=Count('votes'))
    sentiment_votes_with_percentages = calculate_vote_percentages(sentiment_voting_options)
    sentiment_data = {
        "labels": [option.name for option in sentiment_voting_options],
        "data": [percentage for _, percentage in sentiment_votes_with_percentages]
    }

    try:
        telegram_messages = get_telegram_messages_sync(api_id, api_hash)
    except Exception as e:
        print(e)
        telegram_messages = {}

    if should_update_prediction():
        print("calculating as we cannot use previously cached value")

        data_fng = fetch_fng_data()
        data_global = fetch_global_data()
        #pearson, spearman, kendall = get_correlation()
        cache.set('data_fng', data_fng, 86400)
        cache.set('data_global', data_global, 86400)  # Expire after one day
        #cache.set('pearson', pearson, 86400),
        #cache.set('spearman', spearman, 86400),
        #cache.set('kendall', kendall, 86400),
        cache.set('last_prediction_update', datetime.now(pytz.timezone('Asia/Seoul')))

    data_fng = cache.get('data_fng', [])
    data_global = cache.get('data_global', {})
    #pearson = cache.get('pearson')
    #spearman = cache.get('spearman')
    #kendall = cache.get('kendall')

    context = {
        "board_posts": board_posts,
        "popular_posts": popular_posts,
        "data_fng": data_fng,
        "data_global": data_global,
        "kimchi_data": kimchi_data,
        "sentiment_voting_options": sentiment_voting_options,
        "sentiment_data": sentiment_data,
        #"pearson": pearson,
        #"spearman": spearman,
        #"kendall": kendall,
        "telegram_messages": telegram_messages,
    }
    # merge context and prediction_contexts then return it as context
    # context = {**context, **prediction_contexts}
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

'''
def detail(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    context = {"question": question}
    return render(request, 'aiphabtc/question_detail.html', context)
'''

def detail(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    page = request.GET.get('page', '1')
    kw = request.GET.get('kw', '')
    so = request.GET.get("so", "recent")

    question_list = Question.objects.filter(board=question.board)

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

    paginator = Paginator(question_list, 10)  # Show 10 questions per page
    page_obj = paginator.get_page(page)

    # for open graph meta tag preview image
    soup = BeautifulSoup(question.content, "html.parser")
    # Find the first image or video in the content
    media_url = None
    first_img = soup.find('img')
    if first_img:
        media_url = request.build_absolute_uri(first_img['src'])
    else:
        first_video = soup.find('video')
        if first_video and first_video.find('source'):
            media_url = request.build_absolute_uri(first_video.find('source')['src'])

    if not media_url:
        media_url = request.build_absolute_uri(static('aiphabtc_mascot.jpeg'))

    context = {
        "question": question,
        "question_list": page_obj,
        "board": question.board,
        'page': page,
        'kw': kw,
        'so': so,
        'media_url': media_url,
        'page_title': question.subject,
    }
    return render(request, 'aiphabtc/question_detail.html', context)


def community_guideline(request):
    return render(request, "guidelines.html", {})


# for perceptive board
def get_current_price(request, ticker):
    try:
        price = pyupbit.get_current_price(ticker)
        print(price)
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



def custom_413_error(request):
    return render(request, "aiphabtc/custom_413.html", status=413)