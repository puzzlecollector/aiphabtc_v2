from django import forms
from aiphabtc.models import Question, Answer, Comment
from django.utils import timezone
from datetime import timedelta
import pyupbit
import requests # needed for MEXC API

class QuestionForm(forms.ModelForm):
    class Meta:
        model = Question
        fields = ['subject', 'content']
        labels = {
            "subject": "제목",
            "content": "내용",
        }

class AnswerForm(forms.ModelForm):
    class Meta:
        model = Answer
        fields = ['content']
        labels = {
            'content': '답변내용',
        }
        
class CommentForm(forms.ModelForm):
    class Meta:
        model = Comment
        fields = ['content']
        labels = {
            'content': '댓글내용',
        }

import pyupbit
from django.utils import timezone
from datetime import timedelta


# Function to get MEXC tickers for USDT market
# with market cap
import requests
import time
from requests.exceptions import RequestException


# Function to get all available tickers from MEXC
def get_all_tickers():
    url = "https://www.mexc.com/open/api/v2/market/symbols"
    response = requests.get(url)
    data = response.json()
    if data['code'] == 200:
        return data['data']
    else:
        raise Exception("Failed to fetch tickers")


# Function to filter USDT pairs
def get_usdt_tickers(tickers):
    usdt_tickers = [ticker['symbol'] for ticker in tickers if ticker['symbol'].endswith('_USDT')]
    return usdt_tickers


# Sort USDT tickers based on the reference list
def sort_tickers_based_on_market_cap(usdt_tickers, market_cap_ordered_tickers):
    # Create a dictionary for the market cap tickers with their ranking positions
    market_cap_rankings = {ticker: i for i, ticker in enumerate(market_cap_ordered_tickers)}

    # Split the tickers into those that are in the reference and those that aren't
    known_tickers = []
    unknown_tickers = []

    for ticker in usdt_tickers:
        if ticker in market_cap_rankings:
            known_tickers.append(ticker)
        else:
            unknown_tickers.append(ticker)

    # Sort known tickers by their rank in the market cap list
    sorted_known_tickers = sorted(known_tickers, key=lambda x: market_cap_rankings[x])

    # Combine the sorted known tickers with the unknown tickers at the end
    sorted_tickers = sorted_known_tickers + unknown_tickers

    return sorted_tickers


# Function to get MEXC USDT tickers and sort by market cap
def get_mexc_usdt_tickers(retries=3, backoff_factor=1.0):
    tickers = get_all_tickers()
    usdt_tickers = get_usdt_tickers(tickers)

    # Reference list for market cap sorting (as provided)
    market_cap_ordered_tickers = [
        "BTC_USDT", "ETH_USDT", "USDT_USDT", "BNB_USDT", "SOL_USDT", "USDC_USDT",
        "XRP_USDT", "STETH_USDT", "DOGE_USDT", "TON_USDT", "TRX_USDT", "ADA_USDT",
        "AVAX_USDT", "WSTETH_USDT", "WBTC_USDT", "SHIB_USDT", "WETH_USDT", "LINK_USDT",
        "BCH_USDT", "DOT_USDT", "DAI_USDT", "LEO_USDT", "UNI_USDT", "LTC_USDT",
        "NEAR_USDT", "WEETH_USDT", "KAS_USDT", "FET_USDT", "SUI_USDT", "ICP_USDT",
        "APT_USDT", "PEPE_USDT", "XMR_USDT", "TAO_USDT", "FDUSD_USDT", "POL_USDT",
        "XLM_USDT", "ETC_USDT", "STX_USDT", "USDE_USDT", "IMX_USDT", "OKB_USDT",
        "CRO_USDT", "AAVE_USDT", "FIL_USDT", "ARB_USDT", "RENDER_USDT", "INJ_USDT",
        "HBAR_USDT", "MNT_USDT", "OP_USDT", "VET_USDT", "FTM_USDT", "ATOM_USDT",
        "WIF_USDT", "WBT_USDT", "GRT_USDT", "RUNE_USDT", "THETA_USDT", "RETH_USDT",
        "MKR_USDT", "SOLVBTC_USDT", "AR_USDT", "BGB_USDT", "METH_USDT", "SEI_USDT",
        "FLOKI_USDT", "BONK_USDT", "TIA_USDT", "MATIC_USDT", "HNT_USDT", "PYTH_USDT",
        "JUP_USDT", "ALGO_USDT", "GT_USDT", "QNT_USDT", "JASMY_USDT", "OM_USDT",
        "ONDO_USDT", "LDO_USDT", "CORE_USDT", "BSV_USDT", "EZETH_USDT", "WETH_USDT",
        "KCS_USDT", "FLOW_USDT", "POPCAT_USDT", "BTT_USDT", "EETH_USDT", "BEAM_USDT",
        "KLAY_USDT", "EOS_USDT", "BRETT_USDT", "GALA_USDT", "EGLD_USDT", "TKX_USDT",
        "NOT_USDT", "AXS_USDT", "FTN_USDT"
    ]

    # Sort tickers according to market cap ranking
    sorted_usdt_tickers = sort_tickers_based_on_market_cap(usdt_tickers, market_cap_ordered_tickers)

    return sorted_usdt_tickers


class PerceptiveBoardQuestionForm(forms.ModelForm):
    MARKET_TYPE_CHOICES = [
        ('KRW', 'KRW Market (Upbit)'),
        ('USDT', 'USDT Market (MEXC)')
    ]

    market_type = forms.ChoiceField(choices=MARKET_TYPE_CHOICES, label="Market Type", initial='USDT')
    market = forms.ChoiceField(choices=[], label="Ticker")  # Choices will be populated dynamically
    duration_from = forms.DateField(widget=forms.SelectDateWidget(), label="Duration From")
    duration_to = forms.DateField(widget=forms.SelectDateWidget(), label="Duration To")
    price_lower_range = forms.FloatField(label="Price Lower Range")
    price_upper_range = forms.FloatField(label="Price Upper Range")
    final_verdict = forms.ChoiceField(choices=[('bullish', 'Bullish'), ('bearish', 'Bearish')],
                                      widget=forms.RadioSelect, label="Final Verdict")

    class Meta:
        model = Question
        fields = ['subject', 'content', 'market_type', 'market', 'duration_from', 'duration_to', 'price_lower_range',
                  'price_upper_range', 'final_verdict']

    def __init__(self, *args, **kwargs):
        super(PerceptiveBoardQuestionForm, self).__init__(*args, **kwargs)
        today = timezone.now().date()
        self.fields['duration_from'].initial = today
        self.fields['duration_to'].initial = today + timedelta(days=1)

        # Retain the market type from the submitted data, or use the default
        market_type = self.data.get('market_type') or self.fields['market_type'].initial
        self.update_market_choices(market_type)

        # Retain the market ticker from the submitted data if available
        if self.data.get('market'):
            self.fields['market'].initial = self.data.get('market')

    def update_market_choices(self, market_type):
        if market_type == 'KRW':
            tickers = pyupbit.get_tickers(fiat="KRW")
        elif market_type == 'USDT':
            tickers = get_mexc_usdt_tickers()
        else:
            tickers = []

        # Update the 'market' choices dynamically based on the selected market type
        self.fields['market'].choices = [(ticker, ticker) for ticker in tickers]

    def clean(self):
        cleaned_data = super().clean()
        duration_from = cleaned_data.get("duration_from")
        duration_to = cleaned_data.get("duration_to")
        if duration_to and duration_from and duration_to < duration_from:
            self.add_error('duration_to', "Duration 'To date' must be after 'From date'.")
        return cleaned_data

class ImageUploadForm(forms.Form):
    image = forms.ImageField(label='Upload an Image')
    language = forms.ChoiceField(choices=[('en', '🇺🇸 English'), ('ko', '🇰🇷 Korean')], widget=forms.RadioSelect)
