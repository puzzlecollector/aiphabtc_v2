from django.shortcuts import render

def trading_bot_indicator(request):
    """Render the 'Coming Soon!' page."""
    return render(request, 'aiphabtc/bot_indicator.html', {})
