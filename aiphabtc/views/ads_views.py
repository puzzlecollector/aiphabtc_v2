from django.http import HttpResponse

def ads_txt(request):
    with open("aiphabtc/ads.txt", "r") as f:
        file_content = f.read()
    return HttpResponse(file_content, content_type="text/plain")