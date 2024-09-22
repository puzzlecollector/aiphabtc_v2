# views.py
import base64
import os
import requests
from django.shortcuts import render
from ..forms import ImageUploadForm
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from io import BytesIO
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.http import JsonResponse


OPENAI_API_KEY = settings.OPENAI_API_KEY

def encode_image_to_base64(image):
    return base64.b64encode(image.read()).decode('utf-8')

def call_openai_image_api(encoded_image, language):
    if language == 'ko':
        content_text = "첨부된 차트를 자세히 분석해 주세요."
        analysis_prompt = "분석 시, 가능하면 추세, 패턴, 지지 및 저항 수준, 이동 평균 또는 거래량과 같은 지표를 포함하여 향후 가격 변동에 대한 통찰을 제공할 수 있는 요소들을 포함해 주세요."
    else:
        content_text = "Please analyze the attached chart in detail."
        analysis_prompt = "When analyzing, if possible, please include indicators such as trends, patterns, support and resistance levels, moving averages, or volume that can provide insights into future price movements."

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": content_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}",
                            "detail": "high"
                        },
                    }
                ],
            },
            {
                "role": "user",
                "content": analysis_prompt
            }
        ],
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()


def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            language = form.cleaned_data['language']

            # save the uploaded image using file system storage
            # Create a copy of the image in memory
            image_copy = BytesIO(image.read())  # Create a copy of the image in memory
            image.seek(0)  # Reset the original image's file pointer to the beginning
            fs = FileSystemStorage()
            filename = fs.save(image.name, image_copy)
            uploaded_image_url = fs.url(filename)

            # Store the image filename in the session for future deletion
            request.session["uploaded_image_name"] = filename

            # Encode the uploaded image to base64
            encoded_image = base64.b64encode(image_copy.getvalue()).decode('utf-8')

            # Call OpenAI API for analysis
            api_response = call_openai_image_api(encoded_image, language)

            # Extract the analysis text from API response
            analysis_text = api_response['choices'][0]['message']['content']

            return render(request, 'aiphabtc/upload_chart_image.html', {
                'form': form,
                'analysis_text': analysis_text,
                'uploaded_image_url': uploaded_image_url,
            })
    else:
        form = ImageUploadForm()

    return render(request, 'aiphabtc/upload_chart_image.html', {'form': form})


def delete_uploaded_image(request):
    """
    View to handle the deletion of the uploaded image when the user navigates away.
    """
    if request.method == 'POST':
        if 'uploaded_image_name' in request.session:
            image_path = os.path.join(settings.MEDIA_ROOT, request.session['uploaded_image_name'])
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                    del request.session['uploaded_image_name']
                    return JsonResponse({'status': 'success'}, status=200)
                except Exception as e:
                    return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
            else:
                return JsonResponse({'status': 'error', 'message': 'File not found'}, status=404)
    return JsonResponse({'status': 'error', 'message': 'Invalid request'}, status=400)
