from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from common.forms import UserForm, ProfileForm
from common.models import Profile, Attendance, PointTokenTransaction
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ValidationError
from django.core.files.images import get_image_dimensions
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from common.forms import CustomPasswordChangeForm
from django.utils import timezone
from django.contrib.auth.models import User
from datetime import timedelta
from django.utils.timezone import now, localtime
from aiphabtc.models import Question, Answer, Comment
from django.core.paginator import Paginator
from django.utils.safestring import mark_safe
import requests
import json
from django.db import models, IntegrityError, transaction
from django.contrib.auth.views import LoginView
from django.core.exceptions import ObjectDoesNotExist
from django.contrib.auth.hashers import make_password
import os
from django.http import HttpResponseForbidden
from django.middleware.csrf import CsrfViewMiddleware


@login_required(login_url='common:login')
def base(request):
    # account settings base page
    if request.method == "POST":
        # Instantiate the form with the posted data and files (if there are any)
        form = ProfileForm(request.POST, request.FILES, instance=request.user.profile)
        if form.is_valid():
            form.save()
            messages.success(request, '프로필이 정상적으로 업데이트 되었습니다!')
            return redirect("common:settings_base")
    else:
        # Instantiate the form with the current user's profile data
        form = ProfileForm(instance=request.user.profile)
    context = {'settings_type': 'base', 'form': form}
    return render(request, 'common/settings/base.html', context)


@login_required(login_url='common:login')
def account_page(request):
    user = request.user
    profile, created = Profile.objects.get_or_create(user=user)

    user_questions = Question.objects.filter(author=user)
    user_answers = Answer.objects.filter(author=user)
    user_comments = Comment.objects.filter(author=user)

    context = {
        'user': user,
        'profile': profile,
        'questions': user_questions,
        'answers': user_answers,
        'comments': user_comments
    }
    return render(request, 'common/account_page.html', context)


@login_required(login_url='common:login')
def user_questions(request, user_id):
    # Fetch the user based on the passed user_id
    user = get_object_or_404(User, pk=user_id)
    questions_list = Question.objects.filter(author=user).order_by('-create_date')
    # Set up pagination
    paginator = Paginator(questions_list, 10)  # 10 questions per page
    page_number = request.GET.get('page')
    questions = paginator.get_page(page_number)
    profile = user.profile
    return render(request, 'common/user_questions.html', {'questions': questions, 'profile': profile})


@login_required(login_url='common:login')
def user_answers(request, user_id):
    user = get_object_or_404(User, pk=user_id)
    answers_lists = Answer.objects.filter(author=user).order_by('-create_date')
    paginator = Paginator(answers_lists, 10)  # 10 answers per page
    page_number = request.GET.get('page')
    answers = paginator.get_page(page_number)
    profile = user.profile
    return render(request, 'common/user_answers.html', {'answers': answers, 'profile': profile})


@login_required(login_url='common:login')
def user_comments(request, user_id):
    user = get_object_or_404(User, pk=user_id)
    comments_list = Comment.objects.filter(author=user).order_by('-create_date')
    paginator = Paginator(comments_list, 10)  # 10 comments per page
    page_number = request.GET.get('page')
    comments = paginator.get_page(page_number)
    profile = user.profile
    return render(request, 'common/user_comments.html', {'comments': comments, 'profile': profile})


@login_required(login_url='common:login')
def profile_modify_image(request):
    if request.method == "POST" and 'profile_picture' in request.FILES:
        profile_picture = request.FILES["profile_picture"]
        try:
            if not profile_picture.name.endswith(('.png', '.jpg', '.jpeg')):
                raise ValidationError("Invalid file type: Accepted file types are .png, .jpg, .jpeg")
            width, height = get_image_dimensions(profile_picture)
            max_dimensions = 800
            if width > max_dimensions or height > max_dimensions:
                raise ValidationError("Invalid image size: Max dimensions are 800x800px")
            # Save image to user's profile
            user_profile = request.user.profile  # assumes a related_name of 'profile'
            user_profile.image = profile_picture
            user_profile.save()

            messages.success(request, "Profile picture updated successfully!")
            return redirect('common:settings_base')  # Redirect to a different view after success
        except ValidationError as e:
            messages.error(request, f"Upload failed: {str(e)}")
    elif request.method == "POST":
        messages.error(request, "Something went wrong.")

    return render(request, 'common/settings/profile_picture.html')  # Render the template for image upload


@login_required(login_url="common:login")
def password_reset(request):
    if request.method == 'POST':
        form = CustomPasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, '비밀번호가 정상적으로 변경되었습니다.')
            return redirect('common:settings_base')
    else:
        form = CustomPasswordChangeForm(request.user)
    return render(request, 'common/settings/password_reset.html', {'form': form})


def signup(request):
    if request.user.is_authenticated:
        return redirect("aiphabtc:index") # redirect to the home page if the user is already logged in
    if request.method == "POST":
        form = UserForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get("username")
            raw_password = form.cleaned_data.get("password1")
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            return redirect("aiphabtc:index")
    else:
        form = UserForm()
    return render(request, "common/signup.html", {"form": form})


class CustomLoginView(LoginView):
    template_name = 'common/login.html'  # template we are using for login
    def get(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            return redirect('aiphabtc:index')  # Adjust the redirect as needed
        return super().get(request, *args, **kwargs)



# needed for live run
def page_not_found(request, exception):
    return render(request, 'common/404.html', {})

def csrf_failure(request, reason=""):
    # Optionally, you can use 'reason' to provide more details about the failure
    context = {
        "reason": reason,
    }
    # Render your custom CSRF failure template with the context
    return render(request, "common/csrf_failure.html", context, status=403)



def user_ranking(request):
    search_query = request.GET.get("search", "")
    if search_query:
        profiles = Profile.objects.filter(user__username__icontains=search_query).order_by("-score")
    else:
        profiles = Profile.objects.all().order_by("-score")

    # Pagination
    paginator = Paginator(profiles, 10)  # Show 10 profiles per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    profiles = Profile.objects.all().order_by("-score")
    context = {"profiles": profiles, "page_obj": page_obj, "search_query": search_query}
    return render(request, "common/user_ranking.html", context)


@login_required(login_url="common:login")
def attendance_check(request):
    user = request.user
    profile, created = Profile.objects.get_or_create(user=user)  # Ensure a profile exists
    today = timezone.localdate()
    already_checked_in = Attendance.objects.filter(user=user, check_in_date=today).exists()

    if request.method == "POST" and not already_checked_in:
        Attendance.objects.create(user=user)
        # profile.score += 4  # Update score on profile, not directly on user
        new_score = profile.calculate_score()
        profile.score = new_score
        profile.save()  # Save the profile after modifying

        PointTokenTransaction.objects.create(
            user=user,
            points=4,
            reason="출석체크 보상"
        )

        return redirect("common:attendance")

    checked_in_dates = Attendance.objects.filter(user=user).values_list("check_in_date", flat=True)
    checked_in_dates_json = json.dumps([date.strftime("%Y-%m-%d") for date in checked_in_dates])

    user_tier = profile.get_tier()  # Use the get_tier method from the profile
    attendance_summary = {
        "initial_registration_date": user.date_joined,  # Use date_joined from User model
        "days_since_registration": (today - user.date_joined.date()).days,
        "days_checked": Attendance.objects.filter(user=user).count(),
        "points_from_attendance": user_tier,  # Use score from profile
    }
    context = {
        "attendance_summary": attendance_summary,
        "already_checked_in": already_checked_in,
        "today": today,
        "logged_in": request.user.is_authenticated,  # Correct typo in is_authenticated
        "checked_in_dates": mark_safe(checked_in_dates_json),
    }
    return render(request, "common/attendance_check.html", context)


def referral_view(request):
    if not request.user.is_authenticated:
        return redirect('common:login')
    user_profile = request.user.profile
    if request.method == "POST":
        referral_code = request.POST.get('referral_code').strip()
        # check if the user has already used a referral code
        if user_profile.referred_by is not None:
            messages.error(request, '이미 추천 코드를 사용하셨습니다.')
        # check if the referral code is the user's own code
        elif referral_code == user_profile.referral_code:
            messages.error(request, "본인의 레퍼럴 코드를 사용할 수 없어요!")
        elif referral_code:
            try:
                referrer_profile = Profile.objects.get(referral_code=referral_code)
                # additional check to prevent self referral
                if referrer_profile.user == request.user:
                    messages.error(request, '본인의 레퍼럴 코드를 사용할 수 없어요!')
                    return redirect('referral')
                user_profile.referred_by = referrer_profile
                user_profile.score += 50
                referrer_profile.score += 50
                user_profile.save()
                referrer_profile.save()
                PointTokenTransaction.objects.create(
                    user=user_profile,
                    points=50,
                    reason="친구추천 보상"
                )
                PointTokenTransaction.objects.create(
                    user=referrer_profile,
                    points=50,
                    reason="친구추천 보상"
                )
                messages.success(request, '추천 코드가 승인되었습니다. 귀하와 추천인 모두 포인트를 받게 되었습니다.')
            except Profile.DoesNotExist:
                messages.error(request, '유효하지 않은 레퍼릴 코드입니다!')
    else:
        referral_code = None
    context = {
        "referral_code": user_profile.referral_code,
        "has_referred": user_profile.referred_by is not None
    }
    return render(request, 'common/referral.html', context)


@login_required(login_url='common:login')
def transaction_detail(request, transaction_id):
    transaction = get_object_or_404(PointTokenTransaction, id=transaction_id, user=request.user)
    transaction_list = PointTokenTransaction.objects.filter(user=request.user).order_by('-timestamp')
    paginator = Paginator(transaction_list, 10)  # 10 transactions per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    return render(request, 'common/transaction_detail.html', {'transaction': transaction, 'transaction_list': page_obj})


def find_id_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        try:
            user = User.objects.get(email=email)  # Search for the user by email
            send_simple_message(user.email, user.username)  # Send email with the username
            messages.success(request, "이메일이 발송 되었습니다.")
        except ObjectDoesNotExist:
            messages.error(request, "해당 이메일이 존재 하지 않습니다.")
            return redirect('aiphabtc:index')
        return redirect('aiphabtc:index')  # Redirect back to the find ID page or to a success page
    else:
        return render(request, 'common/findid.html')


def send_simple_message(email_to, username):
    mailgun_api_key = os.environ.get("MAILGUN_API_KEY")
    res = requests.post(
        "https://api.mailgun.net/v3/mail.aiphabtc-v2.com/messages",
        auth=("api", mailgun_api_key),
        data={"from": "AIPHABTC 고객응대 <mailgun@mail.aiphabtc-v2.com>",
              "to": [email_to],
              "subject": "AIPHABTC - 아이디찾기 아이디 전달 드립니다.",
              "template": "findid",
              "h:X-Mailgun-Variables": json.dumps({"username": username}, ensure_ascii=False)
              })
    print("findid: ", res.text)  # DEBUG
    return True


def generate_password(length=12):
    import secrets
    import string
    """Generate a random password"""
    alphabet = string.ascii_letters + string.digits
    password = ''.join(secrets.choice(alphabet) for i in range(length))
    return password

def send_simple_message2(email_to, username, password):
    mailgun_api_key = os.environ.get("MAILGUN_API_KEY")
    res = requests.post(
        "https://api.mailgun.net/v3/mail.aiphabtc-v2.com/messages",
        auth=("api", mailgun_api_key),
        data={"from": "AIPHABTC 고객응대 <mailgun@mail.aiphabtc-v2.com>",
              "to": [email_to],
              "subject": "AIPHABTC - 비밀번호 찾기의 임시 비밀번호 전달 드립니다.",
              "template": "findpassword",
              "h:X-Mailgun-Variables": json.dumps({"username": username, "password": password}, ensure_ascii=False)
              })
    print("!!!================= findpassword   ", res)
    return True


def find_password_view(request):
    if request.method == 'POST':
        username = request.POST.get('userid')
        email = request.POST.get('email')
        try:
            user = User.objects.get(username=username, email=email)  # Find user by ID and email
            new_password = generate_password()  # Generate a new password
            user.password = make_password(new_password)  # Hash and set the new password
            user.save()  # Save the user object to update the password in the database
            send_simple_message2(user.email, user.username, new_password)  # Send email with the new password
            messages.success(request, "이메일이 발송 되었습니다.")
        except ObjectDoesNotExist:
            messages.error(request, "해당 유저가 존재 하지 않습니다.")
            return redirect('aiphabtc:index')
        return redirect('aiphabtc:index')  # Redirect back to the find ID page or to a success page
    else:
        # If not a POST request, simply render the Find Password page
        return render(request, 'common/findpassword.html')
