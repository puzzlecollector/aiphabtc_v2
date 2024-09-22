from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse
from ..models import Question, Answer, Comment, Board
from django.utils import timezone
from ..forms import QuestionForm, AnswerForm, CommentForm, PerceptiveBoardQuestionForm
from django.core.paginator import Paginator
from django.contrib.auth.decorators import login_required
from django.contrib import messages
import pyupbit
from common.models import PointTokenTransaction
import pytz

@login_required(login_url="common:login")
def question_create(request, board_name):
    board = get_object_or_404(Board, name=board_name)
    if board_name == "perceptive_board":
        form_class = PerceptiveBoardQuestionForm
    else:
        form_class = QuestionForm

    if request.method == "POST":
        form = form_class(request.POST)
        if form.is_valid():
            question = form.save(commit=False)
            question.author = request.user
            seoul_timezone = pytz.timezone("Asia/Seoul")
            question.create_date = timezone.now().astimezone(seoul_timezone)
            print(question.create_date)
            question.board = board

            # Handle perceptive_board specific logic
            if board_name == "perceptive_board":
                market = form.cleaned_data.get('market')
                final_verdict = form.cleaned_data.get('final_verdict')
                duration_from = form.cleaned_data.get('duration_from').strftime('%Y-%m-%d')
                duration_to = form.cleaned_data.get('duration_to').strftime('%Y-%m-%d')
                price_lower_range = form.cleaned_data.get('price_lower_range')
                price_upper_range = form.cleaned_data.get('price_upper_range')
                title_prefix = f"[{market}][{final_verdict}][{duration_from} - {duration_to}] "
                question.subject = title_prefix + question.subject

                # Prepend the form inputs to the content
                content_prefix = f"Market: {market}\nDuration: {duration_from} to {duration_to}\nPrice Range:{price_lower_range}-{price_upper_range}\nVerdict: {final_verdict}\n\n"
                question.content = content_prefix + question.content

            question.save()

            profile = request.user.profile
            score_obtained = 5
            reason = ""
            if board_name == "perceptive_board":
                score_obtained = 5
                profile.score += 5
                reason = "관점 게시글 작성 보상"
            elif board_name == "free_board":
                score_obtained = 2
                profile.score += 2
                reason = "자유 게시판 게시글 작성 보상"
            elif board_name in ["AI", "trading", "blockchain", "economics"]:
                score_obtained = 3
                profile.score += 3
                reason = "전문 게시판 게시글 작성 보상"
            elif board_name in ["general_qa", "creator_reviews"]:
                score_obtained = 4
                profile.score += 4
                reason = "Q&A / 리뷰 게시판 게시글 작성 보상"
            profile.save()
            PointTokenTransaction.objects.create(
                user=request.user,
                points=score_obtained,
                reason=reason
            )
            return redirect("aiphabtc:board_filtered", board_name=board.name)
    else:
        form = form_class()
    context = {"form": form, "board": board, "board_name": board_name}
    return render(request, 'aiphabtc/question_form.html', context)

@login_required(login_url="common:login")
def question_modify(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    if request.user != question.author:
        messages.error(request, "수정권한이 없습니다")
        return redirect("aiphabtc:detail", question_id=question.id)
    if request.method == "POST":
        form = QuestionForm(request.POST, instance=question)
        if form.is_valid():
            question = form.save(commit=False)
            question.author = request.user
            seoul_timezone = pytz.timezone("Asia/Seoul")
            question.modify_date = timezone.now().astimezone(seoul_timezone)
            question.save()
            return redirect("aiphabtc:detail", question_id=question.id)
    else:
        form = QuestionForm(instance=question)
    context = {"form":form}
    return render(request, "aiphabtc/question_form.html", context)

@login_required(login_url="common:login")
def question_delete(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    if request.user != question.author:
        messages.error(request, "삭제권한이 없습니다")
        return redirect("aiphabtc:detail", question_id=question.id)
    board_name = question.board.name if question.board else None
    profile = request.user.profile
    score_obtained = -5
    reason = ""
    if board_name == "perceptive_board":
        profile.score -= 5
        score_obtained = -5
        reason = "관점공유 글 삭제 보상 철회"
    elif board_name == "free_board":
        profile.score -= 2
        score_obtained = -2
        reason = "자유게시판 글 삭제 보상 철회"
    elif board_name in ["AI", "trading", "blockchain", "economics"]:
        profile.score -= 3
        score_obtained = -3
        reason = "전문 게시판 게시글 삭제 보상 철회"
    elif board_name in ["general_qa", "creator_reviews"]:
        profile.score -= 4
        score_obtained = -4
        reason = "Q&A / 리뷰 게시판 게시글 삭제 보상 철회"
    if profile.score < 0:
        profile.score = 0
    profile.save()
    question.delete()

    PointTokenTransaction.objects.create(
        user=request.user,
        points=score_obtained,
        reason=reason
    )
    return redirect("aiphabtc:board_filtered", board_name=board_name)
