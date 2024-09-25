from django.shortcuts import render, get_object_or_404, redirect, resolve_url
from django.http import HttpResponse
from ..models import Question, Answer, Comment
from django.utils import timezone
from ..forms import QuestionForm, AnswerForm, CommentForm
from django.core.paginator import Paginator
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from common.models import PointTokenTransaction



@login_required(login_url="common:login")
def answer_create(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    if request.method == "POST":
        form = AnswerForm(request.POST)
        if form.is_valid():
            answer = form.save(commit=False)
            answer.author = request.user
            answer.create_date = timezone.now()
            answer.question = question
            answer.save()

            # Update score based on the board associated with the answer's question
            score_obtained = 1
            reason = ""
            profile = request.user.profile
            if question.board.name in ["general_qa"]:
                profile.score += 4
                score_obtained = 4
                reason = "Reward for writing a reply for general qa board"
            else:
                profile.score += 1  # Default score for other boards
                score_obtained = 1
                reason = "Reward for writing a reply"
            profile.save()
            PointTokenTransaction.objects.create(
                user=request.user,
                points=score_obtained,
                reason=reason
            )
            return redirect("{}#answer_{}".format(resolve_url("aiphabtc:detail", question_id=question.id), answer.id))
    else:
        form = AnswerForm()
    context = {'question': question, 'form': form}
    return render(request, 'aiphabtc/question_detail.html', context)

@login_required(login_url="common:login")
def answer_modify(request, answer_id):
    answer = get_object_or_404(Answer, pk=answer_id)
    if request.user != answer.author:
        messages.error(request, "수정권한이 없습니다")
        return redirect("aiphabtc:detail", question_id=answer.question.id)
    if request.method == "POST":
        form = AnswerForm(request.POST, instance=answer)
        if form.is_valid():
            answer = form.save(commit=False)
            answer.author = request.user
            answer.modify_date = timezone.now()
            answer.save()
            return redirect("{}#answer_{}".format(resolve_url("aiphabtc:detail", question_id=answer.question.id), answer.id))
    else:
        form = AnswerForm(instance=answer)
    context = {"answer":answer, "form":form}
    return render(request, "aiphabtc/answer_form.html", context)

@login_required(login_url='common:login')
def answer_delete(request, answer_id):
    answer = get_object_or_404(Answer, pk=answer_id)
    if request.user != answer.author:
        messages.error(request, "삭제권한이 없습니다")
    else:
        # Before deleting the answer, adjust the user's score
        profile = request.user.profile
        score_obtained = -1
        reason = ""
        if answer.question.board.name in ["general_qa"]:
            profile.score -= 4  # Subtract the points for a general_qa board
            score_obtained = -4
            reason = "Reward retracted for deleting a reply from general qa board"
        else:
            profile.score -= 1  # Default subtraction for other boards
            score_obtained = -1
            reason = "Reward retracted for deleting a reply"
        if profile.score < 0:
            profile.score = 0
        profile.save()
        answer.delete()
        PointTokenTransaction.objects.create(
            user=request.user,
            points=score_obtained,
            reason=reason
        )
    return redirect("aiphabtc:detail", question_id=answer.question.id)