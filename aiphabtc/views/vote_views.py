from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect
from ..models import Question, Answer, Board, Vote, VotingOption
from common.models import PointTokenTransaction
from django.http import JsonResponse


@login_required(login_url="common:login")
def vote_question(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    if request.user == question.author:
        # messages.error(request, "본인의 글을 추천할 수 없습니다.")
        return JsonResponse({'success': False, 'error': "본인의 글은 추천할 수 없습니다"}, status=403)
    if question.voter.filter(id=request.user.id).exists():
        question.voter.remove(request.user)
        voted = False
    else:
        question.voter.add(request.user)
        voted = True
    return JsonResponse({'success': True, 'voted': voted, 'total_votes': question.voter.count()})

@login_required(login_url="common:login")
def vote_answer(request, answer_id):
    answer = get_object_or_404(Answer, pk=answer_id)
    if request.user == answer.author:
        # messages.error(request, "본인이 작성한 글을 추천할 수 없습니다.")
        return JsonResponse({'success': False, 'error': "본인이 작성한 글을 추천할 수 없습니다"}, status=403)
    if answer.voter.filter(id=request.user.id).exists():
        answer.voter.remove(request.user)
        voted = False
    else:
        answer.voter.add(request.user)
        voted = True
    return JsonResponse({'success': True, 'voted': voted, 'total_votes': answer.voter.count()})
