from django.urls import path
from .views import base_views, question_views, answer_views, comment_views, vote_views, profile_click_views, pattern_matching_views

app_name = 'aiphabtc'

urlpatterns = [
    # base_views.py
    #path("", base_views.loading, name="loading"),
    path("", base_views.index, name="index"), # main landing page
    path("<int:question_id>/", base_views.detail, name="detail"),
    path("board/<str:board_name>/", base_views.index_orig, name="board_filtered"),
    path("news-sentiment/", base_views.get_news_and_sentiment, name="get_news_and_sentiment"),

    # question_views.py
    path("question/create/<str:board_name>", question_views.question_create, name="question_create"),
    path("question/modify/<int:question_id>/", question_views.question_modify, name="question_modify"),
    path("question/delete/<int:question_id>/", question_views.question_delete, name="question_delete"),

    # answer_views.py
    path("answer/create/<int:question_id>/", answer_views.answer_create, name="answer_create"),
    path("answer/modify/<int:answer_id>/", answer_views.answer_modify, name="answer_modify"),
    path("answer/delete/<int:answer_id>/", answer_views.answer_delete, name="answer_delete"),

    # comment_views.py
    path("comment/create/question/<int:question_id>/", comment_views.comment_create_question, name="comment_create_question"),
    path("comment/modify/question/<int:comment_id>/", comment_views.comment_modify_question, name="comment_modify_question"),
    path("comment/delete/question/<int:comment_id>/", comment_views.comment_delete_question, name="comment_delete_question"),
    path("comment/create/answer/<int:answer_id>/", comment_views.comment_create_answer, name="comment_create_answer"),
    path("comment/modify/answer/<int:comment_id>/", comment_views.comment_modify_answer, name="comment_modify_answer"),
    path("comment/delete/answer/<int:comment_id>/", comment_views.comment_delete_answer, name="comment_delete_answer"),

    # vote_views.py
    path("vote/question/<int:question_id>/", vote_views.vote_question, name="vote_question"),
    path("vote/answer/<int:answer_id>/", vote_views.vote_answer, name="vote_answer"),

    # view other profiles
    path('profile/<int:user_id>/', profile_click_views.profile_detail, name='profile_detail'),

    # ai technical response
    path("fetch-gpt-analysis/", base_views.fetch_ai_technical1d, name="fetch_gpt_analysis"),

    # sentiment votes
    path("submit_sentiment_vote/", base_views.submit_sentiment_vote, name="submit_sentiment_vote"),

    # getting current price of ticker
    path('get-current-price/<str:ticker>/', base_views.get_current_price, name='get_current_price'),

    # getting main landing page search results
    path('search/', base_views.search_results, name='search_results'),

    # view community guideline
    path('guidelines/', base_views.community_guideline, name="guideline"),

    # dynamically update votes
    path('latest_voting_data/', base_views.latest_voting_data, name='latest_voting_data'),

    # pattern matching view
    path('pattern_matching_tools/', pattern_matching_views.pattern_matching_views, name="pattern_matching"),

    # news similarity
    path('news_similarity/', pattern_matching_views.search_news, name="news_similarity"),
]