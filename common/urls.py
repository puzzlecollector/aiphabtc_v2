from django.urls import path
from django.contrib.auth import views as auth_views
from . import views
from .views import CustomLoginView

app_name = 'common'

urlpatterns = [
    path('login/', CustomLoginView.as_view(), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('signup/', views.signup, name='signup'),
    path('findid/', views.find_id_view, name='findid'),
    path('findpassword/', views.find_password_view, name='findpassword'),

    # account settings
    path("settings/base/", views.base, name="settings_base"),
    # path("settings/password_change/", settings_views.PasswordChangeView.as_view(), name="password_change),
    path("settings/image/", views.profile_modify_image, name="settings_image"),
    # path("settings/image/delete/", settings_views.profile_image_delete, name="settings_image_delete"),
    path('settings/password_reset/', views.password_reset, name='password_reset'),

    # account page
    path("account_page/", views.account_page, name="account_page"),

    # view user questions, answers and comments
    path('user/<int:user_id>/questions/', views.user_questions, name='user_questions'),
    path('user/<int:user_id>/answers/', views.user_answers, name='user_answers'),
    path('user/<int:user_id>/comments/', views.user_comments, name='user_comments'),

    # view user ranking
    path("ranking/", views.user_ranking, name="user_ranking"),

    # view user attendance
    path('attendance/', views.attendance_check, name='attendance'),
    
    # view user referral 
    path('referral/', views.referral_view, name='referral'),

    # view user transaction
    path('transactions/<int:transaction_id>/', views.transaction_detail, name='transaction_detail'),
]