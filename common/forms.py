from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from common.models import Profile
from django import forms
from django.contrib.auth.forms import PasswordChangeForm
from django.utils.translation import gettext_lazy as _
from django.contrib.auth.forms import PasswordResetForm, SetPasswordForm
from django import forms
from django.core.exceptions import ValidationError

class UserForm(UserCreationForm):
    email = forms.EmailField(label=_("이메일"), required=True)

    class Meta:
        model = User
        fields = ("username", "email")

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise ValidationError(_("이미 존재하는 이메일입니다!"))
        return email

# Adding new form for introduction + social profile links
class ProfileForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['intro', 'instagram_url', 'twitter_url', 'youtube_url', 'personal_url']
        widgets = {
            'intro': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Write a brief introduction here.'}),
            'instagram_url': forms.URLInput(attrs={'class': 'form-control', 'placeholder': 'Instagram URL'}),
            'twitter_url': forms.URLInput(attrs={'class': 'form-control', 'placeholder': 'Twitter URL'}),
            'youtube_url': forms.URLInput(attrs={'class': 'form-control', 'placeholder': 'YouTube URL'}),
            'personal_url': forms.URLInput(attrs={'class': 'form-control', 'placeholder': 'Personal URL'}),
        }

class CustomPasswordChangeForm(PasswordChangeForm):
    old_password = forms.CharField(
        label=_("Current Password"),
        strip=False,
        widget=forms.PasswordInput(attrs={'autocomplete': 'current-password', 'autofocus': True}),
    )
    new_password1 = forms.CharField(
        label=_("New Password"),
        widget=forms.PasswordInput(attrs={'autocomplete': 'new-password'}),
        strip=False,
    )
    new_password2 = forms.CharField(
        label=_("Confirm New Password"),
        strip=False,
        widget=forms.PasswordInput(attrs={'autocomplete': 'new-password'}),
    )
