from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver
from aiphabtc.models import Question, Answer, Comment
import secrets
from django.db import models, IntegrityError, transaction


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    intro = models.CharField(max_length=300, blank=True, null=True, verbose_name="한줄소개")
    image = models.ImageField(upload_to="profile_images/")
    score = models.PositiveIntegerField(default=0)
    referral_code = models.CharField(max_length=12, unique=True, null=True, blank=True)  # Updated length
    referred_by = models.ForeignKey('self', on_delete=models.SET_NULL, null=True, blank=True, related_name="referrals")
    instagram_url = models.URLField(max_length=255, blank=True, null=True)
    twitter_url = models.URLField(max_length=255, blank=True, null=True)
    youtube_url = models.URLField(max_length=255, blank=True, null=True)
    personal_url = models.URLField(max_length=255, blank=True, null=True)

    def get_user_questions(self):
        return Question.objects.filter(author=self.user)

    def get_user_answers(self):
        return Answer.objects.filter(author=self.user)

    def get_user_comments(self):
        return Comment.objects.filter(author=self.user)

    def calculate_score(self):
        score = 0
        user = self.user
        # Add logic to calculate the score based on past posts, answers, comments and likes
        for question in Question.objects.filter(author=user):
            if question.board.name == "free_board":
                score += 2
            elif question.board.name in ["AI", "trading", "blockchain", "economics"]:
                score += 3
            elif question.board.name in ["general_qa", "creator_reviews"]:
                score += 4
            elif question.board.name == "perceptive_board":
                score += 5

        for answer in Answer.objects.filter(author=user):
            if answer.question.board.name in ["general_qa"]:
                score += 4
            elif answer.question.board.name in ["free_board", "AI", "trading", "blockchain", "economics", "creator_reviews", "perceptive_board"]:
                score += 1

        # Calculate score from attendance
        attendance_points = Attendance.objects.filter(user=user).count() * 4  # Assuming 4 points per attendance day
        score += attendance_points
        return score

    def get_tier(self):
        if self.score < 100:
            return "Beginner"
        elif self.score < 200:
            return "Bronze"
        elif self.score < 300:
            return 'Silver'
        elif self.score < 1000:
            return 'Gold'
        elif self.score < 2000:
            return 'Platinum'
        elif self.score < 10000:
            return 'Master'
        else:
            return 'Grandmaster'

    def generate_unique_referral_code(self):
        # Adjusted for desired length without slicing, assuming token_urlsafe(16)
        # more consistently yields a result >=12 characters before the slice.
        # Note: Adjust the input to token_urlsafe as necessary to consistently achieve your desired output length.
        referral_code = secrets.token_urlsafe(16)[:12]  # Adjusted to generate and slice to 12 characters
        while Profile.objects.filter(referral_code=referral_code).exists():
            referral_code = secrets.token_urlsafe(16)[:12]  # Repeat until unique
        return referral_code

    def save(self, *args, **kwargs):
        if not self.referral_code:
            self.referral_code = self.generate_unique_referral_code()
            # Attempt to save with the new referral code in a transaction
            while True:
                try:
                    with transaction.atomic():
                        super(Profile, self).save(*args, **kwargs)
                    break  # If save was successful, break out of the loop
                except IntegrityError:
                    # If an IntegrityError occurred, it could be due to a referral code collision.
                    # Generate a new code and try again.
                    self.referral_code = self.generate_unique_referral_code()
        else:
            # If the referral code is already set, just save the instance normally.
            super(Profile, self).save(*args, **kwargs)

    def __str__(self):
        return self.user.username

@receiver(post_save, sender=User)
def create_or_update_user_profile(sender, instance, created, **kwargs):
    if created:
        profile = Profile.objects.create(user=instance)
    else:
        profile = instance.profile
    profile.score = profile.calculate_score()
    profile.save()

class Attendance(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    check_in_date = models.DateField(auto_now_add=True) # automatically set to the current date
    def __str__(self):
        return f"{self.user.username} - {self.check_in_date}"
    class Meta:
        unique_together = ("user", "check_in_date") # Ensures one entry per day per user


class PointTokenTransaction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="transactions")
    points = models.IntegerField(default=0)
    reason = models.CharField(max_length=500)
    timestamp = models.DateTimeField(auto_now_add=True)
    class Meta:
        ordering = ["-timestamp"]
    def __str__(self):
        return f"{self.user.username} - 포인트: {self.points}, 내역: {self.reason}"