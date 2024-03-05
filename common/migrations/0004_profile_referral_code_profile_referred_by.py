# Generated by Django 4.2.9 on 2024-03-05 07:42

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('common', '0003_attendance'),
    ]

    operations = [
        migrations.AddField(
            model_name='profile',
            name='referral_code',
            field=models.CharField(blank=True, max_length=12, null=True, unique=True),
        ),
        migrations.AddField(
            model_name='profile',
            name='referred_by',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='referrals', to='common.profile'),
        ),
    ]
