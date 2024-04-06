from .base import *

ALLOWED_HOSTS = []

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'localtest',
        'USER': 'postgres',
        'PASSWORD': 'pwd',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}