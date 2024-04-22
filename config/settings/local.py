from .base import *

# ALLOWED_HOSTS = ['localhost', '127.0.0.1']
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

# DEBUG=True