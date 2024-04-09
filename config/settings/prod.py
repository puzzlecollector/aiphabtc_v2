from .base import *

ALLOWED_HOSTS = ["43.200.37.42", "aiphabtc-v2.com", "www.aiphabtc-v2.com"]

STATIC_ROOT = BASE_DIR / "static/"
STATICFILES_DIRS = []
DEBUG = False

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'aiphabtc',
        'USER': 'dbmasteruser',
        'PASSWORD': '^HOJ{H}6Q4G{Zbr`#d3J?Nqwy][nsG:p',
        'HOST': 'ls-f43abde185ef373bcd6253588b8b9e0dd3bdafc2.cdca0i6ucaaq.ap-northeast-2.rds.amazonaws.com',
        'PORT': '5432',
    }
}

