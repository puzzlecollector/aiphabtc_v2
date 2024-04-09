from django.contrib import admin
from django.urls import path, include
from aiphabtc.views import base_views
from django.conf import settings
from django.conf.urls.static import static
from common.views import csrf_failure


urlpatterns = [
                  path('admin/', admin.site.urls),
                  path('aiphabtc/', include('aiphabtc.urls')),
                  path('common/', include('common.urls')),
                  path('', base_views.loading, name='loading'),
              ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

handler404 = 'common.views.page_not_found'
handler403 = 'common.views.csrf_failure'