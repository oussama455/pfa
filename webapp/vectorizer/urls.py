from django.urls import path

from . import views

app_name = 'vectorizer'

urlpatterns = [
    path('',                      views.upload_view, name='upload'),
    path('maps/<int:pk>/',        views.result_view, name='result'),
    path('maps/<int:pk>/status/', views.status_view, name='status'),
]
