from django.urls import path
from caliber_pipe import views

urlpatterns = [
    path('mv-image-analysis/pipe/caliber', views.pipe_caliber),
    path('mv-image-analysis/pipe_depth/caliber', views.pipe_depth_cal)
]