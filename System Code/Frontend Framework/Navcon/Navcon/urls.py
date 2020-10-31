"""Navcon URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from accounts.views import loginView, registerView, logoutView, baseView
from home.views import home
from demo.views import index, create_dataset, eigenTrain, detect, SVMDetection, video_feed
from contact.views import contactView
from navconui.views import NavconUI
from ObjectDetection.views import objectDetection
from indoorNavigation.views import IndoorNavigation
from OCR.views import OCR
from outdoorNavigation.views import OutdoorNavigation
from user_profile.views import UserProfile
from object_cloud.views import ObjectCloudView
from googleImageDownload.views import getImages

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/login/', loginView, name='login'),
    path('accounts/register/', registerView, name='signup'),
    path('accounts/logout', logoutView, name='logout'),
    path('', home, name = 'home'),
    path('demo', index, name='demo'),
    path('createDataset', create_dataset, name='createDataset'),
    path('eigenTrain', eigenTrain, name = 'eigenTrain'),
    path('contact', contactView, name='contact'),
    path('detect', detect, name='detect'),
    path('svmdetect', SVMDetection, name='svmdetect'),
    path('Navcon', NavconUI, name='Navcon'),
    path('video_feed', video_feed, name = 'video_feed'),
    path('objectDetection', objectDetection, name='objectDetection'),
    path('IndoorNavigation', IndoorNavigation, name='IndoorNavigation'),
    path('ocr', OCR, name='ocr'),
    path('OutdoorNavigation', OutdoorNavigation, name='OutdoorNavigation'),
    path('UserProfile', UserProfile, name="UserProfile"),
    path('ObjectCloud', ObjectCloudView, name="ObjectCloud"),
    path('DownloadImages', getImages, name="DownloadImages")
]
