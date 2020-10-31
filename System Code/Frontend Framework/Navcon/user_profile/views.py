from django.shortcuts import render, redirect

# Create your views here.

def UserProfile(request, *args, **kwargs):
    return render(request, 'UserProfile.html', {})