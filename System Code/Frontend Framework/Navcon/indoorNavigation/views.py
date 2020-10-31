from django.shortcuts import render

# Create your views here.

def IndoorNavigation(request, *args, **kwargs):
    return render(request, 'IndoorNavigation.html', {})