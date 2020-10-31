from django.shortcuts import render

# Create your views here.

def OutdoorNavigation(request, *args, **kwargs):
    return render(request, 'OutdoorNavigation.html', {})