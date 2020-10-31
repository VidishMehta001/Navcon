from django.shortcuts import render

# Create your views here.

def objectDetection(request, *args, **kwargs):
    return render (request, 'ObjectDetection.html', {})