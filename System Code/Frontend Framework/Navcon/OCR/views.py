from django.shortcuts import render

# Create your views here.

def OCR(request, *args, **kwargs):
    return render (request, 'OCR.html', {})
