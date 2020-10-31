from django.shortcuts import render

# Create your views here.

def NavconUI(request):
    return render (request, 'NavconUI.html', {})

