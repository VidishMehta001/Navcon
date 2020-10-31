from django.shortcuts import render

# Create your views here.

def contactView(request, *args, **kwargs):
    return render(request, 'contact.html', {})
