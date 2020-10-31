from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required


# Create your views here.

@login_required
def home(request, *args, **kwargs):
    if request.method == "GET":
        return render(request, 'index.html', {})
    