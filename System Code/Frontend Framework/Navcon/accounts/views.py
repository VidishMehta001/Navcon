from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout, get_user_model
from .forms import UserLoginForm, UserRegisterForm

# Create your views here.

# Simple view for the base html file
def baseView(request):
    return render(request, 'base.html', {})

def loginView(request):
    next = request.GET.get('next')
    form = UserLoginForm(request.POST or None)
    if form.is_valid():
        username = form.cleaned_data.get('username')
        password = form.cleaned_data.get('password')
        print(username, password)
        user = authenticate(username=username, password=password)
        login(request, user)
        if next:
            return redirect(next)
        return redirect('ObjectCloud')

    return render(request, 'login.html', {'form': form})


def registerView(request):
    next = request.GET.get('next')
    form = UserRegisterForm(request.POST or None)

    if form.is_valid():
        user = form.save(commit=False)
        password = form.cleaned_data.get('password')
        user.set_password(password)
        user.save()
        new_user = authenticate(username=user.username, password = password)
        login(request, new_user)
        if next:
            return redirect(next)
        return redirect('ObjectCloud')

    return render(request, 'signup.html', {'form': form})

def logoutView(request):
    logout(request)
    return redirect("/")