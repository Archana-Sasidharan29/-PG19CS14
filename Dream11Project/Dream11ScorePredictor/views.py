from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
import pandas as pd
import pickle
# Create your views here.
def guest(request):
    return render(request,'Guest.html')

def reg(request):
    if request.method == 'POST':
        fname = request.POST.get('fname')
        lname = request.POST.get('lname')
        email = request.POST.get('email')
        username = request.POST.get('uname')
        pass1 = request.POST.get('pass')
        pass2 = request.POST.get('re_pass')
        print(fname, lname, email, username, pass1, pass2)
        if pass1 == pass2:
            if User.objects.filter(username=username).exists():
                messages.warning(request, "Username already Exists")
            elif User.objects.filter(email=email).exists():
                messages.warning(request, "Email Exists")
            elif len(pass1) < 6:
                messages.warning(request, "Minimum password length is 6")
            elif len(pass1) > 8:
                messages.warning(request, "Maximun password length is 8")
            else:
                my_user = User.objects.create_user(username=username, email=email, password=pass1, first_name=fname,
                                                   last_name=lname)
                my_user.save()
                print("user created")
                return redirect(home)

        else:
            messages.warning(request, "Password mismatch")
    return render(request,'Register.html')

def Login(request):
    if request.method == 'POST':
        username = request.POST.get('uname')
        password = request.POST.get('pass')
        print(username, password)
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            print("logged in")
            return redirect(home)
        else:
            messages.warning(request, "Invalid Credentials..")
    return render(request, 'login.html')


def forgot(request):
    if request.method == "POST":
        username = request.POST.get('user')
        pass1 = request.POST.get('pass')
        pass2 = request.POST.get('re_pass')
        print(username, pass1, pass2)
        if pass1 == pass2:
            if len(pass1) < 6:
                messages.warning(request, "Minimum password length must be 6")
            elif len(pass1) > 8:
                messages.warning(request, "Maximum password length must be 8")
            elif User.objects.filter(username=username).exists():
                us = User.objects.get(username=username)
                us.set_password(pass1)
                us.save()
                print("password changed")
                return redirect(home)
            else:
                messages.warning(request, "User does not exists")
        else:
            messages.warning(request, "Password Mismatch")
    return render(request, 'forgotPass.html')


def home(request):
    return render(request, 'home.html')

def bat(request):
    return render(request,'batting.html')

def bow(request):
    return render(request,'bowling.html')

def Logout(request):
    logout(request)
    return redirect(guest)
