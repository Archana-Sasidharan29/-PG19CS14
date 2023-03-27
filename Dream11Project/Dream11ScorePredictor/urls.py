from django.urls import path
from .import views
urlpatterns = [
    path('',views.guest,name="Guest"),
    path('Sign Up/',views.reg,name="register"),
    path('SignIn/',views.Login,name="login"),
    path('ForgotPassword/',views.forgot,name="forgot"),
    path('Home/',views.home,name="Home"),
    path('Batting/',views.bat,name="bat"),
    path('Bowling/',views.bow,name="bow"),
    path('Logout/',views.Logout,name="logout")
]