from django import forms
from django.contrib.auth import authenticate, get_user_model

User = get_user_model()

class UserLoginForm(forms.Form):
    username = forms.CharField()
    password = forms.CharField(widget = forms.PasswordInput)

    def clean(self, *args, **kwargs):
        username = self.cleaned_data.get("username")
        password = self.cleaned_data.get("password")

        if username and password:
            print(username, password)
            user = authenticate(username=username, password=password)
            if not user:
                raise forms.ValidationError("Username does not exists!")
#            if user.check_password(password):
#                raise forms.ValidationError("Incorrect Password!")
            
        return super(UserLoginForm, self).clean(*args, **kwargs)


class UserRegisterForm(forms.ModelForm):
    email = forms.EmailField(label="Enter Email Address")
    email2 = forms.EmailField(label="Confirm Email Address")
    password = forms.CharField(widget=forms.PasswordInput)

    class Meta:
        model = User
        fields = [
            'username',
            'email',
            'email2',
            'password'
        ]

        widgets = {
            'username': forms.TextInput(attrs={
                'class': 'form-group',
                'placeholder': 'Enter your username',
            }),
            'email':forms.TextInput(attrs={
                'class': 'form-group',
                'placeholder': 'Enter your email',
            }),
            'email2': forms.TextInput(attrs={
            'class': 'form-group',
            'placeholder': 'Confirm your email',
            }),
            'password': forms.TextInput(attrs={
            'class': 'form-group',
            'placeholder': 'Enter the password',
            }),
        }


    def clean(self, *args, **kwargs):
        email = self.cleaned_data.get('email')
        email2 = self.cleaned_data.get('email2')

        if email != email2:
            raise forms.ValidationError("Emails must match!")

        return super(UserRegisterForm, self).clean(*args, **kwargs)
