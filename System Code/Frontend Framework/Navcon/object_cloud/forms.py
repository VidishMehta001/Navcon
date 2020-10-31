from django import forms
from .models import Volunteer


Choices = ( 
    ("Umbrella", "Umbrella"), 
    ("Shoes", "Shoes"), 
    ("Ball", "Ball"),
    ("Coca Cola", "Coca Cola"),
    ("Macdonalds", "Macdonalds"),
    ("Socks", "Socks"),
    ("Pillow", "Pillow"),
) 


class VolunteerForm(forms.Form):
    Service_Rating = forms.ChoiceField(label='Which object you would like to label next?', choices=Choices)