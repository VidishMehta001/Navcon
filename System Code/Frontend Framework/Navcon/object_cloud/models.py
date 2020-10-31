from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator 

Choices = ( 
    ("Umbrella", "Umbrella"), 
    ("Shoes", "Shoes"), 
    ("Ball", "Ball"),
    ("Coca Cola", "Coca Cola"),
    ("Macdonalds", "Macdonalds"),
    ("Socks", "Socks"),
    ("Pillow", "Pillow"),
) 

# Create your models here.

class Volunteer(models.Model):
    ServiceRating = models.CharField(max_length=90, choices=Choices)
    