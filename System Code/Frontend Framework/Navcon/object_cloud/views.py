from django.shortcuts import render
from .forms import VolunteerForm
import argparse
from cv2 import cv2
import os
from google_images_download import google_images_download   #importing the library
from Navcon.settings import BASE_DIR

# Create your views here.



# Create your views here.

def ObjectCloudView(request, *args, **kwargs):
    if request.method == "GET":
        form = VolunteerForm()
    else:
        form  = VolunteerForm(request.POST)
        if form.is_valid():
            response = google_images_download.googleimagesdownload()   #class instantiation
            path = os.path.join(BASE_DIR,'svm/ImageData')
            print(path)
            arguments = {"keywords":"Polar bears","limit":40,"print_urls":True,"output_directory": path}   #creating list of arguments
            paths = response.download(arguments)   #passing the arguments to the function
            print(paths)   #printing absolute paths of the downloaded images
            return render (request, "object_cloud.html", {})

        else:
            form = VolunteerForm()
    
    return render(request, 'object_cloud.html', {'form': form})