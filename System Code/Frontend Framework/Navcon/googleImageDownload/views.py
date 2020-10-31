from django.shortcuts import render
import argparse
from cv2 import cv2
import os
from google_images_download import google_images_download   #importing the library
# Create your views here.

def getImages(request, *args, **kwargs):
    response = google_images_download.googleimagesdownload()   #class instantiation

    arguments = {"keywords":"Polar bears","limit":40,"print_urls":True,"output_directory": "C:/Users/vidish/Downloads/test/"}   #creating list of arguments
    paths = response.download(arguments)   #passing the arguments to the function
    print(paths)   #printing absolute paths of the downloaded images

    return render(request, 'object_cloud.html', {})