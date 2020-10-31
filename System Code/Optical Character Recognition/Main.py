# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 21:53:05 2020

@author: vidis
"""

import numpy as np
import cv2
import pytesseract

#Please store the Tesseract-OCR\\tesseract in your program file folder
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe' 

#Please upload a video for OCR Recognition. 
cap = cv2.VideoCapture("C://Users//vidis//Videos//Captures//Menu.mp4")

while True:
    ret, screen = cap.read()
    hImg, wImg = 1000,750
    boxes = pytesseract.image_to_data(screen)
    for index, b in enumerate(boxes.splitlines()):
        if index!= 0:
            b = b.split()
            if len(b) == 12:
                x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
                cv2.rectangle(screen,(x,y),(w+x,h+y),(0,0,255),3)
                #cv2.putText(screen,b[11],(x,hImg-y+25),(w,hImg-h),cv2.FONT_HERSHEY_COMPLEX,(50,50,255),2)
               
              
           
    cv2.imshow('object detection', cv2.resize(screen, (800,600)))
    if cv2.waitKey(25) & 0xFF == ord('q'):
      cv2.destroyAllWindows()
      break