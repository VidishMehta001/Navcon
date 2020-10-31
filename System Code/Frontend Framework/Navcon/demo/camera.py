import os
import numpy as np
from django.conf import settings
from cv2 import cv2
from Navcon.settings import Detected

#face_detection_videocam = cv2.CascadeClassifier(os.path.join(settings.BASE_DIR, 'cv2/data/haarcascade_frontalface_default.xml'))

class VideoCamera(object):
    def __init__(self, faceDetect, cam, detection_counter, rec, getId, userId, font):
        self.faceDetect = faceDetect
        self.video = cam
        self.detection_counter = detection_counter
        self.rec = rec
        self.getId = getId
        self.userId = userId
        self.font = font

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, img = self.video.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = self.faceDetect.detectMultiScale(gray, 1.3, 5)
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0), 2)

            getId,conf = self.rec.predict(gray[y:y+h, x:x+w]) #This will predict the id of the face

            self.detection_counter += 1

            #print conf;
            if conf>35:
                userId = self.getId
                cv2.putText(img, "Detected",(x,y+h), self.font, 2, (0,255,0),2)
                Detected = True
            else:
                cv2.putText(img, "Unknown",(x,y+h), self.font, 2, (0,0,255),2)

            # Printing that number below the face
            # @Prams cam image, id, location,font style, color, stroke
        
        if self.detection_counter < 40:
            #frame_flip = cv2.flip(img, 1)
            ret, jpeg = cv2.imencode('.jpg', img)
            return jpeg.tobytes()
        else:    
            if(userId != 0):
                Detected = True
                print(userId)
                cv2.waitKey(1000)
                self.video.release()
                cv2.destroyAllWindows()
            else:
                Detected = False
            
                return Detected







