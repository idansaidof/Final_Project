import cv2
import numpy as np
import os 
import imutils
import smtplib
from gpiozero import AngularServo
from gpiozero import LED
from time import sleep
from imutils.video import VideoStream
import time
from datetime import datetime

#define for led&servo
green_led = LED(15)
red_led = LED(14)
servo = AngularServo(21, initial_angle=0, min_pulse_width=0.0001, max_pulse_width=0.0023) 

#define for mail
SMTP_SERVER = 'smtp.gmail.com' #Email Server (don't change!)
SMTP_PORT = 587 #Server Port (don't change!)
GMAIL_USERNAME = 'project5183907@gmail.com' #change this to match your gmail account
GMAIL_PASSWORD = 'thkihxphvabwvyzh' #change this to match your gmail app-password
sendTo = 'idansaidof24@gmail.com'
emailSubject = "Unknown user"
emailContent = "messege from idan - An unknown user was detected. please check the picture in the unknown folder"

class Emailer:
	def sendmail(self, recipient, subject, content):
		#Create Headers
		headers = ["From: " + GMAIL_USERNAME, "Subject: " + subject, "To: " + recipient,
			"MIME-Version: 1.0", "Content-Type: text/html"]
		headers = "\r\n".join(headers)
	
		#Connect to Gmail Server
		session = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
		session.ehlo()
		session.starttls()
		session.ehlo()

		#Login to Gmail
		session.login(GMAIL_USERNAME, GMAIL_PASSWORD)

		#Send Email & Exit
		session.sendmail(GMAIL_USERNAME, recipient, headers + "\r\n\r\n" + content)
		session.quit

def open_gate():
    
    green_led.on() 
    servo.angle = 90
    time.sleep(1)
    servo.angle = 60
    time.sleep(1)
    servo.angle = 30
    time.sleep(1)
    servo.angle = 0
    green_led.off()

def close_gate():
    
    red_led.on()   
    time.sleep(0.5)
    red_led.off()
    time.sleep(0.5)
    red_led.on()
    time.sleep(0.5)
    red_led.off()

def MarkAttedance(name):
        with open('idan.txt', 'r+') as f:
                myDataList = f.readlines()
                nameList = []
                for line in myDataList:
                        entry = line.split(',')
                        nameList.append(entry[0])
                if name not in nameList:
                        now = datetime.now()
                        dtString = now.strftime(' %H:%M:%S, %d.%m.%Y')
                        f.writelines(f'\n{name},{dtString}')
                else:
                        now = datetime.now()
                        dtString = now.strftime(' %H:%M:%S, %d.%m.%Y')
                        f.writelines(f'\n{name},{dtString}')    


print("      ###############################################################")
print("      ###############################################################")
print("      ###                                                          ##") 
print("      ###  welcome to the project of Idan Saidof & Hanan Lalmayev  ##")
print("      ###                                                          ##") 
print("      ###############################################################") 
print("      ###############################################################") 
print("") 
print("") 



input('\n welcome to our company, to continue press enter:  ')
print("!!!look at the camera please!!!")
id = 0
unknown_counter = 0
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
names = ['none','Idan Saidof','Hanan Lalmayev']
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height
# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
    ret, img =cam.read()
    img = cv2.flip(img, 1) # Flip vertically
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    for(x,y,w,h) in faces:
        input('\n please press enter to start recognize:  ')
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        # If confidence is less them 100 ==> "0" : perfect match 
        
        if (confidence < 70):
            id = names[id]
            print("welcome" , id ) 
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            confidence = "  {0}%".format(round(100 - confidence))
            open_gate()
            MarkAttedance(id)
            cv2.putText(
                    img, 
                    str(id), 
                    (x+5,y-5), 
                    font, 
                    1, 
                    (255,255,255), 
                    2
                   )  
        else:
            id = "Unknown"
            print("sorry, you are not recognize in the systen") 
            confidence = "  {0}%".format(round(100 - confidence))
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)

            close_gate()
            MarkAttedance(id)
            cv2.putText(
                    img, 
                    str(id), 
                    (x+5,y-5), 
                    font, 
                    1, 
                    (255,255,255), 
                    2
                   )  
            unknown_counter += 1
            cv2.imwrite("unknown/unknown." +  str(unknown_counter) + ".jpg", img)
            sender = Emailer()
            sender.sendmail(sendTo, emailSubject, emailContent)
            
        
    
    cv2.imshow('camera',img) 
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()

