import cv2
import subprocess 
import os
from gpiozero import Button
import RPi.GPIO as GPIO
import sys
from time import sleep

#thres = 0.45 # Threshold to detect object


#If code is stopped while the solenoid is active it stays active
#This may produce a warning if the code is restarted and it finds the GPIO Pin, which it defines as non-active in next line, is still active
#from previous time the code was run. This line prevents that warning syntax popping up which if it did would stop the code running.
GPIO.setwarnings(False)
#This means we will refer to the GPIO pins
#by the number directly after the word GPIO. A good Pin Out Resource can be found here https://pinout.xyz/
GPIO.setmode(GPIO.BCM)
#This sets up the GPIO 18 pin as an output pin
GPIO.setup(27, GPIO.OUT)
GPIO.setup(18, GPIO.OUT)
GPIO.setup(26, GPIO.OUT)

button = Button(16)


classNames = []
classFile = "/home/sampa/Desktop/Object_Detection_Files/coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/sampa/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/sampa/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box,className])
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    


    return img,objectInfo

def turn_on_safety(program, exit_code=0):
    # Start safety script
    subprocess.Popen(program)
    # close this script
    sys.exit(exit_code)
    
def sword_time():
#This Turns Relay Off. Brings Voltage to Max GPIO can output ~3.3V
    GPIO.output(27, 1)
    GPIO.output(26, 1)
#Wait 1 Seconds
    sleep(1)
#Turns Relay On. Brings Voltage to Min GPIO can output ~0V.
    GPIO.output(27, 0)
    GPIO.output(26, 0)
#Wait 1 Seconds
    sleep(2)
#This Turns Relay Off. Brings Voltage to Max GPIO can output ~3.3V
    GPIO.output(18, 1)
    GPIO.output(26, 1)
#Wait 1 Seconds
    sleep(1)
#Turns Relay On. Brings Voltage to Min GPIO can output ~0V.
    GPIO.output(18, 0)
    GPIO.output(26, 0)
    sleep(2)

    
    turn_on_safety(['python', 'safeoff.py'])


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    #cap.set(10,70)


    while True:
        if button.is_pressed:
            turn_on_safety(['python', 'safeoff.py'])

        else:
            success, img = cap.read()
            result, objectInfo = getObjects(img,0.65,0.2,objects=['apple','banana','orange','carrot','pizza','cake','sandwich','broccoli','hot dog','donut','person'])
            #print(objectInfo)
            cv2.imshow("Output",img)
            cv2.waitKey(1)
        
        for obj in objectInfo:
            foundClass = obj[1]   ##loop through objects identified in picture and speak  
            
            if foundClass == ('person'):
                turn_on_safety(['python', 'safeoff.py'])
            else:
                sword_time()
    
    

