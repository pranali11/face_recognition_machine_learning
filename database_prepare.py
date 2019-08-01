import cv2
import numpy as np
#intilizing the counter for naming the samples 
i=0
#Intilizing the Cam 
cap = cv2.VideoCapture(0)
def Capture():
    
    ret, frame  = cap.read()
    cv2.imshow('frame', frame)
    return(frame)

#def save():
    
    
while True:
    frame= Capture()
    #press s for saving image and it will get automatically updated in the s1 foler
    if cv2.waitKey(1) & 0xff == ord('s'):
        i = i+1
        print(i)
        #add path here
        buf = "training-data/d1/%d.bmp" % i
        cv2.imwrite(buf,frame)
        #press q to quit the cam 
    elif cv2.waitKey(1) & 0xff == ord('q'):
        break
    
cap.release()
cv2. destroyAllWindows()
