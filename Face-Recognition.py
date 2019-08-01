
import cv2
import os
import numpy as np


#there is no label 0 in our training data so subject name for index/label 0 is empty
#add tags here
#example ["", "add_name"]
subjects = ["",""]

#function to detect face using haar detection
def detect_face(img):
    
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
   
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
 
    if (len(faces) == 0):
        return None, None
    

    (x, y, w, h) = faces[0]
    
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]
    
#trial face detection using skin tone comment the above if in case one wants to test this
"""
    Hsv= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    Ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    gray= cv2.cvtColor (img,cv2.COLOR_BGR2GRAY)
    th2= cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    lower_red = np.array([0, 50,50])
    upper_red = np.array([30,255,255])
    mask = cv2.inRange(Hsv,lower_red, upper_red)
    mask= cv2.medianBlur(mask,5)
    mask= cv2.medianBlur(mask,5)
    res = cv2.bitwise_and(th2, th2,mask=mask)
    
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(res,kernel,iterations = 1)
    dilation = cv2.dilate(erosion,kernel,iterations = 1)
    dilation_1 = cv2.dilate(dilation,kernel,iterations = 1)
    dilation_2 = cv2.dilate(dilation_1,kernel,iterations = 1)
    closing = cv2.morphologyEx(dilation_2, cv2.MORPH_CLOSE, kernel)
    closing_1 = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
    dilation_3 = cv2.dilate(closing_1,kernel,iterations = 1)
    dilation_4 = cv2.dilate(dilation_3,kernel,iterations = 1)
    dilation_5 = cv2.dilate(dilation_4,kernel,iterations = 1)
    dilation_6 = cv2.dilate(dilation_5,kernel,iterations = 1)
    dilation_7 = cv2.dilate(dilation_6,kernel,iterations = 1)
    dilation_8 = cv2.dilate(dilation_7,kernel,iterations = 1)
    closing_2 = cv2.morphologyEx(dilation_8, cv2.MORPH_CLOSE, kernel)
    erosion = cv2.erode(closing_2,kernel,iterations = 1)
    erosion = cv2.erode(erosion,kernel,iterations = 1)
    erosion = cv2.erode(erosion,kernel,iterations = 1)
    erosion = cv2.erode(erosion,kernel,iterations = 1)
    erosion = cv2.erode(erosion,kernel,iterations = 1)
    erosion = cv2.erode(erosion,kernel,iterations = 1)
    erosion = cv2.erode(erosion,kernel,iterations = 1)
    erosion = cv2.erode(erosion,kernel,iterations = 1)
    erosion = cv2.erode(erosion,kernel,iterations = 1)
    erosion = cv2.erode(erosion,kernel,iterations = 1)
    
    im2,contours,hierarchy = cv2.findContours(erosion ,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    lent = len(contours)

    
    largest = cv2.contourArea(contours[0])
    for i in range(lent):
        if (largest <= cv2.contourArea(contours[i])):
            largest = cv2.contourArea(contours[i])
            x,y,w,h = cv2.boundingRect(contours[i])
            ratio= float(w)/h
            if(ratio <= (1.0) ):
                if(ratio >= 0.4):
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.imshow('image',img[y:y+w,x:x+h])
                    return gray[y:y+w, x:x+h], (x,y,w,h)
                    #cv2.imshow('image',img)

    if (largest == cv2.contourArea(contours[0])):
        
        x,y,w,h = cv2.boundingRect(contours[0])
        ratio= float(w)/h
        if(ratio <= (1.0) ):
            if(ratio >= 0.4):
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.imshow('image',img[y:y+w,x:x+h])
                return gray[y:y+w, x:x+h], (x,y,w,h)
                #cv2.imshow('image', img)
"""



def prepare_training_data(data_folder_path):
    
    #------STEP-1--------
    #get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
    
    #list to hold all subject faces
    faces = []
    #list to hold labels for all subjects
    labels = []
    
    #let's go through each directory and read images within it
    for dir_name in dirs:
        
        #our subject directories start with letter 's' so
        #ignore any non-relevant directories if any
        if not dir_name.startswith("d"):
            continue;
            
        #------STEP-2--------
        #extract label number of subject from dir_name
        #format of dir name = slabel
        #, so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("d", ""))
        
        #build path of directory containin images for current subject subject
        #sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name
        
        #get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)
        
        #------STEP-3--------
        #go through each image name, read image, 
        #detect face and add face to list of faces
        for image_name in subject_images_names:
            
            #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;
            
            #build image path
            #sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            #read image
            
            image = cv2.imread(image_path)
            image_size= cv2.resize(image,(500,500))
            #display an image window to show the image 
            #cv2.imshow("Training on image...", cv2.resize(image, (400, 300)))
            #cv2.waitKey(100)
            
            #detect face
            face, rect = detect_face(image)
            
            #------STEP-4--------
            #for the purpose of this tutorial
            #we will ignore faces that are not detected
            if face is not None:
                #add face to list of faces
                faces.append(face)
                #add label for this face
                labels.append(label)
            print(label)
            print (face)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels



print("Preparing data...")

faces, labels = prepare_training_data('training-data')
print("Data prepared")

#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))


#create our LBPH face recognizer 
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#or use EigenFaceRecognizer by replacing above line with 
#face_recognizer = cv2.face.EigenFaceRecognizer_create()

#or use FisherFaceRecognizer by replacing above line with 
#   face_recognizer = cv2.face.FisherFaceRecognizer_create()


#train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))


#function to draw rectangle on image according to given (x, y) coordinates and given width and heigh
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
#function to draw text on give image starting frompassed (x, y) coordinates. 
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 11), 2)


#this function recognizes the person in image passed
#and draws a rectangle around detected face with name of the 
#subject
def predict(test_img):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img)

    #predict the image using our face recognizer 
    label, confidence = face_recognizer.predict(face)
    print(label)
    print(confidence)
    #get name of respective label returned by face recognizer
    label_text = subjects[label]
    
    #draw a rectangle around face detected
    draw_rectangle(img, rect)
    #draw name of predicted person
    draw_text(img, label_text, rect[0], rect[1]-5)
    
    return img



print("Predicting images...")
#cap = cv2.VideoCapture(0)
#while True:
    #ret,img = cap.read()
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#load test images - add path here
test_img1 = cv2.imread("")
#test_img3 = cv2.imread("test-data/test6.jpg")
#test_img4 = cv2.imread("test-data/test5.jpg")
#perform a prediction
#predicted_img1 = predict(test_img1)

predicted_img = predict(test_img1)

print("Prediction complete")
cv2.imshow('test', predicted_img)

cv2.waitKey(0)
cv2.destroyAllWindows()







