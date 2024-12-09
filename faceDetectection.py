#esc to exit
import cv2
alg="/Users/srivarshini/Library/CloudStorage/OneDrive-RathinamGroupOfInstitutions/DeepLearning/haarcascade_frontalface_default.xml"
haar_cascade=cv2.CascadeClassifier(alg) #loading algorithm
cam=cv2.VideoCapture(0) #primary ca=0 , secondary cam=1
# FOR giving video as input
#video_path="path/"
#img_path="path/"
#img=cv2.imread(img_path)
#cam=cv2.VideoCapture(video_path)
while True:
    _,img=cam.read()
    grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces =haar_cascade.detectMultiScale(grayimg,1.3,9) #getting coordinates detectMultiScale(img,scaling factor,no of things to compare)
    for(x,y,w,h) in faces: 
        cv2.rectangle(img,(x,y),(x+w,y+h),(200,25,25),5) #BGR
    cv2.imshow("Face Detection ",img)

    key=cv2.waitKey(10)
    if key==27: #ESC KEY VALUE=27
        break
cam.release()
cv2.destroyAllWindows()
