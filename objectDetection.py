import cv2 
import imutils
cam=cv2.VideoCapture(0)
firstFrame=None
area=500

while(True):
    _,img=cam.read()
    text="Normal"
    img=imutils.resize(img,width=1000)
    grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    guassianImg=cv2.GaussianBlur(grayImg,(21,21),0)
    if firstFrame is None:
        firstFrame=guassianImg
        continue
    imgDiff=cv2.absdiff(firstFrame,guassianImg)
    treshImg=cv2.threshold(imgDiff,25,255,cv2.THRESH_BINARY)[1]
    treshImg=cv2.dilate(treshImg,None,iterations=2)
    cnts=cv2.findContours(treshImg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts=imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c)<area:
            continue
        (x,y,w,h)=cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text="Moving Object Detected"
    print(text)
    cv2.putText(img,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv2.imshow("cameraFeed",img)
    key=cv2.waitKey(10)
    print(key)
    if key==ord("q"):
        break
cam.release()
cv2.destroyAllWindows()
