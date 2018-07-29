import numpy as np
import cv2
import time

def blend_transparent(img,img1):
    h,w,c=img.shape
    height,width,channel=img1.shape
    img1 = cv2.resize(img1, None, fx=w /width, fy=h /height, interpolation=cv2.INTER_AREA)
    overlayimage=img1[:,:,:3]
    overlaymask=img1[:,:,3:]
    backgroundmask=255-overlaymask
    overlaymask=cv2.cvtColor(overlaymask,cv2.COLOR_GRAY2BGR)
    backgroundmask = cv2.cvtColor(backgroundmask, cv2.COLOR_GRAY2BGR)
    facepart=(img*(1/255.0))*(backgroundmask*(1/255.0))
    overlaypart = (overlayimage * (1 / 255.0)) * (overlaymask * (1 / 255.0))
    return np.uint8(cv2.addWeighted(facepart,255.0,overlaypart,255.0,0))


img2=cv2.imread('chasma.png',-1)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_righteye.xml')
cap=cv2.VideoCapture(0)
while 1:
    ret, img = cap.read()
    h1,w1=img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 2)
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break
    for (x, y, w, h) in faces:
        #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if (h1-y-h)<0.25*h1:
            f=0.2*(h1-h-y)
        else:
            f=0.16*(y+h)
        if (w1-x-w)<0.25*w1:
            g=0.15*(w1-x-w)
        else:
            g=0.115*(x+w)
        if x>w1/2:
            k=0.145*(w1-x)
        else:
            k=0.145*x
        if y>h1/2:
           l=0.2*(h1-y)
        else:
            l=0.2*y
        #wow=img[y-int(l):y+h+int(f),x-int(k):x+h+int(g)]
        wow=img[y:y+h,x:x+w]
        wow1=blend_transparent(wow,img2)
        #img[y-int(l):y+h+int(f),x-int(k):x+h+int(g)]=wow1
        img[y:y + h, x:x + w]=wow1
    cv2.imshow('new',img)

cap.release()
cv2.destroyAllWindows()