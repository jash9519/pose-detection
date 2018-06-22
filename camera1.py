import numpy as np
import cv2
import time

global y,h,x,w,no,img

def blend_transparent(img,img1):
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


img1=cv2.imread('hulk.png',-1)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
fullbody_cascade = cv2.CascadeClassifier('haarcascade_fullbody old.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap=cv2.VideoCapture(0)
start_time=time.time()
while 1:
    ret, im = cap.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = fullbody_cascade.detectMultiScale(gray, 1.1, 1)

    for (x, y, w, h) in faces:
        #  x=x-30
        #   y=y-25
        # h=h+60
        # w=w+55
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        img = im[y:y + h, x:x + w]
        no = blend_transparent(img, img1)
    cv2.imshow('img', im)
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break
    end_time = time.time()
    elapsed = end_time - start_time
    if int(elapsed) > 10:

        im[y:y + h, x:x + w] = no
        cv2.imshow('fuck',im)
        cv2.destroyWindow("img")
        break
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()