from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import numpy as np
import cv2
import time
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#=========================SNAPSHOT\\CV2_VIDEO====================================================
def nw(event):
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    cv2.namedWindow('img',cv2.WINDOW_NORMAL)

    while 1:
        ret, img = cap.read()
        ret, img1 = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            roi_color1 = img1[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                cv2.rectangle(roi_color1, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        cv2.imshow('img', img)
        k = cv2.waitKey(1) & 0xff

        if k == 27:
            break
        end_time = time.time()
        elapsed = end_time - start_time
        if int(elapsed) > 10:
            cv2.imshow('tan', img1)
            cv2.destroyWindow("img")
            cv2.waitKey(0)
            break
    cap.release()
    cv2.destroyAllWindows()
#=====================VIEW===================================================
def browseFile(event):
    filename = askopenfilename(filetypes=(("Jpeg images","*.jpg"),("PNG images","*.png"),("All files","*.*")))
    if filename!="i":
        root1=Toplevel()
        lo = Image.open(filename)
        ren = ImageTk.PhotoImage(lo)
        width=ren.width()
        height=ren.height()
        resolution=str(width)+'x'+str(height)
        root1.geometry(resolution)
        img12 = Label(root1, image=ren)
        img12.image = ren
        img12.place(x=0, y=0)
        root1.mainloop()
#=====================MAIN===================================================
root = Tk()
root.title("POSE COPY")
root.geometry('1600x1000')
root.wm_iconbitmap("icon.ico")

load = Image.open("final.png")
render = ImageTk.PhotoImage(load)

# labels can be text or images
img = Label(root, image=render)
img.image = render
img.place(x=0, y=0)

new = Button(root, text="SNAPSHOT", activebackground="green", fg="cyan", bg="indian red", width=15, height=3,relief="ridge",borderwidth=7)
new.bind("<Button-1>",nw)
new.place(x=680, y=450)

view = Button(root, text="VIEW", fg="cyan", bg="indian red", width=15, height=3,relief="ridge",borderwidth=7)
view.place(x=680, y=530)
view.bind("<Button-1>",browseFile)

root.mainloop()
#====================FINISH=======================================================
