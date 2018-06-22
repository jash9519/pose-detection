from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import numpy as np
import cv2
import time
MODE = "MPI"
m=1
if MODE is "COCO":
    protoFile = "pose_deploy_linevec.prototxt"
    weightsFile = "pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

elif MODE is "MPI" :
    protoFile = "pose_deploy_linevec_faster_4_stages.prototxt.txt"
    weightsFile = "pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

#=========================================================new======================================================================
def nw(event):
    global m
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    while 1:
        ret, frame = cap.read()
        cv2.imshow('img', frame)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
        end_time = time.time()
        elapsed = end_time - start_time
        if int(elapsed) > 10:
            cv2.imshow('tan', frame)
            cv2.destroyWindow("img")
            frameCopy = np.copy(frame)
            frameWidth = frame.shape[1]
            t = time.time()
            frameHeight = frame.shape[0]
            threshold = 0.1

            net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

            # input image dimensions for the network
            inWidth = 368
            inHeight = 368
            inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

            net.setInput(inpBlob)

            output = net.forward()
            print("time taken by network : {:.3f}".format(time.time() - t))

            H = output.shape[2]
            W = output.shape[3]

            # Empty list to store the detected keypoints
            points = []

            for i in range(nPoints):
                # confidence map of corresponding body's part.
                probMap = output[0, i, :, :]

                # Find global maxima of the probMap.
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

                # Scale the point to fit on the original image
                x = (frameWidth * point[0]) / W
                y = (frameHeight * point[1]) / H

                if prob > threshold:
                  cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                  cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2,lineType=cv2.LINE_AA)

                    # Add the point to the list if the probability is greater than the threshold
                  points.append((int(x), int(y)))
                else:
                    points.append(None)

            # Draw Skeleton
            for pair in POSE_PAIRS:
                partA = pair[0]
                partB = pair[1]

                if points[partA] and points[partB]:
                    cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
                    cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)


            # cv2.imshow('Output-Keypoints', frameCopy)
            cv2.imshow('Output-Skeleton', frame)
            cv2.waitKey(0)
            break
    cap.release()
    cv2.destroyAllWindows()
#====================VIEW==========================================
def browseFile(event):
    filename = askopenfilename(filetypes=(("Jpeg images","*.jpg"),("PNG images","*.png"),("All Files","*.*")))
    print(filename)
    root1=Toplevel()
    root1.wm_iconbitmap("meet.ico")
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

#================================================================
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


new = Button(root, text="SNAPSHOT", activebackground="green", fg="white", bg="blue", width=15, height=3,relief='ridge',borderwidth=7)
new.bind("<Button-1>",nw)
new.place(x=670, y=430)

view = Button(root, text="view", fg="white", bg="blue", width=15, height=3,relief='ridge',borderwidth=7)
view.place(x=670, y=500)
view.bind("<Button-1>",browseFile)

root.mainloop()