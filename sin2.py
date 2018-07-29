from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import time
import math
import variable
import urllib.request
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++===
#============================================== blend transparent fuction ==========================================================================
def blend_transparent(img, img1):

    height, width= img1.shape[:2]
    h, w, channels = img.shape
    img1 = cv2.resize(img1, None, fx= (w / width), fy= (h / height),interpolation=cv2.INTER_AREA)

    # Split out the transparency mask from the colour info
    overlay_img = img1[:, :, :3]  # Grab the BRG planes
    overlay_mask = img1[:, :, 3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


#=========================================================== modes of display  =============================================================


# THE MODE OF PROCESSING
MODE = "MPI"


if MODE is "COCO":

    protoFile = "pose_deploy_linevec.prototxt.txt"
    weightsFile = "pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]


elif MODE is "MPI" :

    protoFile = "pose_deploy_linevec_faster_4_stages.prototxt.txt"
    weightsFile = "pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

def rotateAndScale(img, degreesCCW, scaleFactor=1):

        (oldY, oldX) = img.shape[:2]  # note: numpy uses (y,x) convention but most OpenCV functions use (x,y)

        M = cv2.getRotationMatrix2D(center=(oldX / 2, oldY / 2), angle=degreesCCW,
                                    scale=scaleFactor)  # rotate about center of image.

        # choose a new image size.
        newX, newY = oldX * scaleFactor, oldY * scaleFactor

        # include this if you want to prevent corners being cut off
        r = np.deg2rad(degreesCCW)
        newX, newY = (abs(np.sin(r) * newY) + abs(np.cos(r) * newX), abs(np.sin(r) * newX) + abs(np.cos(r) * newY))

        # the warpAffine function call, below, basically works like this:

        # 1. apply the M transformation on each pixel of the original image
        # 2. save everything that falls within the upper-left "dsize" portion of the resulting image.

        # So I will find the translation that moves the result to the center of that region.
        (tx, ty) = ((newX - oldX) / 2, (newY - oldY) / 2)
        M[0, 2] += tx  # third column of matrix holds translation, which takes effect after rotation.
        M[1, 2] += ty

        rotatedImg = cv2.warpAffine(img, M, dsize=(int(newX), int(newY)))
        return rotatedImg


def nailedit(frame):
    #frame = cv2.imread("man8.jpg")
    h1, w1 = frame.shape[:2]
    # frame=cv2.resize(frame,None,fx=1.2,fy=1,interpolation=cv2.INTER_AREA)

    # THE SELECTION IMAGES
    img33 = variable.img33
    img44 = variable.img44
    img56 = variable.img56
    img67 = variable.img67
    img78 = variable.img78
    img89 = variable.img89
    img90 = variable.img90
    img01 = variable.img01
    img1  = variable.img1

    def pointer(p1, p2, p3, p4, part, q):
        if p1 > p2:
            x1 = p2
            x2 = p1
        else:
            x2 = p2
            x1 = p1
        if p3 > p4:
            y1 = p4
            y2 = p3
        else:
            y1 = p3
            y2 = p4
        d = 0
        if x1 > w1 / 2:
            k = 0.05 * (w1 - x1)
        else:
            k = 0.05 * x1
        if y1 > h1 / 2:
            l = 0.06 * (h1 - y1)
        else:
            l = 0.06 * y
        if (h1 - y2) < 0.5 * h1:
            f = 0.14 * (h1 - y2)
        else:
            f = 0.14 * y2
        if (w1 - x2) < 0.5 * w1:
            g = 0.1 * (w1 - x2)
        else:
            g = 0.1 * x2
        if x1 - x2 != 0:
            slope = (p3 - p4) / (p1 - p2)
            s = math.atan(slope)
            d = math.degrees(s)
        if d > 0:
            d = 90 - d
        if d < 0:
            d = -(90 + d)
        if k == 0:
            if p1 < p2 | p3 > p4:
                d = d + 180
        else:
            if p1 > p2 | p3 > p4:
                d = d + 180
        roi = frame[y1 - int(l):y2 + int(f), x1 - int(k):int(g) + x2]
        ret = rotateAndScale(part, d)
        ret2 = blend_transparent(roi, ret)

        height, width = ret2.shape[:2]
        h = y2 + int(f) - (y1 - int(l))
        w = x2 + int(g) - (x1 - int(k))

        final1 = cv2.resize(ret2, None, fx=(w / width), fy=(h / height), interpolation=cv2.INTER_AREA)
        frame[y1 - int(l):y2 + int(f), x1 - int(k):int(g) + x2] = final1

    ############################## image processing begins %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    threshold = 0.1

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    t = time.time()

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

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ initializing the for loop of contours $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

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
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))

        else:
            points.append(None)

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$  torsoe  &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #cv2.rectangle(frame, (x-int(0.75*w), y+h), (x +2*w, y + 4*h), (255, 0, 0), 2)  ########
        img13= frame[y+h:y+4*h, x-int(w*0.75):x+2*w]
        result_3 = blend_transparent(img13, img1)
        frame[y+h:y+4*h, x-int(w*0.75):x+2*w]=result_3

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ left arm &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    pointer(points[2][0], points[3][0], points[2][1], points[3][1], img33, 0)

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ left hand $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$44

    pointer(points[3][0], points[4][0], points[3][1], points[4][1], img44, 0)

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ right arm $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    pointer(points[5][0], points[6][0], points[5][1], points[6][1], img56, 1)

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ right hand $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    pointer(points[6][0], points[7][0], points[6][1], points[7][1], img67, 1)

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ left knee $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    pointer(points[8][0], points[9][0], points[8][1], points[9][1], img78, 0)

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ left leg $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    pointer(points[9][0], points[10][0], points[9][1], points[10][1], img89, 0)

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ right knee $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    pointer(points[11][0], points[12][0], points[11][1], points[12][1], img90, 1)

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ right leg $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    pointer(points[12][0], points[13][0], points[12][1], points[13][1], img01, 1)

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ final image $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    global root3
    cv2.imwrite("frame.jpg", frame)
    root3 = Toplevel()
    root3.title("TRANSFORMATION.......")
    root3.wm_iconbitmap("meet.ico")
    load = Image.open("frame.jpg")
    render = ImageTk.PhotoImage(load)
    width = render.width()
    height = render.height()
    resolution = str(width) + 'x' + str(height)
    root3.geometry(resolution)

    # labels can be text or images
    img = Label(root3, image=render)
    img.image = render
    img.place(x=0, y=0)
    ################################# THE END OF THE IMAGE PROCESSING PROGRAMME ##################################################



#=========================================================new======================================================================

def nw(event):
    if event.widget['text']=='Captian America':

        variable.img33 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\captianamerica\leftarm.png", -1)
        variable.img44 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\captianamerica\lefthand.png", -1)
        variable.img56 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\captianamerica\rightarm.png", -1)
        variable.img67 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\captianamerica\righthand.png", -1)
        variable.img78 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\captianamerica\leftknee.png", -1)
        variable.img89 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\captianamerica\leftleg.png", -1)
        variable.img90 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\captianamerica\rightknee.png", -1)
        variable.img01 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\captianamerica\rightleg.png", -1)
        variable.img1 =  cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\captianamerica\torsoe.png", -1)
        root2.destroy()

    elif  event.widget['text']=='Spiderman':

        variable.img33 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\ironspider\leftarm.png", -1)
        variable.img44 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\ironspider\lefthand.png", -1)
        variable.img56 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\ironspider\rightarm.png", -1)
        variable.img67 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\ironspider\righthand.png", -1)
        variable.img78 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\ironspider\leftknee.png", -1)
        variable.img89 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\ironspider\leftleg.png", -1)
        variable.img90 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\ironspider\rightknee.png", -1)
        variable.img01 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\ironspider\rightleg.png", -1)
        variable.img1 =  cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\ironspider\torsoe.png", -1)
        root2.destroy()

    elif  event.widget['text']=='Hulk':

         variable.img33 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\hulk\leftarm.png", -1)
         variable.img44 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\hulk\lefthand.png", -1)
         variable.img56 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\hulk\rightarm.png", -1)
         variable.img67 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\hulk\righthand.png", -1)
         variable.img78 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\hulk\leftknee.png", -1)
         variable.img89 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\hulk\leftleg.png", -1)
         variable.img90 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\hulk\rightknee.png", -1)
         variable.img01 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\hulk\rightleg.png", -1)
         variable.img1 =  cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\hulk\torsoe.png", -1)
         root2.destroy()

    elif  event.widget['text']=='Batman':

         variable.img33 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\batman\leftarm.png", -1)
         variable.img44 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\batman\lefthand.png", -1)
         variable.img56 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\batman\rightarm.png", -1)
         variable.img67 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\batman\righthand.png", -1)
         variable.img78 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\batman\leftknee.png", -1)
         variable.img89 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\batman\leftleg.png", -1)
         variable.img90 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\batman\rightknee.png", -1)
         variable.img01 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\batman\rightleg.png", -1)
         variable.img1 =  cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\batman\torsoe.png", -1)
         root2.destroy()

    elif event.widget['text'] == 'Superman':

        variable.img33 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\superman\leftarm.png", -1)
        variable.img44 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\superman\lefthand.png", -1)
        variable.img56 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\superman\rightarm.png", -1)
        variable.img67 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\superman\righthand.png", -1)
        variable.img78 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\superman\leftknee.png", -1)
        variable.img89 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\superman\leftleg.png", -1)
        variable.img90 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\superman\rightknee.png", -1)
        variable.img01 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\superman\rightleg.png", -1)
        variable.img1  = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\superman\torsoe.png", -1)
        root2.destroy()

    elif event.widget['text'] == 'Thanos':

        variable.img33 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\thanos\leftarm.png", -1)
        variable.img44 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\thanos\lefthand.png", -1)
        variable.img56 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\thanos\rightarm.png", -1)
        variable.img67 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\thanos\righthand.png", -1)
        variable.img78 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\thanos\leftknee.png", -1)
        variable.img89 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\thanos\leftleg.png", -1)
        variable.img90 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\thanos\rightknee.png", -1)
        variable.img01 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\thanos\rightleg.png", -1)
        variable.img1 =  cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\thanos\torsoe.png", -1)
        root2.destroy()

    elif event.widget['text'] == 'Goku':

        variable.img33 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\goku\leftarm.png", -1)
        variable.img44 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\goku\lefthand.png", -1)
        variable.img56 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\goku\rightarm.png", -1)
        variable.img67 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\goku\righthand.png", -1)
        variable.img78 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\goku\leftknee.png", -1)
        variable.img89 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\goku\leftleg.png", -1)
        variable.img90 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\goku\rightknee.png", -1)
        variable.img01 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\goku\rightleg.png", -1)
        variable.img1 =  cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\goku\torsoe.png", -1)
        root2.destroy()

    elif event.widget['text'] == 'Vegeta':

        variable.img33 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\vegeta\leftarm.png", -1)
        variable.img44 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\vegeta\lefthand.png", -1)
        variable.img56 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\vegeta\rightarm.png", -1)
        variable.img67 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\vegeta\righthand.png", -1)
        variable.img78 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\vegeta\leftknee.png", -1)
        variable.img89 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\vegeta\leftleg.png", -1)
        variable.img90 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\vegeta\rightknee.png", -1)
        variable.img01 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\vegeta\rightleg.png", -1)
        variable.img1 =  cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\vegeta\torsoe.png", -1)
        root2.destroy()


    elif event.widget['text'] == 'Black Widow':

        variable.img33 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\blackwidow\leftarm.png", -1)
        variable.img44 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\blackwidow\lefthand.png", -1)
        variable.img56 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\blackwidow\rightarm.png", -1)
        variable.img67 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\blackwidow\righthand.png", -1)
        variable.img78 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\blackwidow\leftknee.png", -1)
        variable.img89 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\blackwidow\leftleg.png", -1)
        variable.img90 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\blackwidow\rightknee.png", -1)
        variable.img01 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\blackwidow\rightleg.png", -1)
        variable.img1 =  cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\blackwidow\torsoe.png", -1)
        root2.destroy()


    elif event.widget['text'] == 'Captain Marvel':

        variable.img33 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\captainmarvel\leftarm.png", -1)
        variable.img44 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\captainmarvel\lefthand.png", -1)
        variable.img56 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\captainmarvel\rightarm.png", -1)
        variable.img67 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\captainmarvel\righthand.png", -1)
        variable.img78 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\captainmarvel\leftknee.png", -1)
        variable.img89 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\captainmarvel\leftleg.png", -1)
        variable.img90 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\captainmarvel\rightknee.png", -1)
        variable.img01 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\captainmarvel\rightleg.png", -1)
        variable.img1 =  cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\captainmarvel\torsoe.png", -1)
        root2.destroy()


    elif event.widget['text'] == 'Kratos':

        variable.img33 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\kratos\leftarm.png", -1)
        variable.img44 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\kratos\lefthand.png", -1)
        variable.img56 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\kratos\rightarm.png", -1)
        variable.img67 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\kratos\righthand.png", -1)
        variable.img78 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\kratos\leftknee.png", -1)
        variable.img89 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\kratos\leftleg.png", -1)
        variable.img90 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\kratos\rightknee.png", -1)
        variable.img01 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\kratos\rightleg.png", -1)
        variable.img1 =  cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\kratos\torsoe.png", -1)
        root2.destroy()


    elif event.widget['text'] == 'Ironman':

        variable.img33 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\ironman\leftarm.png", -1)
        variable.img44 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\ironman\lefthand.png", -1)
        variable.img56 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\ironman\rightarm.png", -1)
        variable.img67 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\ironman\righthand.png", -1)
        variable.img78 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\ironman\leftknee.png", -1)
        variable.img89 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\ironman\leftleg.png", -1)
        variable.img90 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\ironman\rightknee.png", -1)
        variable.img01 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\ironman\rightleg.png", -1)
        variable.img1 =  cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\ironman\torsoe.png", -1)
        root2.destroy()


    elif event.widget['text'] == 'Thor':

        variable.img33 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\thor\leftarm.png", -1)
        variable.img44 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\thor\lefthand.png", -1)
        variable.img56 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\thor\rightarm.png", -1)
        variable.img67 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\thor\righthand.png", -1)
        variable.img78 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\thor\leftknee.png", -1)
        variable.img89 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\thor\leftleg.png", -1)
        variable.img90 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\thor\rightknee.png", -1)
        variable.img01 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\thor\rightleg.png", -1)
        variable.img1 =  cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\thor\torsoe.png", -1)

        root2.destroy()

    url='http://10.172.85.96:8080/shot.jpg'
    #cap = cv2.VideoCapture(0)
    start_time = time.time()
    ##cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    #while 1:
    #    ret, frame = cap.read()
    #    cv2.imshow('img', frame)
    #    k = cv2.waitKey(10) & 0xff
    #    if k == 27:
    #        break
    #    end_time = time.time()
    #    elapsed = end_time - start_time
    #    if int(elapsed) > 10:
    #        cv2.destroyWindow("img")
    #        nailedit(frame)
    #        break
    #cap.release()
    while True:
        # Use urllib to get the image and convert into a cv2 usable format
        imgResp = urllib.request.urlopen(url)
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
        img = cv2.imdecode(imgNp, -1)

        # put the image on screen
        cv2.imshow('IPWebcam', img)

        # To give the processor some less stress
        # time.sleep(0.1)

        k=cv2.waitKey(1) & 0xFF
        if k==27:
            break
        end_time = time.time()
        elapsed = end_time - start_time
        if int(elapsed) > 10:
            cv2.destroyWindow("IPWebcam")
            #height, width = img[:2]
            #img = cv2.resize(img, None, fx=1600 / width, fy=1000 / height, interpolation=cv2.INTER_AREA)
            #img=rotateAndScale(img,-90)
            nailedit(img)
            break

    root3.mainloop()
    cv2.destroyAllWindows()

#====================VIEW==========================================
def browseFile(event):

    filename = askopenfilename(filetypes=(("Jpeg images","*.jpg"),("PNG images","*.png"),("All Files","*.*")))
    print(filename)
    root1=Toplevel()
    root1.wm_iconbitmap("icon.ico")
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

def lol33(event):
    img2 = cv2.imread('chasma.png', -1)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    while 1:
        ret, img = cap.read()
        h1, w1 = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 2)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
        if k==ord('s'):
            cv2.imshow("snap",img1)
            cv2.imwrite("snapshot.jpg",img1)
            cv2.destroyWindow('PRESS "s" TO CAPTURE')
            cv2.waitKey(0)
            break

        for (x, y, w, h) in faces:
            # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if (h1 - y - h) < 0.25 * h1:
                f = 0.2 * (h1 - h - y)
            else:
                f = 0.16 * (y + h)
            if (w1 - x - w) < 0.25 * w1:
                g = 0.15 * (w1 - x - w)
            else:
                g = 0.115 * (x + w)
            if x > w1 / 2:
                k = 0.145 * (w1 - x)
            else:
                k = 0.145 * x
            if y > h1 / 2:
                l = 0.2 * (h1 - y)
            else:
                l = 0.2 * y
            # wow=img[y-int(l):y+h+int(f),x-int(k):x+h+int(g)]
            wow = img[y:y + h, x:x + w]
            wow1 = blend_transparent(wow, img2)
            # img[y-int(l):y+h+int(f),x-int(k):x+h+int(g)]=wow1
            img[y:y + h, x:x + w] = wow1
        cv2.imshow('PRESS "s" TO CAPTURE', img)
        img1=img
    cap.release()
    cv2.destroyAllWindows()

def charselect(event):
    global root2
    root2=Toplevel()
    root2.geometry("400x400")
    ima=cv2.imread('firefox.jpg')
    b,c=ima.shape[:2]
    ima=cv2.resize(ima,None,fx=400/c,fy=400/b,interpolation=cv2.INTER_AREA)
    cv2.imwrite("bk.jpg",ima)
    load = Image.open("bk.jpg")
    render = ImageTk.PhotoImage(load)

    # labels can be text or images
    img = Label(root2, image=render)
    img.image = render
    img.place(x=0, y=0)
    #lab1=Label(root2,text="SELECT CHARACTER....",fg="white", bg="blue")
    #lab1.place(x=10,y=10)
    s1=Button(root2,text="Thor",fg="white", bg="black", width=12, height=2,relief='ridge',borderwidth=4)
    s2=Button(root2,text="Spiderman",fg="black", bg="white", width=12, height=2,relief='ridge',borderwidth=4)
    s3=Button(root2,text="Ironman",fg="white", bg="black", width=12, height=2,relief='ridge',borderwidth=4)
    s4=Button(root2,text="Batman",fg="black", bg="white", width=12, height=2,relief='ridge',borderwidth=4)
    s5=Button(root2,text="Superman",fg="white", bg="black", width=12, height=2,relief='ridge',borderwidth=4)
    s7=Button(root2,text="Captian America",fg="black", bg="white", width=12, height=2,relief='ridge',borderwidth=4)
    s8=Button(root2,text="Black Widow",fg="black", bg="white", width=12, height=2,relief='ridge',borderwidth=4)
    s10=Button(root2,text="Goku",fg="white", bg="black", width=12, height=2,relief='ridge',borderwidth=4)
    s12=Button(root2,text="Kratos",fg="white", bg="black", width=12, height=2,relief='ridge',borderwidth=4)
    s1.bind("<Button-1>",nw)
    s2.bind("<Button-1>", nw)
    s3.bind("<Button-1>",nw)
    s4.bind("<Button-1>",nw)
    s4.bind("<Button-1>",nw)
    s5.bind("<Button-1>",nw)
    s7.bind("<Button-1>",nw)
    s8.bind("<Button-1>", nw)
    s10.bind("<Button-1>", nw)
    s12.bind("<Button-1>", nw)
    s1.place(x=10,y=20)
    s2.place(x=150,y=20)
    s5.place(x=150,y=80)
    s7.place(x=290,y=80)
    s3.place(x=290,y=20)
    s4.place(x=10,y=80)
    s10.place(x=10,y=140)
    s8.place(x=150,y=140)
    s12.place(x=290,y=140)
    root2.mainloop()
#+++++++++++++++++++++main window============================================
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


#new = Button(root, text="SNAPSHOT", activebackground="green", fg="white", bg="blue", width=15, height=3,relief='ridge',borderwidth=7)
#new.bind("<Button-1>",nw)
#new.place(x=670, y=400)

app=Button(root,text="AVATARS",fg="white", bg="blue", width=15, height=3,relief='ridge',borderwidth=7)
app.bind("<Button-1>",charselect)
app.place(x=670,y=340)

view = Button(root, text="View", fg="white", bg="blue", width=15, height=3,relief='ridge',borderwidth=7)
view.place(x=670, y=600)
view.bind("<Button-1>",browseFile)

Filters = Button(root, text="Filters", fg="white", bg="blue", width=15, height=3,relief='ridge',borderwidth=7)
Filters.place(x=670, y=470)
Filters.bind("<Button-1>",lol33)

root.mainloop()