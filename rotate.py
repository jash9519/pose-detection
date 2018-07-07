#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% IMPORTS ###################################################################################################


import cv2
import time
import numpy as np
import math
import app1
import variable

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


#=================================================== reading the images =====================================================================


# MAIN IMAGE OR THE READ IMAGE
frame =cv2.imread("man8.jpg")
h1,w1=frame.shape[:2]
#frame=cv2.resize(frame,None,fx=1.2,fy=1,interpolation=cv2.INTER_AREA)


# THE SELECTION IMAGES
img33 = variable.img33
img44 = variable.img44
img56 = variable.img56
img67 = variable.img67
img78 = variable.img78
img89 = variable.img89
img90 = variable.img90
img01 = variable.img01
img1=   variable.img1
#img33 = cv2.imread(r"C:\Users\Meet\PycharmProjects\untitled\venv\thor\leftarm.png", -1)
#img44 = cv2.imread(r"C:\Users\Meet\PycharmProjects\untitled\venv\thor\lefthand.png", -1)
#img56 = cv2.imread(r"C:\Users\Meet\PycharmProjects\untitled\venv\thor\rightarm.png", -1)
#img67 = cv2.imread(r"C:\Users\Meet\PycharmProjects\untitled\venv\thor\righthand.png", -1)
#img78 = cv2.imread(r"C:\Users\Meet\PycharmProjects\untitled\venv\thor\leftknee.png", -1)
#img89 = cv2.imread(r"C:\Users\Meet\PycharmProjects\untitled\venv\thor\leftleg.png", -1)
#img90 = cv2.imread(r"C:\Users\Meet\PycharmProjects\untitled\venv\thor\rightknee.png", -1)
#img01 = cv2.imread(r"C:\Users\Meet\PycharmProjects\untitled\venv\thor\rightleg.png", -1)
#img1 = cv2.imread(r"C:\Users\Meet\PycharmProjects\untitled\venv\thor\torsoe.png", -1)

def pointer(p1,p2,p3,p4):
    if p1>p2:
        x1=p2
        x2=p1
    else:
        x2=p2
        x1=p1
    if p3>p4:
        y1=p4
        y2=p3
    else:
        y1=p3
        y2=p4
    return x1,x2,y1,y2

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
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),(0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)
output = net.forward()

print("time taken by network : {:.3f}".format(time.time() - t))

H = output.shape[2]
W = output.shape[3]

# Empty list to store the detected keypoints
points = []


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ initializing the for loop of contours $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


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
        cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        # Add the point to the list if the probability is greater than the threshold
        points.append((int(x), int(y)))

    else :


############################################ Draw Skeleton #######################################################################

#%%%%%%%%%&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& rotating the image (fuction) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        points.append(None)

def rotateAndScale(img, degreesCCW , scaleFactor = 1):

    (oldY,oldX) = img.shape[:2] #note: numpy uses (y,x) convention but most OpenCV functions use (x,y)

    M = cv2.getRotationMatrix2D(center=(oldX/2,oldY/2), angle=degreesCCW, scale=scaleFactor) #rotate about center of image.

    #choose a new image size.
    newX,newY = oldX*scaleFactor,oldY*scaleFactor

    #include this if you want to prevent corners being cut off
    r = np.deg2rad(degreesCCW)
    newX,newY = (abs(np.sin(r)*newY) + abs(np.cos(r)*newX),abs(np.sin(r)*newX) + abs(np.cos(r)*newY))

    #the warpAffine function call, below, basically works like this:

    # 1. apply the M transformation on each pixel of the original image
    # 2. save everything that falls within the upper-left "dsize" portion of the resulting image.

    #So I will find the translation that moves the result to the center of that region.
    (tx,ty) = ((newX-oldX)/2,(newY-oldY)/2)
    M[0,2] += tx #third column of matrix holds translation, which takes effect after rotation.
    M[1,2] += ty


    rotatedImg = cv2.warpAffine(img, M, dsize=(int(newX),int(newY)))
    return rotatedImg

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$  torsoe &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
t1=(points[11][1]+points[1][1])/2
t2=(points[5][0]+points[2][0])/2
d=0
if points[1][0] - points[14][0]!=0:
    slope = ( points[1][1] - points[14][1] )/( points[1][0] - points[14][0])
    a = math.atan(slope)
    d = math.degrees(a)
if d>0:
    d=90-d
elif d<0:
    d=-(90+d)
else:
    d=0.000
ret3=rotateAndScale(img1,d)
img190= frame[points[1][1]-int(0.07*t1):points[11][1]+int(0.1*t1),points[2][0]-int(0.1*t2 ):points[5][0]+int(0.1*t2 )]
result_2 = blend_transparent(img190,ret3)
frame[points[1][1]-int(0.07*t1):points[11][1]+int(0.1*t1),points[2][0]-int(0.1*t2 ):points[5][0]+int(0.1*t2 )] = result_2

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ left arm &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
x1,x2,y1,y2=pointer(points[2][0],points[3][0],points[3][1],points[2][1])

d=0
if x1 > w1 / 2:
    k = 0.1 * (w1 - x)
else:
    k = 0.1 * x
if y1 > h1 / 2:
    l = 0.08 * (h1 - y)
else:
    l = 0.08 * y
if x1-x2!=0:
    slope =( points[3][1]-points[2][1])/(points[3][0]-points[2][0])
    s = math.atan(slope)
    d = math.degrees(s)
if d>0:
    d= 90 - d
if d<0:
    d=-(90+d)
if points[2][0]<points[3][0]|points[2][1]>points[3][1]:
    d=d+180
img12 = frame[y1-int(l):y2+20 , x1-int(k):20+x2]
ret=rotateAndScale(img33,d)
ret2 = blend_transparent(img12,ret)

height , width = ret2.shape[:2]
h = y2 + 20-(y1 -int(l))
w =20 + x2-(x1 -int(k))

final1 = cv2.resize(ret2, None, fx= (w / width), fy= (h / height), interpolation=cv2.INTER_AREA)
frame[y1 -int(l):y2 + 20, x1 -int(k):20 + x2] = final1

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ left hand $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$44


img34 = frame[int(0.95*points[3][1]):int(1.33*points[4][1]) , int(0.75*points[4][0]):int(1.18*points[3][0])]                              #@@@@@@@@@@@@@@@@@@
d=0
if  points[3][0] - points[4][0]!=0:
    slope = ( points[3][1] - points[4][1] )/( points[3][0] - points[4][0] )
    s = math.atan(slope)
    d = math.degrees(s)
if d>0:
    d=90-d
if d<0:
    d=-(90+d)
ret1=rotateAndScale(img44,d)
ret2 = blend_transparent(img34,ret1)

height , width = ret2.shape[:2]
w = int(1.18*points[3][0]) - int(0.75*points[4][0])
h = int(1.33*points[4][1])-int(0.95*points[3][1])

final1 = cv2.resize(ret2, None, fx= (w / width), fy= (h / height), interpolation=cv2.INTER_AREA)
frame[int(0.95*points[3][1]):int(1.33*points[4][1]), int(0.75*points[4][0]):int(1.18*points[3][0])] = final1


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ right arm $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


img55 = frame[points[5][1]:points[6][1] , points[5][0]-20:points[6][0]+20]
d=0
if points[5][0] - points[6][0]!=0:
    slope = ( points[5][1] - points[6][1] )/( points[5][0] - points[6][0] )
    a = math.atan(slope)
    d = math.degrees(a)
if d>0:
    d=90-d
if d<0:
    d=-(90+d)
ret3=rotateAndScale(img56,d)
ret4 = blend_transparent(img55,ret3)

height , width = ret4.shape[:2]
w = points[6][0] - points[5][0]+40
h = points[6][1] - points[5][1]

final1 = cv2.resize(ret4, None, fx= (w / width), fy= (h / height), interpolation=cv2.INTER_AREA)
frame[points[5][1]:points[6][1], points[5][0]-20:points[6][0]+20] = final1


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ right hand $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


img66 = frame[points[6][1]-30:points[7][1]+50, points[7][0]-40:points[6][0]+40]
d=0
if points[7][0] - points[6][0]!=0:
    slope = ( points[7][1] - points[6][1] )/( points[7][0] - points[6][0] )
    a = math.atan(slope)
    d = math.degrees(a)
if d>0:
    d=90-d
if d<0:
    d=-(90+d)
ret3=rotateAndScale(img67,d)
ret4 = blend_transparent(img66,ret3)

height , width = ret4.shape[:2]
w = points[6][0] - points[7][0]+80
h = points[7][1] - points[6][1]+80

final1 = cv2.resize(ret4, None, fx= (w / width), fy= (h / height), interpolation=cv2.INTER_AREA)
frame[points[6][1]-30:points[7][1]+50, points[7][0]-40:points[6][0]+40] = final1


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ left knee $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


img77 = frame[int(0.95*points[8][1]):int(1.06*points[9][1]) , int(0.9*points[9][0]):int(1.1*points[8][0])]
d=0
if points[9][0] - points[8][0]!=0:
    slope = ( points[9][1] - points[8][1] )/( points[9][0] - points[8][0])
    a = math.atan(slope)
    d = math.degrees(a)
if d>0:
    d=90-d
if d<0:
    d=-(90+d)
ret3=rotateAndScale(img78,d)
ret4 = blend_transparent(img77,ret3)

height , width = ret4.shape[:2]
w = int(1.1*points[8][0])-int(0.9*points[9][0])
h = int(1.1*points[9][1]) - points[8][1]

final1 = cv2.resize(ret4, None, fx= (w / width), fy= (h / height), interpolation=cv2.INTER_AREA)
frame[points[8][1]:int(1.1*points[9][1]), int(0.9*points[9][0]):int(1.1*points[8][0])] = final1


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ left leg $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


img88 = frame[points[9][1]:int(1.09*points[10][1]) , int(0.92*points[10][0]):int(1.13*points[9][0])]
d=0
if points[10][0] - points[9][0]!=0:
    slope = ( points[10][1] - points[9][1] )/( points[10][0] - points[9][0] )
    a = math.atan(slope)
    d= math.degrees(a)
if d>0:
    d=90-d
if d<0:
    d=-(90+d)
ret3=rotateAndScale(img89,d)
ret4 = blend_transparent(img88,ret3)

height , width = ret4.shape[:2]
w = int(1.13*points[9][0]) - int(0.92*points[10][0])
h = int(1.09*points[10][1]) - points[9][1]

final1 = cv2.resize(ret4, None, fx= (w / width), fy= (h / height), interpolation=cv2.INTER_AREA)
frame[points[9][1]:int(1.09*points[10][1]), int(0.92*points[10][0]):int(1.13*points[9][0])] = final1


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ right knee $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


img99 = frame[points[11][1]:int(1.1*points[12][1]) , int(0.9*points[12][0]):int(1.1*points[11][0])]
d=0
if  points[12][0] - points[11][0]!=0:
    slope =( points[12][1] - points[11][1] )/( points[12][0] - points[11][0])
    a = math.atan(slope)
    d= math.degrees(a)
if d>0:
    d=90-d
if d<0:
    d=-(90+d)
ret3=rotateAndScale(img90,d)
ret4 = blend_transparent(img99,ret3)

height , width = ret4.shape[:2]
w = int(1.1*points[11][0])-int(0.9*points[12][0])
h = int(1.1*points[12][1]) - points[11][1]

final1 = cv2.resize(ret4, None, fx= (w / width), fy= (h / height), interpolation=cv2.INTER_AREA)
frame[points[11][1]:int(1.1*points[12][1]),int(0.9* points[12][0]):int(1.1*points[11][0])] = final1


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ right leg $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


img00 = frame[points[12][1]:int(1.1*points[13][1]) , int(0.92*points[13][0]):int(1.1*points[12][0])]
d=0
if points[13][0] - points[12][0]!=0:
    slope = ( points[13][1] - points[12][1] )/( points[13][0] - points[12][0] )
    a = math.atan(slope)
    d = math.degrees(a)
if d>0:
    d=90-d
if d<0:
    d=-(90+d)
ret3=rotateAndScale(img01,d)
ret4 = blend_transparent(img00,ret3)

height , width = ret4.shape[:2]
w = int(1.1*points[12][0]) - int(0.92*points[13][0])
h = int(1.1*points[13][1]) - points[12][1]

final1 = cv2.resize(ret4, None, fx= (w / width), fy= (h / height), interpolation=cv2.INTER_AREA)
frame[points[12][1]:int(1.1*points[13][1]), int(0.92*points[13][0]):int(1.1*points[12][0])] = final1





#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ final image $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

cv2.imshow("rotated image",frame)
#cv2.imwrite("jashpandu.jpg",frame)
#man6 = frame

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ end functions ########################################################################

cv2.waitKey(0)

cv2.destroyAllWindows()

################################# THE END OF THE IMAGE PROCESSING PROGRAMME ##################################################