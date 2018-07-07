#################################### IMPORTS #######################################################################################


from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk


import numpy as np
import cv2
import rotate
import time


#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& HAARCASCADE XMLS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


#========================================== the fuction to blend the images ==============================================================


def blend_transparent(img12, img1):

        height,width,channels=img1.shape
        h,w=img12.shape[:2]
        img1=cv2.resize(img1,None,fx=(w/width),fy=(h/height),interpolation=cv2.INTER_AREA)           ##################################

        # Split out the transparency mask from the colour info
        overlay_img = img1[:,:,:3] # Grab the BRG planes
        overlay_mask = img1[:,:,3:]  # And the alpha plane

        # Again calculate the inverse mask
        background_mask = 255 - overlay_mask

        # Turn the masks into three channel, so we can use them as weights
        overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
        background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

        # Create a masked out face image, and masked out overlay

        # We convert the images to floating point in range 0.0 - 1.0
        face_part = (img12* (1 / 255.0)) * (background_mask * (1 / 255.0))
        overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))


        # And finally just add them together, and rescale it back to an 8bit integer image
        return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))



# ============================================= another function to blend the images =========================================================


#def blend_transparent1(img12, img1):
#
#    height, width, channels = img1.shape
#    img1 = cv2.resize(img1, None, fx=(0.49 * (w / width)), fy=3 * (h / height), interpolation=cv2.INTER_AREA)  ##################################
#
#    # Split out the transparency mask from the colour info
#    overlay_img = img1[:, :, :3]  # Grab the BRG planes
#    overlay_mask = img1[:, :, 3:]  # And the alpha plane
#
#    # Again calculate the inverse mask
#    background_mask = 255 - overlay_mask
#
#    # Turn the masks into three channel, so we can use them as weights
#    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
#    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)
#
#    # Create a masked out face image, and masked out overlay
#
#    # We convert the images to floating point in range 0.0 - 1.0
#    face_part = (img12 * (1 / 255.0)) * (background_mask * (1 / 255.0))
#    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))
#
#
#    # And finally just add them together, and rescale it back to an 8bit integer image
#    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))
#

#======================================== reading the images =========================================================================


# MAIN IMAGE

img= rotate.man6


#+++++++++++++++++++++++++++++++++++++ the selective images for the torsoe ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


img1= cv2.imread("hulk1.png", -1) # Load with transparency
#img2= cv2.imread("hulkcutarm.png", -1)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% image processing begins ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:

    #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #cv2.rectangle(img, (x-w, y+h), (x +2*w, y + 5*h), (255, 0, 0), 2)  ########

    img12= img[rotate.points[1][1]:rotate.points[11][1],rotate.points[2][0]:rotate.points[5][0]]                           ##############
    #img13= img[y+2*h:y+5*h, x-w:x]

    result_2 = blend_transparent(img12, img1)
    img[rotate.points[1][1]:rotate.points[11][1], rotate.points[2][0]:rotate.points[5][0]]=result_2                        ##########

    #img13 = img[y + 2 * h:y + 5 * h, x - w:int(x-w/2)]

    #result_3 = blend_transparent1(img13, img2)
    #img[y+2*h:y+5*h, x-w:int(x-w/2)]=result_3


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ final image ***************************************************************************


cv2.imshow("hello",img)


#$$$$$$$$$$$$$$$$$$$$$$ end functions $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


cv2.waitKey(0)

cv2.destroyAllWindows()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END OF THE TORSOE PROGRAMME &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&