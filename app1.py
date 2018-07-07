from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import time
import math
import variable
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++===

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

        variable.img33 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\iron spider\leftarm.png", -1)
        variable.img44 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\iron spider\lefthand.png", -1)
        variable.img56 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\iron spider\rightarm.png", -1)
        variable.img67 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\iron spider\righthand.png", -1)
        variable.img78 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\iron spider\leftknee.png", -1)
        variable.img89 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\iron spider\leftleg.png", -1)
        variable.img90 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\iron spider\rightknee.png", -1)
        variable.img01 = cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\iron spider\rightleg.png", -1)
        variable.img1 =  cv2.imread(r"C:\Users\JASH VYAS\PycharmProjects\untitled1\iron spider\torsoe.png", -1)
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
    s6=Button(root2,text="Hulk",fg="black", bg="white", width=12, height=2,relief='ridge',borderwidth=4)
    s7=Button(root2,text="Captian America",fg="white", bg="black", width=12, height=2,relief='ridge',borderwidth=4)
    s8=Button(root2,text="Black Widow",fg="black", bg="white", width=12, height=2,relief='ridge',borderwidth=4)
    s9=Button(root2,text="Captian Marvel",fg="white", bg="black", width=12, height=2,relief='ridge',borderwidth=4)
    s10=Button(root2,text="Goku",fg="black", bg="white", width=12, height=2,relief='ridge',borderwidth=4)
    s11=Button(root2,text="Vegeta",fg="white", bg="black", width=12, height=2,relief='ridge',borderwidth=4)
    s12=Button(root2,text="Kratos",fg="black", bg="white", width=12, height=2,relief='ridge',borderwidth=4)
    s13 = Button(root2, text="Thanos", fg="white", bg="black", width=12, height=2, relief='ridge', borderwidth=4)
    s1.bind("<Button-1>",nw)
    s2.bind("<Button-1>", nw)
    s3.bind("<Button-1>",nw)
    s4.bind("<Button-1>",nw)
    s4.bind("<Button-1>",nw)
    s5.bind("<Button-1>",nw)
    s6.bind("<Button-1>",nw)
    s7.bind("<Button-1>",nw)
    s8.bind("<Button-1>", nw)
    s9.bind("<Button-1>", nw)
    s10.bind("<Button-1>", nw)
    s11.bind("<Button-1>", nw)
    s12.bind("<Button-1>", nw)
    s13.bind("<Button-1>",nw)
    s1.place(x=10,y=20)
    s2.place(x=150,y=20)
    s5.place(x=150,y=80)
    s6.place(x=290,y=80)
    s3.place(x=290,y=20)
    s4.place(x=10,y=80)
    s7.place(x=10,y=140)
    s8.place(x=150,y=140)
    s9.place(x=290,y=140)
    s10.place(x=10,y=200)
    s11.place(x=150,y=200)
    s12.place(x=290,y=200)
    s13.place(x=10,y=260)
    root2.mainloop()

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
app.place(x=670,y=400)

view = Button(root, text="View", fg="white", bg="blue", width=15, height=3,relief='ridge',borderwidth=7)
view.place(x=670, y=530)
view.bind("<Button-1>",browseFile)
root.mainloop()