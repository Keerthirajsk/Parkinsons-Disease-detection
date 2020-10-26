from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk
import seaborn as sns
from pymongo import MongoClient

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from module2 import knn
from module4 import svm
from module3 import decision_tree
from disease import disease

from sklearn.model_selection import train_test_split




def k_nearest_neighbour():
    ltext = 'Disease Analysis Model Accuracy\n\n KNN Model Accuracy: ' + str(knn())+'\n\n Total Predicted:'+str(disease['total'])+' True:' + str(disease['true'])+' False:'+str(disease['false'])
    labeltext.set(ltext)
    names = ['Predicted True', 'Predicted False']
    values = [disease["true"], disease["false"]]

    plt.figure(figsize=(9, 12))

    plt.bar(names, values)
    plt.suptitle('Prediction Analysis')
    plt.show()
    disease["true"] = 0
    disease["false"] = 0
    disease["total"] = 0

def support_vector():
    ltext = 'Disease Analysis Model Accuracy\n\n Support Vector Machine Model Accuracy: ' + str(svm())+'\n\n Total Predicted:'+str(disease['total'])+' True:' + str(disease['true'])+' False:'+str(disease['false'])
    labeltext.set(ltext)
    names = ['Predicted True', 'Predicted False']
    values = [disease["true"], disease["false"]]

    plt.figure(figsize=(9, 12))

    plt.bar(names, values)
    plt.suptitle('Prediction Analysis')
    plt.show()
    disease["true"] = 0
    disease["false"] = 0
    disease["total"] = 0

def decision_tre():
    ltext = 'Disease Analysis Model Accuracy\n\n Decision Tree Model Accuracy: ' + str(decision_tree())+'\n\n Total Predicted:'+str(disease['total'])+' True:' + str(disease['true'])+' False:'+str(disease['false'])
    labeltext.set(ltext)
    names = ['Predicted True', 'Predicted False']
    values = [disease["true"], disease["false"]]

    plt.figure(figsize=(9, 12))

    plt.bar(names, values)
    plt.suptitle('Prediction Analysis')
    plt.show()
    disease["true"] = 0
    disease["false"] = 0
    disease["total"] = 0

def graph():
    names = ['Predicted True', 'Predicted False']
    values = [disease["true"], disease["false"]]

    plt.figure(figsize=(9, 12))



    plt.bar(names, values)
    plt.suptitle('Prediction Analysis')
    plt.show()
    disease["true"]=0
    disease["false"] = 0
    disease["total"]=0


def kn():
     k_nearest_neighbour()

def sv():
    support_vector()

def dc():
    decision_tre()



def overall():
    global top
    global lres
    global labeltext
    global lt
    top = Toplevel()
    top.geometry('1500x1500')
    top.title("Overall Parkinsons Disease Analysis")

    top.configure(background="black")
   
    b2 = Button(top, text="K Nearest Neighbour",width=21,font=("Calibri", 25, "bold"), bg="black", fg="red", command=kn).place(x=10,y=200)
    b3 = Button(top, text="Support Vector Machine",width=21, font=("Calibri", 25, "bold"), bg="black", fg="red", command=sv).place(x=10,y=300)
    b4 = Button(top, text="Decision Tree",width=21, font=("Calibri", 25, "bold"), bg="black", fg="red", command=dc).place(x=10,y=400)
    

    labeltext = StringVar()
    labeltext.set("Disease Analysis Model Accuracy")
    
    lres = Label(top, font=("Calibi", 20, "bold"), fg="white", bg="grey", width=50, height=10,
                 textvariable=labeltext).place(x=400,y=165)

    top.configure()
    top.mainloop()


def dataRead():
    global df
    df = pd.read_csv('parkinsons.data', low_memory=False)


def info():
    global top
    global df
    top = Toplevel()
    top.geometry('1500x1500')
    top.title("Overall Information")
    top.configure(background="black")

    df = pd.read_csv('parkinsons.data', low_memory=False)
  
    txt = ''
    for i in df.columns.tolist():
        if i=="Shimmer:APQ5":
            txt=txt+"\n"
        if i=="PPE":
            txt = txt + str(i)
            break
        txt = txt + str(i) + ","
    l1 = Label(top, text="Total number of records :\n", bg="black", fg="white", font=("calibri", 20, "bold")).place(x=100,y=100)
    l = Label(top, text=str(df.head(len(df))), bg="black", fg="white", font=("calibri", 17, "bold")).place(x=100,y=150)
    l2 = Label(top, text="\nColumns :", bg="black", fg="white", font=("calibri", 20, "bold")).place(x=100,y=500)
    l3 = Label(top, text=txt,justify=LEFT, bg="black", fg="white", font=("calibri", 17, "bold")).place(x=100,y=580)
    Button(top, text="Show Graph", width=21, font=("Calibri", 25, "bold"), bg="black", fg="red", command=ovrall).place(
        x=900, y=280)


 

def ovrall():
    global df
    names = ['without disease', 'with disease']
    values = [len(df.groupby('status').get_group(0)), len(df.groupby('status').get_group(1))]

    plt.figure(figsize=(9, 4))

    plt.subplot(131)
    plt.bar(names, values)
    plt.subplot(132)
    plt.scatter(names, values)
    plt.subplot(133)
    plt.plot(names, values)
    plt.suptitle('Disease Analysis')
    plt.show()
def main():
    window = Tk()
    window.title("Parkinsons disease Analysis")
    window.geometry('1500x1500')
    window.configure(background="black")


    image = Image.open("parkin.jpg")
    photo = ImageTk.PhotoImage(image)
    l1 = Label(window, text="Parkinsons Data Analysis", bg="black", fg="white", font=("Calibri", 29, "bold")).place(x=800,y=140)
    b1 = Button(window, text="Click Me!", fg="red", image=photo, width=400, height=200, command=overall).place(x=800,y=200)


    image1 = Image.open("info.jpg")
    photo1 = ImageTk.PhotoImage(image1)
    l2 = Label(window, text="\nOverall Disease Information", bg="black", fg="white",
               font=("Calibri", 29, "bold")).place(x=100,y=100)
    b2 = Button(window, image=photo1, width=400, height=200, command=info).place(x=100,y=200)


    b2 = Button(window, text="QUIT", bg="black", fg="red", font=("Calibri", 29, "bold"), command=quit).place(x=100,y=500)
    window.mainloop()




if __name__ == '__main__':
    main()
