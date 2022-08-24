import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from cv2 import destroyAllWindows
from typing import Dict
import cv2
from cv2 import approxPolyDP
import numpy as np
from collections import OrderedDict
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf


def segment(img): 
    
    img2=img.copy()

    imgBlur=cv2.medianBlur(img2, 3) 
    imgGray=cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)


    ret, thresh_img = cv2.threshold(imgGray, 150, 200, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5,5), np.uint8)
    img_dilation = cv2.dilate(thresh_img, kernel, iterations=7)

    # cv2.imshow('Dil', img_dilation)

    contours, hiererachy= cv2.findContours(img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    points=[[0, 0, 0, 0]]
    centre=[[0, 0]]

    Dict={}

    for cnt in contours:
        A=cv2.contourArea(cnt)
        if A>5000:
            Dict[A]=cnt

    Dict=OrderedDict(sorted(Dict.items(), reverse=True))

    for A, cnt in Dict.items():

        peri=cv2.arcLength(cnt, True)
        approx=approxPolyDP(cnt, 0.02*peri, True)

        x, y, w, h= cv2.boundingRect(approx)
        # points
        coordinates=[x, y, x+w, y+h]

        M = cv2.moments(cnt)
        Cx=int(M["m10"]/M["m00"])
        Cy=int(M["m01"]/M["m00"])

        flag=True
        i=50

        for a, b in centre:
            if (a-Cx<i and a-Cx>-i and b-Cy<i and b-Cy>-i) :
                flag=False

        if flag:
            points.append(coordinates)
            centre.append([Cx, Cy])

    i=0
    t=15

    points.sort(key = lambda x: x[0])

    cropped_images=[]

    for x1, y1, x2, y2 in points:

        if i==0:
            i=i+1
            continue

        x1=x1-t
        y1=y1-t
        x2=x2+t
        y2=y2+t
        i=i+1

        cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 255, 0), 2) 
        cv2.rectangle(img_dilation, (x1, y1), (x2, y2), (0, 255, 0), 2) 

        cropped_img=img[y1:y2, x1:x2]

        cropped_images.append(cropped_img)

    i=1

    for cropped_img in cropped_images:
        try:
            cv2.imwrite(f'crroped_img_{i}.png', cropped_img)
            i=i+1
        except:
            continue

    return cropped_images



model = keras.models.load_model('ps1/mosaic/model.h5')
# model = keras.models.load_model('model_1.h5')

labels_dict = {10:65,11:66,12:67,13:68,14:69,15:70,16:71,17:72,18:73,19:74,20:75,21:76,22:77,23:78,24:79,25:80,26:81,27:82,28:83,29:84,30:85,31:86,32:87,33:88,34:89,35:90}


########################################################################

def test(path):
    image=cv2.imread(path,1)
    images = segment(image)

    # print("Total images",len(images))
    IMG_SIZE = 28

    for image in images:
        try:        
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.bitwise_not(image)
            image = cv2.resize(image,(28,28))

            # image = cv2.GaussianBlur(image,(3,3),0)
            _ , image = cv2.threshold(image,150,255,cv2.THRESH_BINARY)
            image = image/255.0

            image = image.reshape(1,28,28,1)


            answer = np.argmax(model.predict(image))

            if answer<=7:
                print(answer,end="")
            else:
                print(chr(labels_dict[answer]),end="")
        except:
            continue

    print('\n')



########################################################################


path_of_image = "ps1/mosaic/8.jpeg"

test(path_of_image)