#color processing and blurring
import numpy as np
import cv2 as cv
import time
st=time.time()
import matplotlib.pyplot as plt
def Ispink1(h,s,v):
    '''
    :param h: hue
    :param s: saturation
    :param v: brigthness
    :return: True if its pink,False otherwise 155
    '''
    if 140<=h<155 and 25<=s<256 and 30<=v<256:
        return True
    else:
        return False



def get_boxfilter(m,n):
   return np.ones((m,n))/(m*n)

flower=cv.imread("C:/Users/hossein rahmani/Desktop/Flowers.jpg")
flower=cv.cvtColor(flower,cv.COLOR_BGR2RGB)
flower_hsv=cv.cvtColor(flower,cv.COLOR_RGB2HSV)
original_shape=flower.shape
processed_image=np.zeros(original_shape,np.uint8)
box_filter=get_boxfilter(3,3)
before=time.time()
print("----%.2f----"%(before-st))
for i in range(0,original_shape[0]):
    for j in range(0,original_shape[1]):
        if(Ispink1(flower_hsv[i,j,0],flower_hsv[i,j,1],flower_hsv[i,j,2])):
            processed_image[i,j,:]=np.array([0,255,255]) #converting to yellow
        else:
                if 1 <= i < original_shape[0]-1 and 1<=j <original_shape[1]-1:
                    #convolve filter
                    processed_image[i,j,0]=np.multiply(flower[i-1:i+2,j-1:j+2,0], box_filter).sum()    #elementwise multiplication
                    processed_image[i,j,1]=np.multiply(flower[i - 1:i + 2, j - 1:j + 2, 1], box_filter).sum()
                    processed_image[i,j,2]=np.multiply(flower[i - 1:i + 2, j - 1:j + 2, 2], box_filter).sum()
print("----%.2f----"%(time.time()-before))
cv.imwrite('C:/Users/hossein rahmani/PycharmProjects/begin/ex1/yellowflower6.jpg',processed_image)



