import numpy as np
import cv2 as cv
import time

st=time.time()
def get_boxfilter(m,n):
   return np.ones((m,n))/(m*n)
def Ispink1(h,s,v):
    '''
    :param h: hue
    :param s: saturation
    :param v: brigthness
    :return: True if its pink,False otherwise 65 165
    '''
    if 140<=h<167 and (s>=30 or s<10) and (40<=v<256):
        return True
    else:
        return False

flower=cv.imread("C:/Users/hossein rahmani/Desktop/Flowers.jpg")
flower=cv.cvtColor(flower,cv.COLOR_BGR2RGB)
flower_hsv=cv.cvtColor(flower,cv.COLOR_RGB2HSV)
original_shape=flower.shape
yellow_flower=np.array(flower,dtype=np.uint8)
box_filter=get_boxfilter(3,3)
before=time.time()
print("----%.2f----"%(before-st))
for i in range(0,original_shape[0]):
    for j in range(0,original_shape[1]):
        if(Ispink1(flower_hsv[i,j,0],flower_hsv[i,j,1],flower_hsv[i,j,2])):
           yellow_flower[i,j,:]=0
           flower_hsv[i,j,0]=(flower_hsv[i,j,0]-155)+30   #yellow

flower_processed=cv.cvtColor(flower_hsv,cv.COLOR_HSV2BGR)
# flower_processed[yellow_flower==0]=0
flower_processed=box_filter[0,0]*flower_processed[0:-2,0:-2,:]+box_filter[0,1]*flower_processed[1:-1,0:-2,:]+box_filter[0,2]*flower_processed[2:,0:-2,:]\
                 +box_filter[1,0]*flower_processed[0:-2,1:-1,:]+box_filter[1,1]*flower_processed[1:-1,1:-1,:]+box_filter[1,2]*flower_processed[2:,1:-1,:]\
            +box_filter[2,0]*flower_processed[0:-2,2:,:]+box_filter[2,1]*flower_processed[1:-1,2:,:]+box_filter[2,2]*flower_processed[2:,2:,:]




print("----%.2f----"%(time.time()-before))
cv.imwrite('C:/Users/hossein rahmani/PycharmProjects/begin/yellowflower7.jpg',flower_processed)
