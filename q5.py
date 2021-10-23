import numpy as np
import cv2 as cv
import time

st=time.time()
pink=cv.imread("C:/Users/hossein rahmani/Desktop/Pink.jpg")
res=cv.boxFilter(pink,-1,(3,3),cv.BORDER_REPLICATE)
print("----%.2f----"%(time.time()-st))
cv.imwrite('C:/Users/hossein rahmani/PycharmProjects/begin/ex1/res3.jpg',res)