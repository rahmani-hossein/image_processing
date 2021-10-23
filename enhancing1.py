import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
def histogram(array):
    res=np.zeros(256,dtype=np.uint8)
    for pixel in array:
        res[pixel]=res[pixel]+1
    return res
#checking version
print(cv.__version__)
darkimage =cv.imread("C:/Users/hossein rahmani/Downloads/Enhance1.JPG")
print(darkimage.shape)
darkimage=cv.cvtColor(darkimage,cv.COLOR_BGR2RGB)
alpha=0.5
beta=1
# theta=0.3
# t=255*(darkimage/255)**theta
# t=np.asarray(t,dtype=np.uint8)
y=(255/np.log(1+alpha*255))*np.log(1+alpha*darkimage)
y=np.asarray(y,dtype=np.uint8)#hatman uint bashe va na int
z=(255/np.log(1+beta*255))*np.log(1+beta*darkimage)
z=np.asarray(z,dtype=np.uint8)
# cv.imwrite('C:/Users/hossein rahmani/PycharmProjects/begin/brigther45.jpg', t)
# cv.imshow("dg",t/255)
# cv.waitKey(0)
# y_hsv=cv.cvtColor(y,cv.COLOR_RGB2HSV)
# y_hsv[:,:,2]=cv.equalizeHist(y_hsv[:,:,2])
# out_y=cv.cvtColor(y_hsv,cv.COLOR_HSV2RGB)
# cv.imwrite('C:/Users/hossein rahmani/PycharmProjects/begin/equalized1.jpg', out_y)# by opencv
# cv.imshow("dg",out_y/255)
# cv.waitKey(0)
# cv.imwrite('C:/Users/hossein rahmani/PycharmProjects/begin/brigther1.jpg', y)
# cv.imshow("dg",y/255)
# cv.waitKey(0)



y_hsv=cv.cvtColor(y,cv.COLOR_RGB2HSV)
arr=y_hsv[:,:,2]
cm=np.cumsum(histogram(arr))
# cm=cm/cm.max()
# normalized_cum = cm*255
cmn=(cm-cm.mean())/cm.std()
cmn=(cmn+1)/2
normalized_cum=cmn*255
# N=cm.max()-cm.min()
# cmn=(cm-cm.min())*255
# normalized_cum=cmn/N
y_hsv[:,:,2]=normalized_cum[arr]
out_y=cv.cvtColor(y_hsv,cv.COLOR_HSV2RGB)
cv.imwrite('C:/Users/hossein rahmani/PycharmProjects/begin/equalizedlast.jpg', out_y)# by opencv


# equalized_y=np.zeros(y.shape,dtype=np.uint8)
# equalized_y[:,:,0]=y[:,:,0]
# for i in range(0,3):
#     arr=y[:,:,i]
#     cm=np.cumsum(histogram(arr))
#     N=cm.max()-cm.min()
#     cmn=(cm-cm.min())*255
#     normalized_cum=cmn/N
#     equalized_y[:,:,i]=normalized_cum[y[:,:,i]]
#
# cv.imwrite('C:/Users/hossein rahmani/PycharmProjects/begin/equalized2.jpg', equalized_y)

highcontrast_y=np.zeros(y.shape,dtype=np.uint8)
rangebit=np.arange(0,256)

for i in range(0,3):
    a=y[:,:,i]
    min_i=np.min(a[np.nonzero(a)])
    max_i=np.amax(a)
    range_i=max_i=min_i
    plt.plot(rangebit,histogram(a))
    highcontrast_y[:,:,i]=((a-min_i)/range_i)*255
cv.imwrite('C:/Users/hossein rahmani/PycharmProjects/begin/contrast2.jpg', y)
plt.show()



