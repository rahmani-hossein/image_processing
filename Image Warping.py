import cv2 as cv
import numpy as np

def get_transform_matrix(x1,y1,x2,y2,x3,y3,x4,y4):#clockwise
    src_points=np.float32([[x1,y1], [x2,y2], [x3,y3], [x4,y4]])
    width=int(np.sqrt((x1-x2)**2 +(y1-y2)**2))
    heigth=int(np.sqrt((x2-x3)**2 +(y2-y3)**2))
    destination_points=np.float32([[0,0],[width-1,0],[width-1,heigth-1],[0,heigth-1]])
    projective_matrix = cv.getPerspectiveTransform(src_points, destination_points)
    return projective_matrix,width,heigth
def my_backward_warping(source,projective_matrix,shape):
    width=shape[1]
    heigth=shape[0]
    projected=np.zeros((heigth,width,3),dtype=np.uint8)
    inv_T=np.linalg.inv(projective_matrix)
    v=np.ones((3,1))
    for i in range(0,width):
        for j in range(0,heigth):
            #now x and y in opencv x=i,y=j
            v[0,0]=i
            v[1,0]=j
            loc=inv_T @ v
            x_prime=loc[0,0]/loc[2,0]
            y_prime=loc[1,0]/loc[2,0]#in opencv coordinates
            # print(loc)
            if 0 <= x_prime < source.shape[1] and 0 < y_prime < source.shape[0]:#bilinear interpolation
                x=np.floor(y_prime).astype(np.int64)
                a=np.abs(y_prime-x)
                y=np.floor(x_prime).astype(np.int64)
                b=np.abs(x_prime-y)
                projected[j,i,:]= (1-a)*(1-b)*source[x,y,:]+a*(1-b)*source[x+1,y,:]+(1-a)*b*source[x,y+1,:]+a*b*source[x+1,y+1,:]
            else:
                print('out of band')

    return projected


books = cv.imread("inputs/books.jpg")
projective_matrix,width,heigth=get_transform_matrix(666,208,600,395,321,289,382,108)
combinatory_book_manual=my_backward_warping(books,projective_matrix,(heigth,width))
cv.imwrite('outputs/combinatory_warping.jpg',combinatory_book_manual)
projective_matrix1,width1,heigth1=get_transform_matrix(359,741,157,708,208,428,409,466)
fourier=my_backward_warping(books,projective_matrix1,(heigth1,width1))
cv.imwrite('outputs/res17.jpg',fourier)
projective_matrix2,width2,heigth2=get_transform_matrix(813,969,609,1098,425,796,620,668)
image_science=my_backward_warping(books,projective_matrix2,(heigth2,width2))
cv.imwrite('outputs/res18.jpg',image_science)


