import numpy as np
import cv2 as cv
import scipy.sparse
import pyamg


def poisson_blend(target, source, mask, region_target, region_source):

    mask = mask[region_source[0]:region_source[2], region_source[1]:region_source[3]]
    region_heigth = region_source[2] - region_source[0]
    region_width = region_source[3] - region_source[1]
    region_area = region_width * region_heigth
    A = scipy.sparse.identity(region_area, format='lil')
    for i in range(region_heigth):
        for j in range(region_width):
            if mask[i, j] == 1:
                index = j + i * region_width
                A[index, index] = 4
                if index + 1 < region_area:
                    A[index, index + 1] = -1
                if index - 1 >= 0:
                    A[index, index - 1] = -1
                if index + region_width < region_area:
                    A[index, index + region_width] = -1
                if index - region_width >= 0:
                    A[index, index - region_width] = -1
    A = A.tocsr()
    # create poisson matrix for b
    P = pyamg.gallery.poisson(mask.shape)

    # for each channel (BGR)
    for channel in range(target.shape[2]):
        t = target[region_target[0]:region_target[2], region_target[1]:region_target[3], channel]
        s = source[region_source[0]:region_source[2], region_source[1]:region_source[3], channel]
        t = t.flatten()
        s = s.flatten()

        # create b
        print(P.shape,s.shape)
        b = P * s
        for i in range(region_heigth):
            for j in range(region_width):
                if mask[i, j] == 0:
                    index = j + i * region_width
                    b[index] = t[index]

        # solve Ax = b
        x, istop, itn, r1norm = scipy.sparse.linalg.lsqr(A,b)[:4]
        # x = pyamg.solve(A, b, verb=False, tol=1e-10) memory limit mishod.
        x = np.reshape(x, (region_heigth, region_width))
        x[x > 255] = 255
        x[x < 0] = 0
        x = np.array(x, target.dtype)
        target[region_target[0]:region_target[2], region_target[1]:region_target[3], channel] = x

    return target


# target = np.array([[2, 5, 2, 2], [7, 7, 7, 7], [9, 9, 8, 9]], dtype=np.float32)
# source = np.array([[8, 6, 6, 6], [6, 6, 2, 6], [6, 8, 6, 6]], dtype=np.float32)
# mask = np.zeros((3, 4))
# mask[1, 1:3] = 1
# mask[np.where(mask>200)]=255
# mask[np.where(mask<200)]=0
# cv.imwrite('source_mask.jpg',mask)
source=cv.imread('res05.jpg')
source = np.float32(source)
height=145
width=181
region_source = (63,56,63+height,width+56)
mask = cv.imread('source_mask.jpg',0)
mask=mask/255
mask=mask.astype(np.uint8)
region_target = (280,385,280+height,385+width)
target=cv.imread('res06.jpg')
target = np.float32(target)
new_target = poisson_blend(target,source,mask,region_target,region_source)

cv.imwrite('res07.jpg',new_target)
