import numpy as np
import cv2 as cv


def laplace_stack(img, k_size=45, number_of_levels=5):
    """
    laplacian stack.
    cv2.getGaussianKernel(ksize, sigma[, ktype])
    # ksize - kernel size, should be odd and positive (3,5,...)
    # sigma - Gaussian standard deviation. If it is non-positive, it is computed from ksize as sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    # ktype - Type of filter coefficients (Optional)
    :param img:
    :param number_of_levels:
    :return:
    """
    sigma = 1
    guass_stack = []
    lap_stack = []
    for i in range(0, number_of_levels):
        smooth = cv.GaussianBlur(img, (k_size, k_size), sigma)
        # smooth = gaussian_filter(img, sigma=sigma)
        guass_stack.append(smooth)
        sigma = sigma * 1.5

    for i in range(0, number_of_levels):
        if i == number_of_levels - 1:
            lap_stack.append(guass_stack[i])
        else:
            lap_stack.append(guass_stack[i] - guass_stack[i + 1])
    return lap_stack, guass_stack


def blend(lap_stack1, lap_stack2, img_shape, mask):
    size = [3,11,21,101]
    number_of_levels = len(lap_stack1)
    orapple = np.zeros(img_shape, np.float32)
    for i in range(0, number_of_levels):
        new_mask = cv.GaussianBlur(mask, (size[i],size[i]), 0)
        blend_level = lap_stack1[i] * new_mask + lap_stack2[i] * (1 - new_mask)  # feathering
        orapple = orapple + blend_level

    return orapple


apple = cv.imread('res08.jpg')
apple = np.float32(apple)
orange = cv.imread('res09.jpg')
orange = np.float32(orange)
mask = np.zeros(apple.shape,dtype=np.float32)
mask[:, 0:256,:] = 1
# orpple=apple *mask + orange*(1-mask)
# cv.imwrite('orplle_naive.jpg',orpple)
laplace_apple_stack, guassian_apple_stack = laplace_stack(apple, number_of_levels=4)
laplace_orange_stack, guassian_orange_stack = laplace_stack(orange, number_of_levels=4)
orapple = blend(laplace_apple_stack, laplace_orange_stack, apple.shape, mask)
cv.imwrite('res10.jpg', orapple)