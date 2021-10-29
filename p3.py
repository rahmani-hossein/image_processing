import numpy as np
import cv2 as cv
from PIL import Image
import time


def mse_error(img1, img2):
    img1 = img1.astype(np.int64)
    img2 = img2.astype(np.int64)
    return np.sum(np.power(img1 - img2, 2))


def abs_error(img1, img2):
    return np.sum(np.abs(img1 - img2))


#  image pyramid with scaling 2
def search_shift(background_image, rolling_image, x_shift, y_shift, s_x=10, s_y=10):
    shiftx = 2 * x_shift
    shifty = 2 * y_shift
    min_error = mse_error(background_image, np.roll(rolling_image, [shiftx, shifty], axis=(0, 1)))
    for i in range(np.max(2 * x_shift - s_x), 2 * x_shift + s_x + 1):
        for j in range(np.max(2 * y_shift - s_y), 2 * y_shift + s_y + 1):
            error = mse_error(background_image, np.roll(rolling_image, [i, j], axis=(0, 1)))
            if error < min_error:
                shiftx = i
                shifty = j
                min_error = error

    return [shiftx, shifty]


def find_shift(background_image, rolling_image):
    shape = rolling_image.shape
    x_shift = 0
    y_shift = 0
    min_error = mse_error(background_image, rolling_image)
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            error = mse_error(background_image, np.roll(rolling_image, [i, j], axis=(0, 1)))
            if error < min_error:
                x_shift = i
                y_shift = j
                min_error = error

    return [x_shift, y_shift]


# cutting to 3 channels


image_address = "C:/Users/hossein rahmani/Desktop/master-pnp-prok-01800-01886a.tif"
st = time.time()
image = Image.open(image_address)
image = np.asarray(image)
row, column = image.shape
channel_row = int(row / 3)
b_channel = image[0:channel_row, :]
g_channel = image[channel_row:channel_row * 2, :]
r_channel = image[channel_row * 2:channel_row * 3, :]
image = np.dstack((b_channel, g_channel, r_channel))
image_pyramid = []
for i in range(0, 4):
    width = int(image.shape[1] / (2 ** i))
    height = int(image.shape[0] / (2 ** i))
    dim = (width, height)
    half_image = cv.resize(image, dim)
    image_pyramid.append(half_image)
image_pyramid.reverse()
print(time.time()-st)
x_g=0
y_g=0
x_r=0
y_r=0
for i in range(0, 4):
    current_image = image_pyramid[i]
    b = current_image[:, :, 0]
    g = current_image[:, :, 1]
    r = current_image[:, :, 2]
    if i == 0:
        x_g, y_g = find_shift(b, g)
        x_r, y_r = find_shift(b, r)
        print(time.time()-st)
    else:
        x_g, y_g = search_shift(b, g, x_g, y_g)
        x_r, y_r = search_shift(b, r, x_r, y_r)
        print(time.time() - st)

b = image[:, :, 0]
g = image[:, :, 1]
r = image[:, :, 2]
g = np.roll(g, [x_g, y_g], axis=(0, 1))
r = np.roll(r, [x_r, y_r], axis=(0, 1))
image[:, :, 1] = g
image[:, :, 2] = r
image = (255*((image - image.min())/image.ptp())).astype(np.uint8)
cv.imwrite('C:/Users/hossein rahmani/Desktop/amir2_jpg.jpg', image)
print(time.time() - st)
