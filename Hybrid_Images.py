import cv2 as cv
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt


def find_cutoff(fourier_image):
    heigth, width = fourier_image.shape
    image_amplitude = np.abs(fourier_image)
    frequency_sum = np.sum(image_amplitude)
    distances = calculate_distances(fourier_image)
    for d0 in range(np.minimum(heigth, width)):
        in_cutoff_frequency = 0
        for i in range(heigth):
            for j in range(width):
                if distances[i, j] <= d0:
                    in_cutoff_frequency += image_amplitude[i, j]
        frac = in_cutoff_frequency / frequency_sum
        if np.abs(frac - 0.5) < 0.00001:
            print(frac - 0.5, d0)
            return d0
        elif frac > 0.5:
            return d0


def calculate_distances(image):
    heigth, width = image.shape
    distances = np.zeros((heigth, width))
    center = (heigth / 2, width / 2)
    for i in range(heigth):
        for j in range(width):
            distances[i, j] = distance((i, j), center)

    return distances


def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def guassian_lowpass(D0, shape):
    heigth, width = shape[:2]
    guass_filter_lp = np.zeros(shape[:2])
    center = (heigth / 2, width / 2)
    for x in range(heigth):
        for y in range(width):
            guass_filter_lp[x, y] = np.exp((-distance((x, y), center) ** 2) / (2 * (D0 ** 2)))
    return guass_filter_lp


def guassiankern(length=3, std=1):
    """Returns a 2D Gaussian kernel array."""
    guassiankernel1d = signal.gaussian(length, std=std).reshape(length, 1)
    guassiankernel2d = guassiankernel1d @ guassiankernel1d.T
    return guassiankernel2d


def my_resize(image, a, b):  # image tol , arzash a brabar o b barabar kochk mishe mishe
    width = int(image.shape[1] / a)
    height = int(image.shape[0] / b)
    resized_image = cv.resize(image, (width, height))
    return resized_image

def compound(I1,I2):
    hybrid_image=np.zeros(I1.shape,dtype=np.complex64)
    guassian=guassiankern(np.maximum(I1.shape[0],I1.shape[1]),90)# by experiment
    center = (int(I1.shape[0] / 2),int(I1.shape[1] / 2))
    for i in range(I1.shape[0]):
        for j in range(I1.shape[1]):
            hybrid_image[i, j] = 1.30*(I1[i, j] * guassian[i, j] + I2[i, j] * (1 - guassian[i, j]))

    return hybrid_image
    # return I1+I2

def inverse_transform(I):
    shifted_I_fourier = np.fft.ifftshift(I)
    hybrid_image = np.fft.ifft2(shifted_I_fourier)
    hybrid_image = np.real(hybrid_image)
    hybrid_image -= hybrid_image.min()
    hybrid_image = hybrid_image * 255 / hybrid_image.max()
    new_hybrid = hybrid_image.astype(np.uint8)
    return new_hybrid


near = cv.imread('C:/Users/hossein rahmani/PycharmProjects/image_processing/ex2/res21-near.jpg')
far = cv.imread('C:/Users/hossein rahmani/PycharmProjects/image_processing/ex2/res22-far.jpg')

near_B = near[:, :, 0]
near_G = near[:, :, 1]
near_R = near[:, :, 2]
far_B = far[:, :, 0]
far_G = far[:, :, 1]
far_R = far[:, :, 2]
# near channels
x_near_B = np.fft.fft2(near_B)
y_near_B = np.fft.fftshift(x_near_B)
x_near_G = np.fft.fft2(near_G)
y_near_G = np.fft.fftshift(x_near_G)
x_near_R = np.fft.fft2(near_R)
y_near_R = np.fft.fftshift(x_near_R)
# res23-dft-near drawing

plt.subplot(1, 3, 1), plt.imshow(np.log(np.abs(y_near_B) + 1),cmap='gray')
plt.title('B Channel'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 2), plt.imshow(np.log(np.abs(y_near_G) + 1),cmap='gray')
plt.title('G Channel'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 3), plt.imshow(np.log(np.abs(y_near_R) + 1),cmap='gray')
plt.title('R Channel'), plt.xticks([]), plt.yticks([])
x_far_B = np.fft.fft2(far_B)
y_far_B = np.fft.fftshift(x_far_B)
x_far_G = np.fft.fft2(far_G)
y_far_G = np.fft.fftshift(x_far_G)
x_far_R = np.fft.fft2(far_R)
y_far_R = np.fft.fftshift(x_far_R)
# res24-dft-far drawing
plt.subplot(1, 3, 1), plt.imshow(np.log(np.abs(y_far_B) + 1), cmap='gray')
plt.title('B Channel'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 2), plt.imshow(np.log(np.abs(y_far_G) + 1), cmap='gray')
plt.title('G Channel'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 3), plt.imshow(np.log(np.abs(y_far_R) + 1), cmap='gray')
plt.title('R Channel'), plt.xticks([]), plt.yticks([])
# D0_near = find_cutoff(y_near_B)
# D0_far = find_cutoff(y_far_B)
D0_near = 20
D0_far = 50
G2 = guassian_lowpass(D0_near, y_near_B.shape)
H2 = 1 - G2  # highpass filter
plt.imshow(H2, cmap='gray')
plt.title('highpass filter'), plt.xticks([]), plt.yticks([])
G1 = guassian_lowpass(D0_far, y_far_B.shape)
plt.imshow(G1, cmap='gray')
plt.title('lowpass filter'), plt.xticks([]), plt.yticks([])
I2_B = y_near_B * H2
I2_G = y_near_G * H2
I2_R = y_near_R * H2
plt.subplot(1, 3, 1), plt.imshow(np.log(np.abs(I2_B) + 1), cmap='gray')
plt.title('B Channel highpass'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 2), plt.imshow(np.log(np.abs(I2_G) + 1), cmap='gray')
plt.title('G Channel highpass'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 3), plt.imshow(np.log(np.abs(I2_R) + 1), cmap='gray')
plt.title('R Channel highpass'), plt.xticks([]), plt.yticks([])

I1_B = y_far_B * G1
I1_G = y_far_G * G1
I1_R = y_far_R * G1
plt.subplot(1, 3, 1), plt.imshow(np.log(np.abs(I1_B) + 1), cmap='gray')
plt.title('B Channel'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 2), plt.imshow(np.log(np.abs(I1_G) + 1), cmap='gray')
plt.title('G Channel'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 3), plt.imshow(np.log(np.abs(I1_R) + 1), cmap='gray')
plt.title('R Channel'), plt.xticks([]), plt.yticks([])


I_B=compound(I1_B,I2_B)
I_G=compound(I1_G,I2_G)
I_R=compound(I1_R,I2_R)
plt.subplot(1, 3, 1), plt.imshow(np.log(np.abs(I_B) + 1), cmap='gray')
plt.title('B Channel '), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 2), plt.imshow(np.log(np.abs(I_G) + 1), cmap='gray')
plt.title('G Channel '), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 3), plt.imshow(np.log(np.abs(I_R) + 1), cmap='gray')
plt.title('R Channel '), plt.xticks([]), plt.yticks([])

B_inverse=inverse_transform(I_B)
G_inverse=inverse_transform(I_G)
R_inverse=inverse_transform(I_R)
I=np.zeros(near.shape)
I[:,:,0]=B_inverse
I[:,:,1]=G_inverse
I[:,:,2]=R_inverse
cv.imwrite('C:/Users/hossein rahmani/PycharmProjects/image_processing/ex2/res30-hybrid-near.jpg', I)
cv.imwrite('C:/Users/hossein rahmani/PycharmProjects/image_processing/ex2/res31-hybrid-far.jpg', my_resize(I,4,4))

