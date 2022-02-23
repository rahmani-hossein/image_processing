import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max


def accomulator(binary_image, theta_values):  # coordinates from up_west coordinates.
    height = binary_image.shape[0]
    width = binary_image.shape[1]
    diagonal = np.sqrt(height ** 2 + width ** 2)
    r = diagonal
    thetas = np.arange(0, np.pi, step=np.pi / theta_values)
    rhos = np.arange(0, r, step=1)
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    accumulator_matrix = np.zeros((rhos.size, thetas.size))
    indexes = np.where(binary_image > 0)
    for p in range(0, indexes[0].size):
        x = indexes[0][p]
        y = indexes[1][p]
        for t in range(thetas.size):
            rho = x * cos_thetas[t] + y * sin_thetas[t]
            r_index = rhos.size * (1.0 * rho) / r
            r_index = np.ceil(r_index).astype(np.int64)
            accumulator_matrix[r_index][t] += 1
    return accumulator_matrix, rhos, cos_thetas, sin_thetas


def draw_lines(image, accumulator_matrix, rhos, cos_thetas, sin_thetas, threshhold=160,
               dist=10):
    height = image.shape[0]
    width = image.shape[1]
    sd = peak_local_max(accumulator_matrix, min_distance=dist, threshold_abs=threshhold)
    print(sd.shape)
    for p in range(sd.shape[0]):
        i = sd[p, 0]  # y ,x ro bara opencv jabeja mikonim
        j = sd[p, 1]
        rho = rhos[i]
        x1, y1, x2, y2 = find_2points(rho, cos_thetas, sin_thetas, j, height, width)
        cv.line(image, (y1, x1), (y2, x2), (0, 0, 255), thickness=3)
    return image


def find_2points(rho, cos_thetas, sin_thetas, j, height, width):
    # 6 halat
    x1=0
    x2=0
    y1=0
    y2=0
    x0_min = 0
    y0_max = width
    y0_min = findy_car(x0_min, cos_thetas[j], sin_thetas[j], rho)
    x0_max = findx_car(y0_max, cos_thetas[j], sin_thetas[j], rho)
    if 0 <= y0_min <= width and 0 <= x0_max <= height:
        x1 = x0_min
        y2 = y0_max
        y1 = y0_min
        x2 = x0_max
    else:
        x0_min = 0
        y0_max = 0
        y0_min = findy_car(x0_min, cos_thetas[j], sin_thetas[j], rho)
        x0_max = findx_car(y0_max, cos_thetas[j], sin_thetas[j], rho)
        if 0 <= y0_min <= width and 0 <= x0_max <= height:
            x1 = x0_min
            y2 = y0_max
            y1 = y0_min
            x2 = x0_max
        else:
            x0_min = height
            y0_max = width
            y0_min = findy_car(x0_min, cos_thetas[j], sin_thetas[j], rho)
            x0_max = findx_car(y0_max, cos_thetas[j], sin_thetas[j], rho)
            if 0 <= y0_min <= width and 0 <= x0_max <= height:
                x1 = x0_min
                y2 = y0_max
                y1 = y0_min
                x2 = x0_max
            else:
                x0_min = height
                y0_max = 0
                y0_min = findy_car(x0_min, cos_thetas[j], sin_thetas[j], rho)
                x0_max = findx_car(y0_max, cos_thetas[j], sin_thetas[j], rho)
                if 0 <= y0_min <= width and 0 <= x0_max <= height:
                    x1 = x0_min
                    y2 = y0_max
                    y1 = y0_min
                    x2 = x0_max
                else:
                    x0_min = 0
                    x0_max = height
                    y0_min = findy_car(x0_min, cos_thetas[j], sin_thetas[j], rho)
                    y0_max = findy_car(x0_max, cos_thetas[j], sin_thetas[j], rho)
                    if 0 <= y0_min <= width and 0 <= x0_max <= height:
                        x1 = x0_min
                        y2 = y0_max
                        y1 = y0_min
                        x2 = x0_max
                    else:
                        y0_min = 0
                        y0_max = width
                        x0_min = findx_car(y0_min, cos_thetas[j], sin_thetas[j], rho)
                        x0_max = findx_car(y0_max, cos_thetas[j], sin_thetas[j], rho)
                        if 0 <= y0_min <= width and 0 <= x0_max <= height:
                            x1 = x0_min
                            y2 = y0_max
                            y1 = y0_min
                            x2 = x0_max
                        else:
                            print('yoho2')
                            print(x0_min, y0_min, x0_max, y0_max)
    return x1, y1, x2, y2


def findy_car(x, costheta, sintheta, rho):
    a = (-costheta / sintheta)
    b = rho / sintheta
    return int(a * x + b)


def findx_car(y, costheta, sintheta, rho):
    a = (-sintheta / costheta)
    b = rho / costheta
    return int(a * y + b)


def draw_corners(image, accumulator_matrix, rhos, cos_thetas, sin_thetas, threshhold=160, dist=10):
    height = image.shape[0]
    width = image.shape[1]
    point = np.zeros((9, 9, 3), dtype=np.uint8)
    point[:, :, 1] = np.ones((9, 9), dtype=np.uint8) * 255  # green square
    sd = peak_local_max(accumulator_matrix, min_distance=dist, threshold_abs=threshhold)
    print(sd.shape)
    for i in range(sd.shape[0] - 1):
        for j in range(i + 1, sd.shape[0]):
            A = np.zeros((2, 2), dtype=np.float64)
            b = np.zeros((2, 1), dtype=np.float64)
            rho1 = rhos[sd[i, 0]]
            rho2 = rhos[sd[j, 0]]
            b[0, 0] = rho1
            b[1, 0] = rho2
            A[0, 0] = cos_thetas[sd[i, 1]]
            A[0, 1] = sin_thetas[sd[i, 1]]
            A[1, 0] = cos_thetas[sd[j, 1]]
            A[1, 1] = sin_thetas[sd[j, 1]]  # Ax=b
            if np.linalg.matrix_rank(A) == 2:
                x, y = np.linalg.lstsq(A, b, rcond=None)[0]  # least squares solution
                x = int(x)
                y = int(y)
                if 0 <= x <= height and 0 <= y <= width:
                    image[x - 4:x + 5, y - 4:y + 5, :] = point
                    print('hhhhh')
            else:
                print('singular matrix so this intersection isnt our solution.')
    return image



img1 = cv.imread('im01.jpg')
edges1 = cv.Canny(img1, 200, 200)
plt.imsave('res01.jpg', edges1, cmap='gray')
accomulator1, rhos1, cos_thetas1, sin_thetas1 = accomulator(edges1, 180)
plt.imsave('res03-hough-space.jpg', accomulator1)
lined_image1 = draw_lines(img1.copy(), accomulator1, rhos1, cos_thetas1, sin_thetas1, 120, 10)
cv.imwrite('res05-lines.jpg', lined_image1)

lined_chess1 = draw_lines(img1.copy(), accomulator1, rhos1, cos_thetas1, sin_thetas1, 120, 10)
cv.imwrite('res07-chess.jpg', lined_chess1)
cornered_image1 = draw_corners(lined_chess1.copy(), accomulator1, rhos1, cos_thetas1, sin_thetas1, 120, 10)
cv.imwrite('res09-corners.jpg', cornered_image1)

# second image
img2 = cv.imread('im02.jpg')
edges2 = cv.Canny(img2, 200, 200)
plt.imsave('res02.jpg', edges2, cmap='gray')
accomulator2, rhos2, cos_thetas2, sin_thetas2 = accomulator(edges2, 180)
plt.imsave('res04-hough-space.jpg', accomulator2)
lined_image2 = draw_lines(img2.copy(), accomulator2, rhos2, cos_thetas2, sin_thetas2, 100, 15)
cv.imwrite('res06-lines.jpg', lined_image2)
lined_chess2 = draw_lines(img2.copy(), accomulator2, rhos2, cos_thetas2, sin_thetas2, 100, 15)
cv.imwrite('res08-chess.jpg', lined_chess2)
cornered_image2 = draw_corners(lined_chess2.copy(), accomulator2, rhos2, cos_thetas2, sin_thetas2, 100, 15)
cv.imwrite('res10-corners.jpg', cornered_image2)
