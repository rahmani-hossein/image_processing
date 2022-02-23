import numpy as np
import cv2 as cv
import time
from skimage.segmentation import mark_boundaries
from skimage.color import label2rgb
import matplotlib.pyplot as plt
from scipy import stats




def gradient_magnitude(img):
    """gradient of gray space based on algorithm."""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sobel_x = np.array([[-1, 0, 1]], np.float32)
    sobel_y = np.array([[-1], [0], [+1]], np.float32)
    I_x = cv.filter2D(src=gray, ddepth=-1, kernel=sobel_x)
    I_y = cv.filter2D(src=gray, ddepth=-1, kernel=sobel_y)
    G = np.sqrt(I_x ** 2 + I_y ** 2)
    return G


def local_minimum_gradients(i, j, gradient):  # window=5
    x, y = np.unravel_index(np.argmin(gradient[i - 2:i + 3, j - 2:j + 3]), (5, 5))
    x = x + i - 2
    y = y + j - 2
    return x, y


def initial_centers(img, lab_image, k):
    h = img.shape[0]
    w = img.shape[1]
    S = int(np.sqrt((h * w) / k))
    gradient = gradient_magnitude(img)
    x = []
    y = []
    for i in range(int(S / 2), h, S):
        for j in range(int(S / 2), w, S):
            i_persurb, j_pursurb = local_minimum_gradients(i, j, gradient)
            x.append(i_persurb)
            y.append(j_pursurb)

    X = np.array(x)
    Y = np.array(y)
    L = lab_image[X, Y, 0]
    A = lab_image[X, Y, 1]
    B = lab_image[X, Y, 2]
    centers = np.column_stack((L, A, B, X, Y))

    return centers


def get_lab_features(image):
    h = image.shape[0]
    w = image.shape[1]
    LAB_image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    features = np.zeros((h, w, 5), dtype=np.float32)
    features[:, :, 0:3] = LAB_image
    for i in range(0, h):
        for j in range(0, w):
            features[i, j, 3:] = np.array([i, j])
    return features


def find_distance(pixels, center, alpha):
    l = pixels[:, :, 0] - center[0]
    a = pixels[:, :, 1] - center[1]
    b = pixels[:, :, 2] - center[2]
    x = pixels[:, :, 3] - center[3]
    y = pixels[:, :, 4] - center[4]
    lab_diff = np.linalg.norm(np.dstack((l, a, b)), axis=2)
    xy_diff = np.linalg.norm(np.dstack((x, y)), axis=2)
    return lab_diff + alpha * xy_diff


def slic_oversegmentation(image, k, alpha1=10, max_iteration=10, TOL=2):
    t = time.time()
    h = image.shape[0]
    w = image.shape[1]
    M = 35
    S = int(np.sqrt((h * w) / k))
    print(S)
    alpha = M / S
    distances = np.full((h, w), np.inf)
    labels = -1 * np.ones((h, w), dtype=np.int64)
    features = get_lab_features(image)
    LAB_image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    centers = initial_centers(image, LAB_image, k)
    cluster_num = centers.shape[0]
    print(time.time() - t)
    for iteration in range(0, max_iteration):
        t1 = time.time()
        for i in range(cluster_num):  # label=i
            cluster_x = centers[i, 3]
            cluster_y = centers[i, 4]
            # assignment
            # first crop suitable array
            x0 = np.maximum(int(cluster_x - S), 0)
            x1 = np.minimum(int(cluster_x + S), h)
            y0 = np.maximum(int(cluster_y - S), 0)
            y1 = np.minimum(int(cluster_y + S), w)
            pixels = features[x0:x1, y0:y1]
            distance = find_distance(pixels, centers[i], alpha)
            l = distances[x0:x1, y0:y1] - distance
            l_indx = np.where(l > 0)
            labels[l_indx[0] + x0, l_indx[1] + y0] = i
            distances[l_indx[0] + x0, l_indx[1] + y0] = distance[l_indx[0], l_indx[1]]
        # updating clusters
        error = 0
        for c in range(0, centers.shape[0]):
            group = np.where(labels == c)
            new_center = np.zeros(5)
            new_center[0] = (features[:, :, 0])[group].mean()
            new_center[1] = (features[:, :, 1])[group].mean()
            new_center[2] = (features[:, :, 2])[group].mean()
            new_center[3] = group[0].mean()
            new_center[4] = group[1].mean()
            error += np.sum(new_center - centers[i])
            centers[i] = new_center
        if error < TOL:
            # return labels
            print(iteration)
            break
        print(time.time() - t1)

    return labels, centers


# postprocess idea=Dr.Kamali
def enforce_coonectivity(labels_input, centers):
    ''' coordinates of pixel'''
    labels=labels_input.copy()
    n = np.max(labels)
    for i in range(0, n + 1):
        print("step ",i)
        mask = np.zeros(labels.shape, dtype=np.uint8)
        indx = np.where(labels == i)
        mask[indx] = 1
        num_labels, labels_im = cv.connectedComponents(mask)
        k = labels_im[centers[i, 0], centers[i, 1]]
        labels_im[labels_im == k] = 0
        print(labels_im.dtype)
        labels_im = labels_im.astype(np.uint8)
        dilated_mask = cv.dilate(labels_im, cv.getStructuringElement(cv.MORPH_RECT, (2, 2)),
                                 iterations=1)
        border = dilated_mask - labels_im
        print(border)
        for j in range(0, num_labels):
            if j != 0 and j != k:
                border_vales = border[np.where(border == j)]
                if len(border_vales) > 0:
                    border_indx = np.where(border == j)
                    x_superborder = np.mean(border_indx[0])
                    y_superborder = np.mean(border_indx[1])
                    mode = stats.mode(labels[border_indx])[0]
                    print(mode[0])
                    # extract = labels_input[int(x_superborder), int(y_superborder)]
                    # labels[labels_im == extract] = mode[0]
                    labels[labels_im==j]=mode[0]
    return labels


slic = cv.imread('slic.jpg')
labels, centers = slic_oversegmentation(slic, 64, alpha=0.08)
new_labels=enforce_coonectivity(labels,centers[:,3:])
image1 = mark_boundaries(slic, new_labels, color=(1, 0, 0)) * 255
cv.imwrite('res06.jpg', image1)

labels,centers = slic_oversegmentation(slic, 256)
rgb_slic=cv.cvtColor(slic,cv.COLOR_BGR2RGB)
image1 = mark_boundaries(rgb_slic, labels, color=(1,0,0)) * 255
cv.imwrite('res07.jpg', image1)

labels,centers = slic_oversegmentation(slic, 1024)
rgb_slic=cv.cvtColor(slic,cv.COLOR_BGR2RGB)
image1 = mark_boundaries(rgb_slic, labels, color=(1,0,0)) * 255
cv.imwrite('res08.jpg', image1)

labels,centers = slic_oversegmentation(slic, 2048)
rgb_slic=cv.cvtColor(slic,cv.COLOR_BGR2RGB)
image1 = mark_boundaries(rgb_slic, labels, color=(1,0,0)) * 255
cv.imwrite('res09.jpg', image1)