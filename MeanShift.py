# mean_shift
import numpy as np
import cv2 as cv



def find_centers(x, Cluster_threshhold=10 ^ -1):
    cluster_ids = np.ones(x.shape[0]) * -1
    assigned_points = 0
    cluster_idx = 0  # index of next cluster center
    cluster_centers = []
    for i in range(x.shape[0]):
        if assigned_points == 0:  # first point
            cluster_ids[i] = cluster_idx
            cluster_centers.append(x[i])
            assigned_points += 1
            cluster_idx += 1
        else:
            for num, center in enumerate(cluster_centers):
                dist = np.linalg.norm(x[i] - center)
                if dist < Cluster_threshhold:
                    cluster_ids[i] = num
                    assigned_points += 1
                else:
                    cluster_ids[i] = cluster_idx
                    cluster_centers.append(x[i])
                    cluster_idx += 1
                    assigned_points += 1

    cluster_centers = np.array(cluster_centers)
    y = np.zeros((x.shape[0], x.shape[1]), dtype=x.dtype)
    for i in range(0, cluster_idx):
        index = np.where(cluster_ids == i)[0]
        cluster_centers[i, 0] = x[:, 0][index].mean()
        cluster_centers[i, 1] = x[:, 1][index].mean()
        cluster_centers[i, 2] = x[:, 2][index].mean()
        y[index, 0] = cluster_centers[i, 0]
        y[index, 1] = cluster_centers[i, 1]
        y[index, 2] = cluster_centers[i, 2]

    return y


def vector_2image(y, h, w):
    image = np.zeros((h, w, 3), dtype=y.dtype)
    for i in range(0, h):
        image[i] = y[i * w:(i + 1) * w]

    return image



def mean_shift(x, window=30, max_iter=30, MIN_DISTANCE=10 ^ -6):
    n = x.shape[0]
    max_distance = 1
    iteration = 0
    need_shift = np.ones(n, dtype=bool)
    while max_distance > MIN_DISTANCE and iteration <= max_iter:
        print("iteration ", iteration)
        is_shifted = np.zeros(n, dtype=bool)
        for i in range(0, n):
            max_distance = 0
            if not need_shift[i]:
                continue
            if is_shifted[i] == False:
                previous_center = x[i]
                indx = np.where(np.linalg.norm(x - previous_center, axis=1) <= window)[0]
                center = np.mean(x[indx], axis=0)
                dist = np.sqrt(np.sum((center - previous_center) ** 2))
                if dist > max_distance:
                    max_distance = dist
                if dist < MIN_DISTANCE:
                    need_shift[i] = False
                x[indx] = center
                is_shifted[indx] = True


        iteration += 1

    # postProcess
    # print("start of postprocess")
    # mean_y = find_centers(x, Cluster_threshhold)
    return x

def postProcess(x,window=25):
    n = x.shape[0]
    max_distance = 1
    iteration = 0
    is_shifted = np.zeros(n, dtype=bool)
    for i in range(0, n):
        max_distance = 0
        if is_shifted[i] == False:
            previous_center = x[i]
            B_x = x[:, 0] - previous_center[0]
            G_x = x[:, 1] - previous_center[1]
            R_x = x[:, 2] - previous_center[2]
            indx = np.where(np.linalg.norm(x - previous_center, axis=1) <= window)[0]
            center = np.mean(x[indx], axis=0)
            dist = np.sqrt(np.sum((center - previous_center) ** 2))
            if dist > max_distance:
                max_distance = dist
            x[indx] = center
            is_shifted[indx] = True
    return x

def my_resize(image, a, b):  # image tol , arzash a brabar o b barabar kochk mishe mishe
    width = int(image.shape[1] / a)
    height = int(image.shape[0] / b)
    resized_image = cv.resize(image, (width, height))
    return resized_image


park = cv.imread('park.jpg')
little_park = my_resize(park, 5, 5)
# lab_park = cv.cvtColor(park, cv.COLOR_BGR2LAB)
x = np.column_stack((little_park[:, :, 0].flatten(), little_park[:, :, 1].flatten(), little_park[:, :, 2].flatten()))
x_mean = mean_shift(x)
y_mean = postProcess(x_mean)
cartoonized_image = vector_2image(y_mean, little_park.shape[0], little_park.shape[1])
#cv.imwrite('res_park_postprocess.jpg', cartoonized_image)
cv.imwrite('res05_resized.jpg', cartoonized_image)
new_park=np.zeros((5*cartoonized_image.shape[0],5*cartoonized_image.shape[1],3),dtype=np.uint8)
for i in range(0,new_park.shape[0]):
    for j in range(0,new_park.shape[1]):
        new_park[i,j,:]=cartoonized_image[i//5, j//5]

cv.imwrite('res05.jpg', new_park)