import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def vertical_minimize_cut(x):
    heigth = x.shape[0]
    width = x.shape[1]
    cut_cost = np.zeros((heigth, width, 3))  # first index for cost and second and third for next.
    cut_visualized = np.zeros((heigth, width), dtype=np.uint8)
    for i in range(0, heigth):
        for j in range(0, width):
            if i == 0:
                cut_cost[i, j, :] = np.array([x[i, j], -1, -1])
            else:
                if j == 0:
                    if x[i - 1, j] < x[i - 1, j + 1]:
                        cut_cost[i, j, :] = np.array([cut_cost[i - 1, j, 0] + x[i, j], i - 1, j])
                    else:
                        cut_cost[i, j, :] = np.array([cut_cost[i - 1, j + 1, 0] + x[i, j], i - 1, j + 1])
                elif j == width - 1:
                    if x[i - 1, j] < x[i - 1, j - 1]:
                        cut_cost[i, j, :] = np.array([cut_cost[i - 1, j, 0] + x[i, j], i - 1, j])
                    else:
                        cut_cost[i, j, :] = np.array([cut_cost[i - 1, j - 1, 0] + x[i, j], i - 1, j - 1])
                else:
                    if x[i - 1, j - 1] < x[i - 1, j] and x[i - 1, j - 1] < x[i - 1, j + 1]:
                        cut_cost[i, j, :] = np.array([cut_cost[i - 1, j - 1, 0] + x[i, j], i - 1, j - 1])
                    if x[i - 1, j] < x[i - 1, j - 1] and x[i - 1, j] < x[i - 1, j + 1]:
                        cut_cost[i, j, :] = np.array([cut_cost[i - 1, j, 0] + x[i, j], i - 1, j])
                    if x[i - 1, j + 1] < x[i - 1, j - 1] and x[i - 1, j + 1] < x[i - 1, j]:
                        cut_cost[i, j, :] = np.array([cut_cost[i - 1, j + 1, 0] + x[i, j], i - 1, j + 1])
    i = heigth - 1
    row = cut_cost[i, :, 0]  # cost
    current_column = np.where(row == np.amin(row))[0][0]
    cut_visualized[i, current_column] = 1
    i = i - 1
    while i >= 0:
        if current_column > 0 and cut_cost[i, current_column - 1, 0] <= cut_cost[i, current_column, 0] and (
                current_column == width - 1 or (
                cut_cost[i, current_column + 1, 0] >= cut_cost[i, current_column - 1, 0])):
            current_column = current_column - 1
            cut_visualized[i, current_column] = 1
            i = i - 1
        elif current_column < width - 1 and cut_cost[i, current_column + 1, 0] <= cut_cost[i, current_column, 0] and (
                current_column == 0 or (cut_cost[i, current_column + 1, 0] <= cut_cost[i, current_column - 1, 0])):
            current_column = current_column + 1
            cut_visualized[i, current_column] = 1
            i = i - 1
        else:  # hamoon sotoon mimoneh
            cut_visualized[i, current_column] = 1
            i = i - 1

    return cut_visualized


def vertical_minimize_mask(x):
    mask = vertical_minimize_cut(x)
    for i in range(mask.shape[0]):
        t = 0
        while mask[i, t] != 1:
            mask[i, t] = 1
            t = t + 1

    return mask


def similar_patch(texture, patch, mask=None):
    similar = cv.matchTemplate(texture, patch, cv.TM_CCORR_NORMED, mask=mask)
    cv.normalize(similar, similar, 0, 1, cv.NORM_MINMAX, -1)
    number = 10
    flattened_sorted_matches = np.sort(similar, axis=None)[-number:]
    rand_index = np.random.randint(0, number)
    random_value = flattened_sorted_matches[rand_index]
    location = np.where(similar == random_value)
    matchLoc = (location[0][0], location[1][0])  # now suitable for numpy indexes

    return matchLoc


def random_initiall_patch(patch_heigth, patch_width, texture):
    x = np.random.randint(0, texture.shape[0] - patch_heigth)
    y = np.random.randint(0, texture.shape[1] - patch_width)
    return texture[x:x + patch_heigth, y:y + patch_width, :]


def random_patch(texture, template, overlap, is_horrizental,
                 is_vertical):  # template ma inja sample_len tol va aarz dareh.
    mask = np.zeros((template.shape[0], template.shape[1]), dtype=np.uint8)
    if is_horrizental:
        mask[:overlap, :] = 255

    if is_vertical:
        mask[:, :overlap] = 255

    p = similar_patch(texture,template,mask)
    return texture[p[0]:p[0] + template.shape[0], p[1]:p[1] + template.shape[1], :]


def suitable_cut(sample1, sample2, overlap, is_horrizental, is_vertical):
    ''' return suitable mask for compounding two patches'''
    mask = np.zeros((sample2.shape[0], sample2.shape[1]), dtype=np.uint8)
    if is_horrizental:
        x1 = np.linalg.norm(sample1[:overlap, :, :] - sample2[:overlap, :, :],
                            axis=2)  # eucledian difference of RGB vectors
        mask[:overlap, :] += vertical_minimize_mask(x1.T).T
    if is_vertical:
        x2 = np.linalg.norm(sample1[:, :overlap, :] - sample2[:, :overlap, :],
                            axis=2)  # eucledian difference of RGB vectors
        mask[:, :overlap] += vertical_minimize_mask(x2)
    mask = np.where(mask > 1, 1, mask)  # har ja hadeaghal yeki yek bodeh yek gozashtam
    mask3 = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    mask3[:, :, 0] = mask
    mask3[:, :, 1] = mask
    mask3[:, :, 2] = mask
    mask3 = cv.GaussianBlur(mask3, (9, 9), 0)  # Dr. Kamali said it was feathering technique.
    return mask3


def synthesize(texture, given_len, sample_len, overlap_len):
    sample_number = np.ceil(given_len / (sample_len - overlap_len)).astype(np.int8)
    synthesized_len = sample_number * (sample_len - overlap_len) + overlap_len
    synthesized_texture = np.zeros((synthesized_len, synthesized_len, 3), dtype=np.uint8)
    for i in range(sample_number):
        for j in range(sample_number):
            if i == 0 and j == 0:
                synthesized_texture[:sample_len, :sample_len, :] = random_initiall_patch(sample_len, sample_len,
                                                                                         texture)  # chose a random patch for initiallization
            else:
                x_start = i * (sample_len - overlap_len)
                y_start = j * (sample_len - overlap_len)
                template = synthesized_texture[x_start:x_start + sample_len, y_start:y_start + sample_len,
                           :]  #  have  vertical and horrizental rectangles and others are zero.
                new_sample = random_patch(texture, template, overlap_len, i != 0, j != 0)
                mask = suitable_cut(template, new_sample, overlap_len, i != 0, j != 0)
                compound_sample = np.uint8((mask) * template + (1 - mask) * new_sample)
                synthesized_texture[x_start:x_start + sample_len, y_start:y_start + sample_len, :] = compound_sample

    synthesized_texture = synthesized_texture[:given_len, :given_len, :]  # cut it to 2500*2500
    return synthesized_texture


def save_synthesizedimage(texture, given_len, sample_len, overlap_len, adress, texture_name):
    synthesized = synthesize(texture, given_len, sample_len, overlap_len)
    rgb_synthesized = cv.cvtColor(synthesized, cv.COLOR_BGR2RGB)
    rgb_texture = cv.cvtColor(texture, cv.COLOR_BGR2RGB)
    plt.subplot(1, 2, 1), plt.imshow(rgb_texture)
    plt.subplot(1, 2, 2), plt.imshow(rgb_synthesized)
    plt.savefig(adress)  # To save figure
    cv.imwrite(texture_name, synthesized)
    return


# initiallizing four texture
given_len = 2500
anar = cv.imread('texture01.jpg')
sample_len1 = 300  # minimum heigth and width divided 2
overlap_len1 = 100
save_synthesizedimage(anar, given_len, sample_len1, overlap_len1, 'res11.jpg', 'anar.jpg')
sample_len2 = 210
overlap_len2 = 70  # hamishe overlap 1/3 sample_len hast.
golf = cv.imread('texture03.jpg')
save_synthesizedimage(golf, given_len, sample_len2, overlap_len2, 'res12.jpg', 'golf.jpg')
sample_len4 = 150
overlap_len4 = 50
rope = cv.imread('rope_texture.jpg')
save_synthesizedimage(rope, given_len, sample_len4, overlap_len4, 'res13.jpg', 'rope.jpg')
sample_len3 = 100
overlap_len3 = 30
bread = cv.imread('bread_texture.jpg')
save_synthesizedimage(bread, given_len, sample_len3, overlap_len3, 'res14.jpg', 'bread.jpg')
