import numpy as np
import cv2 as cv
from scipy.spatial import Delaunay
import imageio
import matplotlib.pyplot as plt


def apply_transformation(source, srcTriangle, dstTriangle, size):
    """
    GOAL= apply affine function(which will be found by triangles) to the rectangle which has the source triangle.
    :param source: source image
    :param srcTriangle: source triangle
    :param dstTriangle: destination triangle
    :param size: size of the output. (like opencv because I used opencv functions.)
    :return: output image which was transformed by an Affine transformation.
    """
    Transformation = cv.getAffineTransform(np.float32(srcTriangle), np.float32(dstTriangle))
    destination = cv.warpAffine(source, Transformation, (size[0], size[1]), None, flags=cv.INTER_LINEAR,
                                borderMode=cv.BORDER_REFLECT_101)  # tabeh amadeh opencv
    return destination


def morphTriangle(img1, img2, img, t1, t2, t, alpha):
    """
    morph a triangle
    :param img1: first image
    :param img2: second image
    :param img: output
    :param t1: first triangle
    :param t2: second triangle
    :param t: output triangle
    :param alpha: mid frame alpha
    :return: img
    """
    r1 = cv.boundingRect(np.float32([t1]))
    r2 = cv.boundingRect(np.float32([t2]))
    r = cv.boundingRect(np.float32([t]))

    t1Rect = []
    t2Rect = []
    tRect = []
    # nesbi coordinates
    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))


    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = apply_transformation(img1Rect, t1Rect, tRect, size)
    warpImage2 = apply_transformation(img2Rect, t2Rect, tRect, size)

    # Alpha blend
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + imgRect[:r[3],
                                                                                                     :r[2]] * mask


# def morphTriangle2(img1, img2, img, t1, t2, t, alpha):
#    # Get mask by filling triangle
#    mask = np.zeros(img1.shape, dtype=np.float32)
#    cv.fillConvexPoly(mask, np.int32(t), (1.0, 1.0, 1.0), 16, 0)
#    size = (img1.shape[0], img1.shape[1])
#    print(size)
#    warpImage1 = apply_transformation(img1, t1, t, size)
#    warpImage2 = apply_transformation(img2, t2, t, size)
#
#    # Alpha blend rectangular patches
#    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2
#
#    # Copy triangular region of the rectangular patch to the output image
#    img= img * (1 - mask) + imgRect * mask

def readPoints(path):
    """
    loading facial landmark points from apath
    :param path: path of file
    :return: 2d-matrix (n*2) in opencv coordinates.
    """
    X = []
    Y = []
    with open(path, 'r') as f:
        for line in f:
            x, y = line.split()
            X.append(int(x))
            Y.append(int(y))
    X = np.array(X)
    Y = np.array(Y)
    points = np.zeros((X.size, 2), np.int32)
    points[:, 0] = X
    points[:, 1] = Y
    return points


def morph(img1, points1, img2, points2, alpha=0.5):
    points = np.zeros(points1.shape)
    for i in range(0, points1.shape[0]):
        points[i, 0] = (1 - alpha) * points1[i, 0] + alpha * points2[i, 0]
        points[i, 1] = (1 - alpha) * points1[i, 1] + alpha * points2[i, 1]

    morphed_image = np.zeros(img1.shape, dtype=img1.dtype)
    tri = Delaunay(points1)
    for i in range(0, tri.simplices.shape[0]):
        x = tri.simplices[i, 0]
        y = tri.simplices[i, 1]
        z = tri.simplices[i, 2]
        t1 = [points1[x], points1[y], points1[z]]
        t2 = [points2[x], points2[y], points2[z]]
        t = [points[x], points[y], points[z]]
        morphTriangle(img1, img2, morphed_image, t1, t2, t, alpha)
    return morphed_image


angelina = cv.imread('res01.jpg')
angelina_withcolumns = np.ones((angelina.shape[0] + 2, angelina.shape[1] + 2, 3))
angelina_withcolumns[1:-1, 1:-1, :] = angelina
angelina = np.float32(angelina_withcolumns)
points1 = readPoints('anjelina.txt')
brad = cv.imread('res02.jpg')
brad_withcolumns = np.ones((brad.shape[0] + 2, brad.shape[1] + 2, 3))
brad_withcolumns[1:-1, 1:-1, :] = brad
brad = np.float32(brad_withcolumns)
points2 = readPoints('brad.txt')
images1 = []
images2 = []
for i in range(0, 45):
    alpha = i / 44
    morphed_image = morph(angelina, points1, brad, points2, alpha=alpha)
    morphed_image = morphed_image.astype(np.uint8)
    RGB_morphed = cv.cvtColor(morphed_image, cv.COLOR_BGR2RGB)
    images1.append(RGB_morphed)
    images2.append(RGB_morphed)
    if i == 15:
        cv.imwrite('res03.jpg', morphed_image)
    if i == 30:
        cv.imwrite('res04.jpg', morphed_image)
images2.reverse()
images1.extend(images2)
imageio.mimsave('morph.mp4', images1, fps=30)
