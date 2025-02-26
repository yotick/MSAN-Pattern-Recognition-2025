import cv2
import numpy as np
import scipy.misc as misc
moon = cv2.imread(r"E:\Project\EvaluationCode\IRVI\IR20.png", 0)
moon1 = cv2.imread(r"E:\Project\EvaluationCode\IRVI\VIS20.png", 0)
row, column = moon.shape
moon_f = np.copy(moon)
moon_f = moon_f.astype("float")

moon1_f = np.copy(moon1)
moon1_f = moon_f.astype("float")

gradient = np.zeros((row, column))
gradient1 = np.zeros((row, column))
for x in range(row - 1):
    for y in range(column - 1):
        gx = abs(moon_f[x + 1, y] - moon_f[x, y])
        gy = abs(moon_f[x, y + 1] - moon_f[x, y])
        gradient[x, y] = gx + gy

for x in range(row - 1):
    for y in range(column - 1):
        gx1 = abs(moon1_f[x + 1, y] - moon1_f[x, y])
        gy1 = abs(moon1_f[x, y + 1] - moon1_f[x, y])
        gradient1[x, y] = gx1 + gy1

sharp = moon_f + gradient
sharp = np.where(sharp < 0, 0, np.where(sharp > 255, 255, sharp))
gradient = gradient + gradient1
gradient = gradient.astype("uint8")
sharp = sharp.astype("uint8")
cv2.imshow("moon", moon)
cv2.imshow("gradient", gradient)
misc.imsave('gradient20.png', gradient)
cv2.imshow("sharp", sharp)
cv2.waitKey()