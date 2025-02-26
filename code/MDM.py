
from init import guidedfilter
import time
from PIL import Image
# import cv2 as cv
import matplotlib.image as mpimg  # mpimg 用于读取图片
import scipy.misc as misc
# import cv2
import numpy as np
def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
img = Image.open(r'E:\Dataset\results_1\epoch_1_index_3.png')
Img = img.convert('L')
Img.save("test1.jpg")
image1 = mpimg.imread(r'E:\Project\EvaluationCode\IRVI\RoadSences\gray_IR\IR39.jpg')            #改动
image2 = mpimg.imread(r'E:\Project\EvaluationCode\IRVI\RoadSences\gray_VIS\VIS39.jpg');          #改动
#guid = mpimg.imread(r'E:\Dataset\EDAT\20.png');
guid = mpimg.imread(r'E:\Project\NewProject\ResNetFusion-master\test1.jpg');
#guid = Img
#guid = mpimg.imread(r'E:\Dataset\results_1\');
#guid = mpimg.imread(r'E:\Project\NewProject\ResNetFusion-master\test3.jpg');
#guid = rgb2gray(guid)
#cv2.imshow('img-Canny', cv2.Canny(guid, 80 , 150))
#guid =  cv2.Canny(guid, 80 , 150)
#misc.imsave('guid.jpg', guid)
#guide1 = image1 * guid
#guide2 = image2 * guid
start = time.time()
new = guidedfilter((guid / 255.0),(image1 / 255.0),5, 0.01)               #改动
#new = guidedfilter(guid , image1 , 10, 0.01)
new = (new - np.min(new)) / (np.max(new) - np.min(new))
misc.imsave('final_fusion1.jpg', new)
new1 = guidedfilter((guid / 255.0), (image2/ 255.0) ,5, 0.01)              #改动
#new1 = guidedfilter(guid, image2 , 10, 0.01)
new1 = (new1 - np.min(new1)) / (np.max(new1) - np.min(new1))
misc.imsave('final_fusion2.jpg', new1)

#new2 =  mpimg.imread(r'E:\Dataset\results_1\epoch_1_index_31.png');
#new2 = rgb2gray(new2)
#new2 = (new2 - np.min(new2)) / (np.max(new2) - np.min(new2))
#misc.imsave('final_fusion2.jpg', new2)
fusion = np.add(np.multiply(image1, new), np.multiply(image2, 1 - new))
fusion1 = np.add(np.multiply(image1, new1), np.multiply(image2, 1 - new1))
misc.imsave(r'E:\Dataset\results/r1.png', fusion)
misc.imsave(r'E:\Dataset\results/r2.png', fusion1)
fusion = 0.5 * fusion + 0.5 * fusion1
end = time.time()
print(end - start)
misc.imsave('AAAAA.png', fusion)                                            #改动



