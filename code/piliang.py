
import os
import numpy as np
from PIL import Image
path = 'E:/Dataset/ir'
count = 1
for file in os.listdir(path):
    os.rename(os.path.join(path,file),os.path.join(path,str(count)+".png"))
    count+=1
