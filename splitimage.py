# Copyright (C) 2021 hohay
# License: MIT
from PIL import Image
import os.path
import numpy as np


data_path = "C:\\thesis\\ndataset\\src\\"
data_out_path = "C:\\thesis\\ndataset\\ctrain\\"
data_dirs = os.listdir(data_path)

Image.MAX_IMAGE_PIXELS = 120560400

OX = 5000# 2500 # Train: 0
OY = 5000#1250 # Train: 5000
W = 5000
H = 5000

for dir in data_dirs:
    dir_path = data_path+dir+"\\"
    img_list = os.listdir(dir_path)
    for fil in img_list:
        fil_path = dir_path+fil
        f, e = os.path.splitext(fil_path)
        if e != ".jp2":
            continue
        dir_path_crop = data_out_path+dir+"\\"
        if not os.path.isdir(dir_path_crop):
            os.mkdir(dir_path_crop)
        fil_path_crop = dir_path_crop+fil
        f, e = os.path.splitext(fil_path_crop)
        fil_path_crop = f + ".png"
        im = Image.open(fil_path)
        im2 = im.crop((OX, OY, OX+W, OY+H))
        a = np.asarray(im2, float)
        a *= 5
        a = a.clip(0, 65535)
        im3 = Image.fromarray(np.uint16(a))
        im3.save(fil_path_crop, "PNG")
        print(fil_path_crop)