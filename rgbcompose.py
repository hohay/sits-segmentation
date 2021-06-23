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
    r = Image.open(dir_path+"B04.jp2")
    g = Image.open(dir_path+"B03.jp2")
    b = Image.open(dir_path+"B02.jp2")

    ra = np.float32(np.array(r))
    ga = np.float32(np.array(g))
    ba = np.float32(np.array(b))

    factor = 12 * 256 / 65535.0
    ra *= factor
    ga *=factor
    ba *= factor

    ra = np.clip(ra, 0, 255)
    ga = np.clip(ga, 0, 255)
    ba = np.clip(ba, 0, 255)

    t = np.array([ra, ga, ba])

    t = np.transpose(t, (1, 2, 0))

    nim = Image.fromarray(np.uint8(t), mode="RGB")

    dir_path_crop = data_out_path + dir + "\\rgb.png"
    nim.save(dir_path_crop, format="png")

    print('poke')
