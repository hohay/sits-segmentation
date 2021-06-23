# Copyright (C) 2021 hohay
# License: MIT
from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = 120560400

col_corr = [
    [ 0.9, 1.0, 0.9, 1.0 ],
    [ 1.0, 1.0, 1.0, 1.0 ],
    [ 1.0, 1.0, 1.0, 0.9 ],
    [ 0.95, 1.0, 0.95, 1.0 ],
    [ 1.0, 1.0, 1.0, 1.0 ],
    [ 1.0, 1.0, 1.0, 1.0 ],
    [ 1.0, 1.0, 1.0, 1.0 ],
    [ 1.0, 1.0, 1.0, 1.0 ],
]

for i in range(4, 8):
    im1 = Image.open("slice0data/encoded_{}.png".format(i))
    im2 = Image.open("slice1data/encoded_{}.png".format(i))
    im3 = Image.open("slice2data/encoded_{}.png".format(i))
    im4 = Image.open("slice3data/encoded_{}.png".format(i))

    a1 = np.asarray(im1, float)
    a1 *= col_corr[i][0]
    a2 = np.asarray(im2, float)
    a2 *= col_corr[i][1]
    a3 = np.asarray(im3, float)
    a3 *= col_corr[i][2]
    a4 = np.asarray(im4, float)
    a4 *= col_corr[i][3]

    ta1 = np.concatenate((a1, a2), axis=1)
    ta2 = np.concatenate((a3, a4), axis=1)
    oa = np.concatenate((ta1, ta2), axis=0)

    if i < 4:
        oim = Image.fromarray(np.uint8(oa), mode='L')
    else:
        oim = Image.fromarray(np.uint16(oa))

    oim.save("concat_{}.png".format(i), format="png")

