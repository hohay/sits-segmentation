# Copyright (C) 2021 hohay
# License: MIT
start_img = 0 # 4 for last pic, 0 for AE
features_n = 8
crd_scale = 16 # 32 for last pic, 16 for AE

W, H = 1000, 1000
TW, TH = 10000, 10000

import cv2
import os
import numpy as np

img_data = []

for i in range(start_img, features_n):
  im = cv2.imread("concat_{}.png".format(i))
  res = cv2.resize(im, dsize=(W, H), interpolation=cv2.INTER_CUBIC)
  a = np.array(res).flatten()
  a = a[::3]
  img_data.append(a)
  
x_data = [0] * (W*H)
y_data = [0] * (W*H)
for i in range(W):
  for j in range(H):
    y_data[j*W+i] = i/crd_scale
    x_data[j*W+i] = j/crd_scale
    
img_data = img_data[:features_n]
img_data.append(x_data)
img_data.append(y_data)

img_data = np.array(img_data)

img_data = np.transpose(img_data, (1, 0))

from sklearn.cluster import KMeans
import numpy as np

cls_n = 200

clustering = KMeans(n_clusters=cls_n, random_state=55).fit(img_data)

import pickle
with open('clustering.data', 'wb') as f:
  pickle.dump(clustering, f)
