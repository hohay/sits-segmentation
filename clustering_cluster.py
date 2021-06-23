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

timg_data = []

for i in range(start_img,features_n):
  im = cv2.imread("concat_{}.png".format(i))
  res = cv2.resize(im, dsize=(TW, TH), interpolation=cv2.INTER_CUBIC)
  a = np.array(res).flatten()
  a = a[::3]
  timg_data.append(a)
  
x_data = [0] * (TW*TH)
y_data = [0] * (TW*TH)
for i in range(TW):
  for j in range(TH):
    y_data[j*TW+i] = i*(W/TW)/crd_scale
    x_data[j*TW+i] = j*(H/TH)/crd_scale
    
timg_data.append(x_data)
timg_data.append(y_data)

timg_data = np.array(timg_data)

timg_data = np.transpose(timg_data, (1, 0))

from sklearn.cluster import KMeans
import numpy as np
import pickle
with open('clustering.data', 'rb') as f:
  clustering = pickle.load(f)
  
cls = clustering.predict(timg_data)
cls_rgb = np.zeros((cls_n, 3))
cls_cnt = np.zeros((cls_n))

for i, c in zip(range(TW*TH), cls):
  cls_rgb[c] += timg_data[i][(4-start_img):(4-start_img+3)]
  cls_cnt[c] += 1

for i in range(cls_n):
  if cls_cnt[i] == 0:
    continue
  cls_rgb[i] /= cls_cnt[i]
  
rgb_md = [5, 5, 5]

borders = False

rgb_im = np.zeros((TW * TH, 3))

for i in range(TW*TH):
  rgb = cls_rgb[cls[i]]
  r = rgb[2] * rgb_md[0]
  g = rgb[1] * rgb_md[1]
  b = rgb[0] * rgb_md[2]
  if borders:
    c1 = cls[i]
    i1 = i % TW
    i2 = i // TW
    isb = False
    if i1 > 0 and cls[i-1] != c1:
      isb = True
    elif i2 > 0 and cls[i-TW] != c1:
      isb = True
    #elif i1 < TW - 1 and cls[i+1] != c1:
    #  isb = True
    #elif i2 < TH - 1 and cls[i+TW] != c1:
    #  isb = True
    if isb:
      r = 255
      g = b = 30

  rgb_im[i] = np.array([r, g, b])
  rgb_im[i] = np.clip(rgb_im[i], 0, 255)
rgb_im = rgb_im.reshape((TW,TH,3))

rgb_im_tg = np.zeros((TW * TH, 3))

for i in range(TW*TH):
  rgb = timg_data[i,(4-start_img):(7-start_img)]
  r = rgb[2] * rgb_md[0]
  g = rgb[1] * rgb_md[1]
  b = rgb[0] * rgb_md[2]
  rgb_im_tg[i] = np.array([r, g, b])
  rgb_im_tg[i] = np.clip(rgb_im_tg[i], 0, 255)

rgb_im_tg = rgb_im_tg.reshape((TW,TH,3))

from PIL import Image
im = Image.fromarray(np.uint8(rgb_im), mode='RGB')
im.save('testclustering.png')

im = Image.fromarray(np.uint8(rgb_im_tg), mode='RGB')
im.save('testtr.png')
