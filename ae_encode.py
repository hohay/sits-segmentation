# Copyright (C) 2021 hohay
# License: MIT
from PIL import Image
import os
import numpy as np

from ae_model import ImageDataset, Decoder, Encoder,

if __name__ == '__main__':
    W, H = 5000, 5000
    PATCH_SIZE, PATCH_SIZE_NDVI = 9, 5

    img_array = []
    ndvi_array = []
    dates_array = []

    data_path = "./ctrain3/"
    dir_list = os.listdir(data_path)

    dir_list = sorted(dir_list, key = lambda x : x) # sort by date

    def extend(band, patch_size):
      to_insert1 = np.flipud(band[1: int(patch_size / 2 + 1)])
      band = np.insert(band, 0, to_insert1, axis=0)
      to_insert2 = np.flipud(band[(len(band) - int(patch_size / 2) - 1):(len(band) - 1)])
      band = np.concatenate((band, to_insert2), axis=0)
      to_insert3 = band[:, range(int(patch_size / 2), 0, -1)]
      band = np.concatenate((to_insert3, band), axis=1)
      to_insert4 = band[:, range((len(band[0]) - 2), (len(band[0]) - int(patch_size / 2) - 2), -1)]
      band = np.concatenate((band, to_insert4), axis=1)
      return band

    bands = { 2, 3, 4, 8 }
    for dir in dir_list:
      #print(dir)
      dates_array.append(dir)
      img_set = []
      for band in bands:
        fil_path = data_path + dir + "/B0" + str(band) + ".png"
        im = Image.open(fil_path)
        a = np.asarray(im, float)
        a = a.clip(0, 20000) # saturation
        img_set.append(a)

      # ndvi

      ndvi = (img_set[3] - img_set[2]) / (img_set[3] + img_set[2])

      for i in range(4):
        img_set[i] = extend(img_set[i], PATCH_SIZE)

      ndvi = extend(ndvi, PATCH_SIZE_NDVI)

      img_set = np.asarray(img_set)
      ndvi = np.asarray(ndvi)

      img_array.append(img_set)
      ndvi_array.append(ndvi)

    img_array = np.asarray(img_array)
    ndvi_array = np.asarray(ndvi_array)

    list_norm = [[10.0, 20000.0], [10.0, 20000.0], [10.0, 20000.0], [0.0, 20000.0]]

    for i in range(len(img_array)):
      for band in range(4):
        img_array[i][band] = (img_array[i][band] - list_norm[band][0]) / (list_norm[band][1] - list_norm[band][0])

    list_norm = [[0.8625083820195137, 0.22279758208456227],
                 [0.610084656078228, 0.23959566075992625],
                 [0.5624912535687514, 0.2685569045705318],
                 [0.5110898182815605, 0.30644996439284977]]

    ndvi_img = ndvi_array[:,:,:].flatten()
    min = -1.0
    max = 0.996078431372549

    ndvi_array = (ndvi_array - min) / (max - min)

    ndvi_img = ndvi_array[:,:,:].flatten()
    mean = 0.4610412515786927
    std = 0.047248814801636244
    list_norm_ndvi = [[mean, std]]

    # min-max
    #list_norm = [[10.0, 20000.0], [10.0, 20000.0], [10.0, 20000.0], [0.0, 20000.0]]
    # mean-std
    #list_norm = [[0.8625083820195137, 0.22279758208456227],
    #             [0.610084656078228, 0.23959566075992625],
    #             [0.5624912535687514, 0.2685569045705318],
    #             [0.5110898182815605, 0.30644996439284977]]
    # min-max
    # -1.0, 0.996078431372549
    # mean, std
    # 0.4610412515786927, 0.047248814801636244

    # normalize images
    #list_norm = []
    #for band in range(4):
    #  all_img = img_array[:, band, :, :].flatten()
    #  min = np.min(all_img)
    #  max = np.max(all_img)
    #  list_norm.append([min, max])

    #list_norm = [[0.8625083820195137, 0.22279758208456227],
    #             [0.610084656078228, 0.23959566075992625],
    #             [0.5624912535687514, 0.2685569045705318],
    #             [0.5110898182815605, 0.30644996439284977]]

    #for i in range(len(img_array)):
    #  for band in range(4):
    #    img_array[i][band] = (img_array[i][band] - list_norm[band][0]) / (list_norm[band][1] - list_norm[band][0])

    #list_norm = []
    #for band in range(4):
    #  all_img = img_array[:, band, :, :].flatten()
    #  mean = np.mean(all_img)
    #  std = np.std(all_img)
    #  list_norm.append([mean, std])

    #ndvi_img = ndvi_array[:,:,:].flatten()
    #min = np.min(ndvi_img)
    #max = np.max(ndvi_img)



    #ndvi_array = (ndvi_array - min) / (max - min)

    #ndvi_img = ndvi_array[:,:,:].flatten()
    #mean = np.mean(ndvi_img)
    #std = np.std(ndvi_img)
    #list_norm_ndvi = [[mean, std]]

    ndvi_array2 = []
    for date in ndvi_array:
      ndvi_array2.append([date])
    ndvi_array2 = np.asarray(ndvi_array2)
    img_train = np.transpose(img_array, (1, 0, 2, 3))
    ndvi_train = np.transpose(ndvi_array2, (1, 0, 2, 3))


    print(img_train.shape)
    print(ndvi_train.shape)


    from torch.utils.data import DataLoader

    gpu = True

    fname = "ae-model_ep_2_loss_0.0004303test.pkl" #"ae-model_ep_2_loss_0.0005411test.pkl"
    tload = torch.load(fname)
    encoder, decoder = tload
    if gpu:
      encoder = encoder.cuda()
      decoder = decoder.cuda()

    SOX = 0
    SOY = 0
    SW = 5000
    SH = 5000
    patches = SW*SH
    image2 = ImageDataset(img_train, ndvi_train, PATCH_SIZE, PATCH_SIZE_NDVI, range(patches), SOX, SOY, SW, SH)

    batch_size2 = 100
    shuffle2 = False
    if gpu:
      loader2 = DataLoader(image2,
                          batch_size=batch_size2,
                          shuffle=shuffle2,
                          num_workers=1,# 1,  # 1 for CUDA
                          pin_memory=True,  # CUDA only
                          drop_last=False
                          )
    else:
      loader2 = DataLoader(image2,
                          batch_size=batch_size2,
                          shuffle=shuffle2,
                          num_workers=0,  # 1 for CUDA
                          drop_last=False
                          )


    encoder.eval()
    encoded_array = None
    for batch_idx, (data_or, data_ndvi, id) in enumerate(loader2):
      if gpu:
        # data = data.cuda(async=True)
        data_or = data_or.cuda()
        data_ndvi = data_ndvi.cuda()
      encoded, _ = encoder(Variable(data_or), Variable(data_ndvi))
      if (batch_idx + 1) % 300 == 0:
        print('Encoding: {}/{} ({:.0f}%)'.format(
          (batch_idx + 1) * batch_size2, len(loader2) * batch_size2,
            100. * (batch_idx + 1) / len(loader2)))
      #encoded = encoded.cpu().detach().numpy()
      if encoded_array is not None:
        encoded_array = np.concatenate((encoded_array, encoded.cpu().detach().numpy()), 0)
      else:
        encoded_array = encoded.cpu().detach().numpy()


    encoded_norm = []
    features_n = 4
    for band in range(features_n):
      min = np.min(encoded_array[:, band])
      max = np.max(encoded_array[:, band])
      encoded_norm.append([min, max])
    for band in range(features_n):
      encoded_array[:, band] = 255 * (encoded_array[:, band] - encoded_norm[band][0]) / (
                encoded_norm[band][1] - encoded_norm[band][0])
    print(encoded_array.shape)

    new_encoded_array = np.transpose(encoded_array, (1, 0))
    new_encoded_array = new_encoded_array.reshape(features_n, SW, SH)


    for band in range(features_n):
      im = Image.fromarray(np.uint8(new_encoded_array[band]), mode="L")
      im.save("encoded_" + str(band) + ".png", "PNG")
