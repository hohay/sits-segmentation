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


    patches = H*W
    image = ImageDataset(img_train, ndvi_train, PATCH_SIZE, PATCH_SIZE_NDVI, range(patches), 0, 0, W, H)

    from torch.utils.data import DataLoader

    gpu = True
    
    bands_n = 4
    temporal_dim = 4
    features_n = 4
    original_layers = [32, 32, 64, 64]
    ndvi_layers = [16, 16, True]
    noise_factor = 0.25
    noise_factor_ndvi = None
    encoder = Encoder(bands_n, PATCH_SIZE, PATCH_SIZE_NDVI, features_n, temporal_dim, original_layers, ndvi_layers, np.asarray(list_norm), np.asarray(list_norm_ndvi), noise_factor, noise_factor_ndvi) # On CPU
    decoder = Decoder(bands_n, PATCH_SIZE, PATCH_SIZE_NDVI, features_n, temporal_dim, original_layers, ndvi_layers) # On CPU

    if gpu:
      encoder = encoder.cuda()  # On GPU
      decoder = decoder.cuda()  # On GPU

    from torch.utils.data import DataLoader

    batch_size = 150
    shuffle = True
    if gpu:
      loader = DataLoader(image,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=1,  # 1 for CUDA
                          pin_memory=True,  # CUDA only
                          drop_last=False
                          )
    else:
      loader = DataLoader(image,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=0,  # 1 for CUDA
                          drop_last=False
                          )
                          
    lr = 0.0001
    epochs = 2 # 2
    optimizer = torch.optim.Adam((list(encoder.parameters()) + list(decoder.parameters())), lr=lr)
    criterion1 = nn.MSELoss(reduce=None)  # Reconstruction loss
    criterion1 = nn.MSELoss()  # Reconstruction loss

    for epoch in range(epochs):
      # epoch_loss_list = []
      encoder.train()
      decoder.train()
      total_loss = 0
      total_loss_or = 0
      total_loss_ndvi = 0
      for batch_idx, (data_or, data_ndvi, id) in enumerate(loader):
        if gpu:
          data_or = data_or.cuda()
          data_ndvi = data_ndvi.cuda()
        encoded, id1 = encoder(Variable(data_or), Variable(data_ndvi))
        decoded_or, decoded_ndvi = decoder(encoded, id1)
        loss_or = criterion1(decoded_or, Variable(data_or))
        loss_ndvi = criterion1(decoded_ndvi, Variable(data_ndvi))
        loss = (loss_or + loss_ndvi)/2
        loss_data_or = loss_or.item()
        loss_data_ndvi = loss_ndvi.item()
        loss_data = (loss_data_or+loss_data_ndvi)/2
        total_loss += loss_data
        total_loss_or += loss_data_or
        total_loss_ndvi += loss_data_ndvi
        optimizer.zero_grad()
        # loss_or.backward(retain_graph=True)
        # loss_ndvi.backward()
        loss.backward()
        optimizer.step()
        if (batch_idx+1) % 200 == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.7f}\tLoss_or: {:.7f}\tLoss_ndvi: {:.7f}'.format(
            (epoch+1), (batch_idx+1) * batch_size, len(loader)*batch_size,
            100. * (batch_idx+1) / len(loader), loss_data, loss_data_or, loss_data_ndvi))
      epoch_loss = total_loss / len(loader)
      epoch_loss_or = total_loss_or / len(loader)
      epoch_loss_ndvi = total_loss_ndvi / len(loader)
      # epoch_loss_list.append(epoch_loss)
      epoch_stats = "Pretraining Epoch {} Complete: Avg. Loss: {:.7f}, Avg. Loss_or: {:.7f}, Avg. Loss_ndvi: {:.7f}".format(epoch + 1, epoch_loss, epoch_loss_or, epoch_loss_ndvi)
      # print_stats(args.stats_file, epoch_stats)
      torch.save([encoder, decoder], ('ae-model_ep_'+str(epoch+1)+"_loss_"+str(round(epoch_loss, 7))+'test'+'.pkl') )
      # if (epoch+1) % 5 == 0:
      #     plotting(epoch+1, epoch_loss_list, path_results)  
