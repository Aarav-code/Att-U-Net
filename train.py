import tensorflow as tf
import os
import random
import numpy as np

from tqdm import tqdm 
from zipfile import ZipFile

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import glob
import cv2
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
from keras.callbacks import TensorBoard, ModelCheckpoint
import scipy.misc as mc
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau

path1='data/images'
path2='data/mask'
files_list=sorted(glob.glob(path1+"/*"))
masks_list=sorted(glob.glob(path2+"/*"))


IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3

ids=len(files_list)
X = np.zeros((ids, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
Y = np.zeros((ids, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
for i in range(len(files_list)):
  filename=files_list[i]
  mask_name=masks_list[i]
  x_img=cv2.imread(filename)
  #x_img=img.pixel_array
  #x_img=median_filter(x_img,3)
  x_img=resize(x_img, (IMG_HEIGHT, IMG_WIDTH))
  X[i] = x_img

  mask1 = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
  mask=cv2.imread(mask_name,cv2.IMREAD_GRAYSCALE)
  mask = np.expand_dims(resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
  mask1 = np.maximum(mask1, mask)
  Y[i] = mask1

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.3, random_state=42)

TensorBoard(log_dir='./autoencoder', histogram_freq=0,
            write_graph=True, write_images=True)

from  SA_UNet import *
model=SA_UNet(input_size=(256,256,3),start_neurons=16,lr=1e-3,keep_prob=0.87,block_size=7)
# weight="Model/CHASE/SA_UNet.h5"

# restore=False

# if restore and os.path.isfile(weight):
#     model.load_weights(weight)

# model_checkpoint = ModelCheckpoint(weight, monitor='val_accuracy', verbose=1, save_best_only=False)

# # plot_model(model, to_file='unet_resnet.png', show_shapes=False, show_layer_names=)

# history=model.fit(X_train, Y_train,
#                 epochs=5, #first  100 with lr=1e-3,,and last 50 with lr=1e-4
#                 batch_size=2,
#                 # validation_split=0.1,
#                 validation_data=(X_valid, Y_valid),
#                 shuffle=True,
#                 callbacks= [TensorBoard(log_dir='./autoencoder'), model_checkpoint])


# print(history.history.keys())

# # summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_accuracy'])
# plt.title('SA-UNet Accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validate'], loc='lower right')
# plt.show()
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model_sk.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, batch_size=32, epochs=100, callbacks=[earlystopper, checkpointer],\
                    validation_data=(X_valid, Y_valid))

