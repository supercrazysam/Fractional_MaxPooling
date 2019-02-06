import numpy
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
from keras.layers import Conv2D,Conv1D,LSTM
from keras.layers.core import Reshape
from keras.layers.pooling import MaxPooling1D
from Fractional_MAXPOOL import FractionalPooling2D
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU

from keras.preprocessing.image import ImageDataGenerator
import os

K.set_image_dim_ordering('tf')


import tensorflow as tf
K.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0}, gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9))))

from keras.callbacks import TensorBoard
tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
#                  batch_size=32,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=True, # 是否可视化梯度直方图
                 write_images=True,# 是否可视化参数
                 embeddings_freq=0, 
                 embeddings_layer_names=None, 
                 embeddings_metadata=None)

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
batch_size = 720


# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train[0:49984]
y_train = y_train[0:49984]
X_test = X_test[0:9984]
y_test = y_test[0:9984]

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]



datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format="channels_last",#None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(X_train)




# Create the model
model = Sequential()
# Block 1
model.add(Conv2D(64, (3, 3),input_shape=X_train.shape[1:], padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(FractionalPooling2D(pool_ratio=(1, 1.6, 1.6, 1),pseudo_random = True,overlap=True))
# Block 2
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(FractionalPooling2D(pool_ratio=(1, 1.25, 1.25, 1),pseudo_random = True,overlap=True))


# Block 3
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(FractionalPooling2D(pool_ratio=(1, 1.6, 1.6, 1),pseudo_random = True,overlap=True))


# Block 4
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(FractionalPooling2D(pool_ratio=(1, 1.25, 1.25, 1),pseudo_random = True,overlap=True))


# Block 5
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(FractionalPooling2D(pool_ratio=(1, 1.6, 1.6, 1),pseudo_random = True,overlap=True))
# Block 6
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(FractionalPooling2D(pool_ratio=(1, 1.25, 1.25, 1),pseudo_random = True,overlap=True))
model.add(Reshape((16,512)))

# fc layer_1
model.add(Dense(1024, kernel_constraint=maxnorm(3)))
model.add(LeakyReLU(alpha = 0.3))
# fc_layer_2
model.add(Dense(512, kernel_constraint=maxnorm(3)))
model.add(LeakyReLU(alpha = 0.3))


model.add(Flatten())

model.add(Dense(num_classes, activation='softmax'))

opt = keras.optimizers.Adadelta(1,decay=1e-4) #0.1

model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
print(model.summary())

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'test.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)



callbacks = [checkpoint,  tbCallBack]
#model.load_weights('Model.hdf5')
epochs = 30
model.fit_generator( datagen.flow(X_train, y_train, batch_size=batch_size), workers=4, validation_data = [X_test,y_test], nb_epoch=epochs,  callbacks=callbacks,shuffle=True,verbose=1  )

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
