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





# coding=utf-8
"""
自定义Tensorboard显示特征图
"""
from keras.callbacks import Callback
from keras import backend as K
import warnings
import math
import numpy as np

from keras.callbacks import TensorBoard

class MyTensorBoard(Callback):
    """TensorBoard basic visualizations.
    log_dir: the path of the directory where to save the log
        files to be parsed by TensorBoard.
    write_graph: whether to visualize the graph in TensorBoard.
        The log file can become quite large when
        write_graph is set to True.
    batch_size: size of batch of inputs to feed to the network
        for histograms computation.
    input_images: input data of the model, because we will use it to build feed dict to
        feed the summary sess.
    write_features: whether to write feature maps to visualize as
        image in TensorBoard.
    update_features_freq: update frequency of feature maps, the unit is batch, means
        update feature maps per update_features_freq batches
    update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`, writes
        the losses and metrics to TensorBoard after each batch. The same
        applies for `'epoch'`. If using an integer, let's say `10000`,
        the callback will write the metrics and losses to TensorBoard every
        10000 samples. Note that writing too frequently to TensorBoard
        can slow down your training.
    """

    def __init__(self, log_dir='./logs',
                 batch_size=64,
                 update_features_freq=1,
                 input_images=None,
                 write_graph=True,
                 write_features=False,
                 update_freq='epoch'):
        super(MyTensorBoard, self).__init__()
        global tf, projector
        try:
            import tensorflow as tf
            from tensorflow.contrib.tensorboard.plugins import projector
        except ImportError:
            raise ImportError('You need the TensorFlow module installed to '
                              'use TensorBoard.')

        if K.backend() != 'tensorflow':
            if write_graph:
                warnings.warn('You are not using the TensorFlow backend. '
                              'write_graph was set to False')
                write_graph = False
            if write_features:
                warnings.warn('You are not using the TensorFlow backend. '
                              'write_features was set to False')
                write_features = False

        self.input_images = input_images[0]
        self.log_dir = log_dir
        self.merged = None
        self.im_summary = []
        self.lr_summary = None
        self.write_graph = write_graph
        self.write_features = write_features
        self.batch_size = batch_size
        self.update_features_freq = update_features_freq
        if update_freq == 'batch':
            # It is the same as writing as frequently as possible.
            self.update_freq = 1
        else:
            self.update_freq = update_freq
        self.samples_seen = 0
        self.samples_seen_at_last_write = 0

    def set_model(self, model):
        self.model = model
        if K.backend() == 'tensorflow':
            self.sess = K.get_session()
        if self.merged is None:
            # 显示特征图
            # 遍历所有的网络层
            for layer in self.model.layers:
                # 获取当前层的输出与名称
                feature_map = layer.output
                feature_map_name = layer.name.replace(':', '_')

                if self.write_features and len(K.int_shape(feature_map)) == 4:
                    # 展开特征图并拼接成大图
                    flat_concat_feature_map = self._concact_features(feature_map)
                    # 判断展开的特征图最后通道数是否是1
                    shape = K.int_shape(flat_concat_feature_map)
                    assert len(shape) == 4 and shape[-1] == 1
                    # 写入tensorboard
                    self.im_summary.append(tf.summary.image(feature_map_name, flat_concat_feature_map, 1))  # 第三个参数为tensorboard展示几个   #4 to 1

            # 显示学习率的变化
            #self.lr_summary = tf.summary.scalar("learning_rate", self.model.optimizer.lr)

        self.merged = tf.summary.merge_all()

        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

   

    def on_train_end(self, _):
        self.writer.close()

    def on_batch_end(self, batch, logs=None):
       
        # 每update_features_freq个batch刷新特征图
        if batch % self.update_features_freq == 0:
            # 计算summary_image
            global feed_dict
            #feed_dict = dict(zip(self.model.inputs, self.input_images[np.newaxis, ...]))
            feed_dict = dict(zip(self.model.inputs, np.expand_dims(self.input_images[np.newaxis, ...], axis=0)))
            for i in range(len(self.im_summary)):
                summary = self.sess.run(self.im_summary[i], feed_dict)
                self.writer.add_summary(summary, self.samples_seen)

      
    def _concact_features(self, conv_output):
        """
        对特征图进行reshape拼接
        :param conv_output:输入多通道的特征图
        :return: all_concact
        """
        all_concact = None

        num_or_size_splits = conv_output.get_shape().as_list()[-1]
        each_convs = tf.split(conv_output, num_or_size_splits=num_or_size_splits, axis=3)

        if num_or_size_splits < 4:
            # 对于特征图少于4通道的认为是输入，直接横向concact输出即可
            concact_size = num_or_size_splits
            all_concact = each_convs[0]
            for i in range(concact_size - 1):
                all_concact = tf.concat([all_concact, each_convs[i + 1]], 1)
        else:
            concact_size = int(math.sqrt(num_or_size_splits) / 1)
            for i in range(concact_size):
                row_concact = each_convs[i * concact_size]
                for j in range(concact_size - 1):
                    row_concact = tf.concat([row_concact, each_convs[i * concact_size + j + 1]], 1)
                if i == 0:
                    all_concact = row_concact
                else:
                    all_concact = tf.concat([all_concact, row_concact], 2)
        return all_concact


# 定义一个读取数据的generate函数
'''
def _get_data(self, path, batch_size, normalize):
     """
     Generator to be used with model.fit_generator()
     :param path: .npz数据路径
     :param batch_size: batch_size
     :param normalize: 是否归一化
     :return:
     """
     while True:
         files = glob.glob(os.path.join(path, '*.npz'))
         np.random.shuffle(files)
         for npz in files:
             # Load pack into memory
             archive = np.load(npz)
             images = archive['images']
             offsets = archive['offsets']
             del archive
             self._shuffle_in_unison(images, offsets)

             # 切分获得batch
             num_batches = int(len(offsets) / batch_size)
             images = np.array_split(images, num_batches)
             offsets = np.array_split(offsets, num_batches)
             while offsets:
                 batch_images = images.pop()
                 batch_offsets = offsets.pop()
                 if normalize:
                     batch_images = (batch_images - 127.5) / 127.5
                 yield batch_images, batch_offsets

train_loader = _get_data(/path/xxx, 128, True)
'''











# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
batch_size = 32






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


feature = MyTensorBoard(log_dir='./logs/feature',  # log 目录
                 update_features_freq=int(len(X_train)/batch_size)/2,
                 input_images=X_train[:batch_size],
                 batch_size=batch_size,
                 write_features=True,
                 write_graph=False,
                 update_freq='batch')


tbCallBack = TensorBoard(log_dir='./logs/loss',  # log 目录
                 histogram_freq=0,  #### 按照何等频率（epoch）来计算直方图，0为不计算
                 #batch_size=32,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=True, # 是否可视化梯度直方图
                 write_images=False,# 是否可视化参数
                 embeddings_freq=0, 
                 embeddings_layer_names=None, 
                 embeddings_metadata=None)


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

opt = keras.optimizers.Adadelta(0.01,decay=1e-4) #0.1

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




#callbacks = [checkpoint,  tbCallBack, showweight]
callbacks = [checkpoint,   feature,  tbCallBack]
#callbacks = [checkpoint,   tbCallBack]
#model.load_weights('Model.hdf5')






epochs = 50
model.fit_generator( datagen.flow(X_train, y_train, batch_size=batch_size), workers=4, validation_data = [X_test,y_test], nb_epoch=epochs,  callbacks=callbacks,shuffle=True,verbose=1 , steps_per_epoch=len(X_train)/batch_size )

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
