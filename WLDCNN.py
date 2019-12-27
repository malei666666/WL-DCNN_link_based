from keras import backend as K
import numpy as np
from keras import layers
from keras.models import Model 
from keras import Input
from keras.utils import plot_model
import keras
import scipy.io
import math
from keras.optimizers import Adam
import loaddataforwldcnn
#this code is an example for a specific data set:PB
global row,col

filename='cel'

X_S,X_L,train_label, X_ts,X_tl = loaddataforwldcnn.load_data(filename)
datasize = X_S.shape[0]
row = X_S.shape[-2]
col = X_S.shape[-1]

print(train_label.shape)

def l2_reg(Wl):# define the regularization item for fullt connected layers
    l2 = K.dot(K.transpose(Wl),Wl)
    return 0.01 * l2

def Fnorm_of_kernel(Wk):#define the F-norm for convolution kernels
    Wk = K.pow(Wk,2)
    Wk = K.sum(K.sum(Wk))
    return 0.01*K.sqrt(Wk)

class BatchGenerator(keras.utils.Sequence):
        def __init__(self, datas, datal, labels, batch_size, row, col, shuffle=True):
            self.batch_size = batch_size
            self.datas = datas
            self.datal = datal
            self.labels = labels
            self.indexes = np.arange(len(self.datas))
            self.row=row
            self.col=col
            self.shuffle = shuffle

        def __len__(self):
            return math.ceil(len(self.datas) / float(self.batch_size))

        def __getitem__(self, index):
            batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
            # 根据索引获取datas集合中的数据
            batch_datas = [self.datas[k] for k in batch_indexs]
            batch_datal = [self.datal[k] for k in batch_indexs]
            y_batch = [self.labels[k] for k in batch_indexs]
            # 生成数据
            Xs, Xl, y = self.data_generation(batch_datas,batch_datal,y_batch)

            return [Xs, Xl], y

        def on_epoch_end(self):
            if self.shuffle == True:
                np.random.shuffle(self.indexes)

        def data_generation(self, batch_datas, batchl, y_b):
            thisbatch=len(batch_datas)
            batch_datas=np.array(batch_datas)
            batch_datal = np.array(batchl)
            y_b = np.array(y_b)
            batch_datas.reshape(thisbatch,1, self.row, self.row)
            batch_datal.reshape(thisbatch, 1, self.row, self.row)
            y_b = y_b.reshape(thisbatch,1)
            return batch_datas,batch_datal, y_b

#DCNN construction process
aj_input = Input(batch_shape=(None, 1, row,col), dtype='float32', name='aj')

aj_cnn = layers.Convolution2D(2, 3, strides=1, padding='valid',
                              kernel_regularizer=Fnorm_of_kernel, data_format='channels_first')(aj_input)

aj_cnn = layers.MaxPooling2D(2, 2, 'same', data_format='channels_first')(aj_cnn)

aj_cnn = layers.Convolution2D(8, 3, strides=1, padding='valid',
                              kernel_regularizer=Fnorm_of_kernel,data_format='channels_first')(aj_cnn)

aj_cnn = layers.MaxPooling2D(2, 2, 'same', data_format='channels_first')(aj_cnn)

aj_cnn = layers.Convolution2D(16, 3, strides=1, padding='valid',
                              kernel_regularizer=Fnorm_of_kernel, data_format='channels_first')(aj_cnn)

aj_cnn = layers.MaxPooling2D(2, 2, 'same', data_format='channels_first')(aj_cnn)

aj_cnn = layers.Flatten()(aj_cnn)

aj_cnn = layers.Dropout(0.5)(aj_cnn)

aj_cnn = layers.Dense(128,activation='relu',kernel_regularizer=l2_reg)(aj_cnn)


dis_input = Input(batch_shape=(None, 1, row, col), dtype='float32', name='dis')

dis_cnn = layers.Convolution2D(2, 3, strides=1,padding='same',
                               kernel_regularizer=Fnorm_of_kernel, data_format='channels_first')(dis_input)

dis_cnn = layers.MaxPooling2D(2, 2, 'same', data_format='channels_first')(dis_cnn)

dis_cnn = layers.Convolution2D(16, 3, strides=1, padding='same',
                               kernel_regularizer=Fnorm_of_kernel, data_format='channels_first')(dis_cnn)

dis_cnn = layers.MaxPooling2D(2, 2, 'same', data_format='channels_first')(dis_cnn)

dis_cnn = layers.Convolution2D(32, 3, strides=1, padding='same',
                               kernel_regularizer=Fnorm_of_kernel, data_format='channels_first')(dis_cnn)

dis_cnn = layers.MaxPooling2D(2, 2, 'same', data_format='channels_first')(dis_cnn)

dis_cnn = layers.Flatten()(dis_cnn)

dis_cnn = layers.Dropout(0.5)(dis_cnn)

dis_cnn = layers.Dense(128,activation='relu',kernel_regularizer=l2_reg)(dis_cnn)

concatenated = layers.concatenate([aj_cnn, dis_cnn],axis=-1)

#output layer
answer = layers.Dense(1,activation='sigmoid')(concatenated)


model = Model([aj_input, dis_input], answer)

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

model.summary()
'''print('Training ------------')
# Another way to train the model
training_generator = BatchGenerator(X_S,X_L,train_label,math.ceil(0.1*datasize),row,col)

#model.train_on_batch([X_S, X_L], train_label, batch_size=batchsize,class_weight={0:1.0,1:3.0})
model.fit_generator(training_generator,epochs=300,verbose=1,class_weight={0:1.0,1:3.0}, workers=6)
print('\nTesting ------------')

# Evaluate the model with the metrics we defined earlier
Y_pred = model.predict([X_ts, X_tl])#use the trained network to predict the score for the test set
savefile='pre'+filename+'d+l.mat'
scipy.io.savemat(savefile, {'pre': np.array(Y_pred)})
#we save the results in .mat file,because we will use matlab to calculate the evaluation metrics and plot the results
'''