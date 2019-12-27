import scipy.io
import numpy as np
def load_data(filename):
    trainname = 'fortrain'+filename+'.mat'
    testname = 'fortest'+filename+'.mat'
    data = scipy.io.loadmat(trainname) # we use matlab generate inputs for each data set

    #read the trainning set
    Sm=data['traind'].transpose(2,0,1)# traind is the similarity matrix constructed by out algorithm, transpose is needed because matlab aixs is different from tensorflow
    samples, row, col=Sm.shape
    Sm=Sm.reshape(samples,1,row,col)#consists of 15043 positive samples and 45129 negative samples 60172

    Lm=data['trainl'].transpose(2,0,1)#trainl is the Laplacian eigen vector matrix constructed by out algorithm
    Lm=Lm.reshape(samples,1,row,col)
    label = data['label']# when generating inputs, we also add the labels


    #read the test set
    datat = scipy.io.loadmat(testname) # 假设文件名为1.mat
    Stm=datat['fortestd'].transpose(2,0,1)
    samples, row, col = Stm.shape
    T_samples, row, col=Stm.shape
    Stm=Stm.reshape(samples,1,row,col)#consists of 1672 positive samples and 10000 negative samples
    Ltm=datat['fortestl'].transpose(2,0,1)
    Ltm=Ltm.reshape(samples,1,row,col)
    return Sm, Lm, label, Stm, Ltm

