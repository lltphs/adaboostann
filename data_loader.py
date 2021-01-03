import numpy as np,dataget,pickle

def i2o(idx):
    onehot=np.zeros((idx.shape[0],10))
    onehot[np.arange(idx.shape[0]),idx]=1
    return onehot

def p2i(prob):
    return np.argmax(prob,1)

def p2o(prob):
    return i2o(p2i(prob))

def cifar10():
    x_train,y_train_idx,x_test,y_test_idx=dataget.image.cifar10(global_cache=True).get()
    # [(x_train,y_train_idx),(x_test,y_test_idx)]=pickle.load(open('cifar10','rb'))
    x_train=np.average(x_train,3).reshape(-1,32*32)/255.0
    x_test=np.average(x_test,3).reshape((-1,32*32))/255.0
    y_train_idx=y_train_idx.reshape(-1)
    y_test_idx=y_test_idx.reshape(-1)
    y_train_one_hot=np.zeros((y_train_idx.shape[0],10))
    y_train_one_hot[np.arange(y_train_idx.shape[0]),y_train_idx]=1
    y_test_one_hot=np.zeros((y_test_idx.shape[0],10))
    y_test_one_hot[np.arange(y_test_idx.shape[0]),y_test_idx]=1
    return [(x_train,y_train_one_hot),(x_test,y_test_one_hot)]

def fashion_mnist():
    x_train,y_train_idx,x_test,y_test_idx=dataget.image.fashion_mnist(global_cache=True).get()
    # [(x_train,y_train_idx),(x_test,y_test_idx)]=pickle.load(open('fashion_mnist','rb'))
    x_train=x_train.reshape(-1,28*28)/255.0
    x_test=x_test.reshape(-1,28*28)/255.0
    y_train_idx=y_train_idx.reshape(-1)
    y_test_idx=y_test_idx.reshape(-1)
    y_train_one_hot=np.zeros((y_train_idx.shape[0],10))
    y_train_one_hot[np.arange(y_train_idx.shape[0]),y_train_idx]=1
    y_test_one_hot=np.zeros((y_test_idx.shape[0],10))
    y_test_one_hot[np.arange(y_test_idx.shape[0]),y_test_idx]=1
    return [(x_train,y_train_one_hot),(x_test,y_test_one_hot)]