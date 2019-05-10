from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
import numpy as np

def get_mnist():
    '''
    Returns the train and test splits of the MNIST digits dataset,
    where x_train and x_test are shaped into the tensorflow image data
    shape and normalized to fit in the range [0, 1]
    '''
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # reshape and standardize x arrays
    x_train = x_train / 255
    x_test = x_test / 255
    return x_train, x_test, y_train, y_test



if __name__ == '__main__':
    x_train, x_test, y_train, y_test = get_mnist()

    x_train = x_train.reshape((-1, np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((-1, np.prod(x_test.shape[1:])))
    y_train = y_train.reshape(y_train.shape[0])
    y_test = y_test.reshape(y_test.shape[0])

    index = np.logical_or(y_train == 9, y_train == 8)
    np.savetxt('train.out', x_train[index], delimiter=' ')
    np.savetxt('train.label', y_train[index]-8, delimiter=' ')
    print("training set:")
    for i in range(10):
        print(i, np.count_nonzero(y_train == i))

    index = np.logical_or(y_test == 9, y_test == 8)
    np.savetxt('test.out', x_test[index], delimiter=' ')
    np.savetxt('test.label', y_test[index]-8, delimiter=' ')
    print("test set: ")
    for i in range(10):
        print(i, np.count_nonzero(y_test == i))


