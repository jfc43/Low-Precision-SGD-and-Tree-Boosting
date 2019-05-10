from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def get_mnist():
    '''
    Returns the train and test splits of the MNIST digits dataset,
    where x_train and x_test are shaped into the tensorflow image data
    shape and normalized to fit in the range [0, 1]
    '''
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    x_train = mnist.train.images
    y_train = mnist.train.labels

    x_test = mnist.test.images
    y_test = mnist.test.labels

    #index = np.logical_or(y_train == 9, y_train == 8)
    index = np.arange(x_train.shape[0])
    np.random.shuffle(index)
    x_train = x_train[index]
    y_train = y_train[index]

    #index = np.logical_or(y_test == 9, y_test == 8)
    #np.random.shuffle(index)
    #x_test = x_test[index]
    #y_test = y_test[index] - 8

    return x_train, x_test, y_train, y_test

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = get_mnist()
    num_class = 10

    with open('train.txt', 'w') as f:
        for i in range(x_train.shape[0]):
            for j in range(x_train.shape[1]):
                f.write(str(x_train[i][j]) + ' ')
            f.write('\n')
            for k in range(num_class):
                if y_train[i] == k:
                    f.write('1 ')
                else:
                    f.write('0 ')
            f.write('\n')

    with open('test.txt', 'w') as f:
        for i in range(x_test.shape[0]):
            for j in range(x_test.shape[1]):
                f.write(str(x_test[i][j]) + ' ')
            f.write('\n')
            for k in range(num_class):
                 if y_test[i] == k:
                     f.write('1 ')
                 else:
                     f.write('0 ')
            f.write('\n')
