import struct
import numpy as np
import tensorflow as tf
import os

def load_data(name):

    """loading train and test data"""
    with open(name + '-labels.idx1-ubyte', 'rb') as label_path:

        magic, n = struct.unpack('>II', label_path.read(8))
        label = np.fromfile(label_path, dtype=np.uint8)

    with open(name + '-images.idx3-ubyte', 'rb') as image_path:

        magic, num, row, col = struct.unpack('>IIII', image_path.read(16))
        image = np.fromfile(image_path, dtype = np.uint8).reshape(len(label), 784)
        image = ((image/255) - 0.5) * 2

    return image, label

X, y = load_data('train')
X_train, y_train = X[:50000, :], y[:50000]
X_valid, y_valid = X[50000: ,:], y[50000:]
X_test, y_test = load_data('t10k')

def batch_generator(

        X,
        y,
        batch_size = 64,
        shuffle = False,
        random_seed = None

):

    idx = np.arange(y.shape[0])
    if shuffle:

        rng = np.random.RandomState(random_seed)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]

    for i in range(0, X.shape[0], batch_size):

        yield (X[i : i + batch_size, :], y[i : i + batch_size])

mean_val = np.mean(X_train, axis = 0)
std_val = np.std(X_train)
X_train_center = (X_train - mean_val)/std_val
X_valid_center = (X_valid - mean_val)/std_val
X_test_center = (X_test - mean_val)/std_val

class ConvNN():

    def __init__(self, batch_size = 64, epochs = 20, learn_rate = 1e-4, dropout_rate = 0.5, shuffle = True, random_seed = None):

        np.random.seed(random_seed)
        self.batch_size = batch_size
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.dropout_rate = dropout_rate
        self.shuffle = shuffle

        g = tf.Graph()
        with g.as_default():

            tf.compat.v1.set_random_seed(random_seed)
            self.build()
            self.init_op = tf.compat.v1.global_variables_initializer()
            self.saver = tf.compat.v1.train.Saver()
            self.sess = tf.compat.v1.Session(graph = g)
            
    def build(self):

        tf_x = tf.compat.v1.placeholder(

            tf.float32,
            shape = [None, 784],
            name = 'tf_x'

        )
        tf_y = tf.compat.v1.placeholder(

            tf.int32,
            shape = [None],
            name = 'tf_y'

        )
        is_train = tf.compat.v1.placeholder(
            
            tf.bool,
            shape = (),
            name = 'is_train'

        )
        tf_x_image = tf.reshape(

            tf_x,
            shape = [-1, 28, 28, 1],
            name = 'input_x_2dimages'

        )
        tf_y_onehot = tf.one_hot(

            indices = tf_y,
            depth = 10,
            dtype = tf.float32,
            name = 'input_y_onehot'

        )
        h1 = tf.compat.v1.layers.conv2d(

            tf_x_image,
            kernel_size = (5, 5),
            filters = 32,
            activation = tf.compat.v1.nn.relu

        )
        h1_pool = tf.compat.v1.layers.max_pooling2d(

            h1,
            pool_size = (2, 2),
            strides = (2,2)
        )
        h2 = tf.compat.v1.layers.conv2d(

            h1_pool,
            kernel_size = (5, 5),
            filters = 64,
            activation = tf.compat.v1.nn.relu

        )
        h2_pool = tf.compat.v1.layers.max_pooling2d(

            h2,
            pool_size = (2, 2),
            strides = (2, 2)

        )
        input_shape = h2_pool.get_shape().as_list()
        n_input_units = np.prod(input_shape[1:])
        h2_pool_flat = tf.reshape(

            h2_pool,
            shape = [-1, n_input_units]

        )
        h3 = tf.compat.v1.layers.dense(

            h2_pool_flat,
            1024,
            activation = tf.nn.relu
        )
        h3_drop = tf.compat.v1.layers.dropout(

            h3,
            rate = self.dropout_rate,
            training = is_train

        )
        h4 = tf.compat.v1.layers.dense(

            h3_drop,
            10,
            activation = None

        )
        predictions = {

            'probabilities': tf.nn.softmax(h4, name = 'probabilities'),
            'labels': tf.cast(tf.argmax(h4, axis = 1), tf.int32, name = 'labels')

        }
        cross_entropy_loss = tf.reduce_mean(

            tf.nn.softmax_cross_entropy_with_logits(logits = h4, labels = tf_y_onehot),
            name = 'cross_entropy_loss'

        )
        optimizer = tf.compat.v1.train.AdamOptimizer(self.learn_rate)
        optimizer = optimizer.minimize(cross_entropy_loss, name = 'train_op')
        correct_predictions = tf.equal(

            predictions['labels'],
            tf_y,
            name = 'correct_preds'

        )
        accuracy = tf.reduce_mean(

            tf.cast(correct_predictions, tf.float32),
            name = 'accuracy'

        )

    def save(self, epoch, path = './tflayers-model/'):

        if not os.path.isdir(path):

            os.makedirs(path)

        print(f'Saving model in the {path}')
        self.saver.save(

            self.sess,
            os.path.join(path, 'model.ckpt'),
            global_step = epoch

        )

    def load(self, epoch, path):

        print(f'loading model from {path}')
        self.saver.restore(

            self.sess,
            os.path.join(path, f'model.ckpt-{epoch}')
        )

    def train(self, train_set, validation_set = None, initialize = True):

        if initialize:

            self.sess.run(self.init_op)

        self.train_cost_ = []
        X_data = np.array(train_set[0])
        y_data = np.array(train_set[1])

        for epoch in range(1, self.epochs + 1):

            batch_gen = batch_generator(X_data, y_data, shuffle = self.shuffle)
            avg_loss = 0.0

            for i, (batch_x, batch_y) in enumerate(batch_gen):

                feed = {

                    'tf_x: 0': batch_x,
                    'tf_y: 0': batch_y,
                    'is_train: 0': True

                }
                loss, _ = self.sess.run(

                    ['cross_entropy_loss: 0', 'train_op'],
                    feed_dict = feed

                )
                avg_loss += loss

            print(f'epoch {epoch} vs training avgloss {avg_loss}')

            if validation_set is not None:

                feed = {

                    'tf_x: 0': batch_x,
                    'tf_y: 0': batch_y,
                    'is_train: 0': False

                }
                valid_acc = self.sess.run('accuracy: 0', feed_dict = feed)
                print(f'validation accuracy {valid_acc}')

            else:

                print()

    def predict(self, X_test, return_prob = False):

        feed = {

            'tf_x: 0': X_test,
            'is_train: 0': False
            
        }
        if return_prob:

            return self.sess.run('probabilities: 0', feed_dict = feed)

        else:

            return self.sess.run('labels: 0', feed_dict = feed)

if __name__ == '__main__':

    """testing model"""
    cnn = ConvNN(random_seed = 5)
    cnn.train(

        train_set = (X_train_center, y_train),
        validation_set = (X_valid_center, y_valid),
        initialize = True
    )
