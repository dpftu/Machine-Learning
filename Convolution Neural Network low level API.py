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

def batch_generator(X, y, batch_size = 64, shuffle = False, random_seed = None):

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

def conv_layer(input_tensor, name, kerner_size, n_output_channels, padding_mode = 'SAME', strides = (1, 1, 1, 1)):

    with tf.compat.v1.variable_scope(name):

        input_shape = input_tensor.get_shape().as_list()
        n_input_channels = input_shape[-1]
        weight_shape = list(kerner_size) + [n_input_channels, n_output_channels]
        weights = tf.compat.v1.get_variable(

            name = '_weights',
            shape = weight_shape

        )

        print(weights)

        biases = tf.compat.v1.get_variable(

            name = 'weights',
            initializer = tf.zeros(shape = [n_output_channels])

        )
        print(biases)
        conv = tf.nn.conv2d(

            input = input_tensor,
            filters = weights,
            strides =  strides,
            padding = padding_mode

        )
        print(conv)
        conv = tf.nn.bias_add(

            conv,
            biases,
            name = 'net_pre-activation'

        )
        print(conv)
        conv = tf.nn.relu(

            conv,
            name = 'activation'

        )
        print(conv)
        
    return conv

g = tf.Graph()
with g.as_default():

    x = tf.compat.v1.placeholder(tf.float32, shape = [None, 28, 28, 1])
    conv_layer(

        x,
        name = 'convtest',
        kerner_size = (3,3),
        n_output_channels = 32
    )

del g, x

def fc_layer(input_tensor, name, n_output_units, activation_fn = None):

    with tf.compat.v1.variable_scope(name):

        input_shape = input_tensor.get_shape().as_list()[1:]
        n_input_units = np.prod(input_shape)
        if len(input_shape) > 1:

            input_tensor = tf.reshape(

                input_tensor,
                shape = (-1, n_input_units)
            )

        weights_shape = [n_input_units, n_output_units]
        weights = tf.compat.v1.get_variable(

            name = 'weights',
            shape = weights_shape

        )
        
        print(weights)
        biases = tf.compat.v1.get_variable(

            name = 'biases',
            initializer = tf.zeros(shape = [n_output_units])

        )
        print(biases)
        layer = tf.matmul(input_tensor, weights)
        print(layer)
        layer = tf.nn.bias_add(layer, biases, name = 'net_pre-activation')
        print(layer)

        if activation_fn is None:

            return layer

        layer = activation_fn(layer, name = 'activation')

        print(layer)

        return layer


g = tf.Graph()
with g.as_default():

    x = tf.compat.v1.placeholder(

        tf.float32,
        shape = [None, 28, 28, 1]

    )

    fc_layer(

        x,
        name = 'fctest',
        n_output_units = 32,
        activation_fn = tf.nn.relu

    )

del g, x

def build_cnn(learning_rate = 1e-4):

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
    tf_x_image = tf.reshape(

        tf_x,
        shape = [-1, 28, 28, 1],
        name = 'tf_x_reshaped'

    )
    tf_y_onehot = tf.compat.v1.one_hot(

        indices = tf_y,
        depth = 10,
        dtype = tf.float32,
        name = 'tf_y_onehot'

    )
    print('\n The first layer:::')
    h1 = conv_layer(

        tf_x_image,
        name = 'conv_1',
        kerner_size = (5,5),
        padding_mode = 'VALID',
        n_output_channels = 32

    )

    h1_pool = tf.nn.max_pool(

        h1,
        ksize = [1, 2, 2, 1],
        strides = [1, 2, 2, 1],
        padding = 'SAME'

    )
    print('\n 2nd layer!!!!!!')
    h2 = conv_layer(

        h1_pool,
        name = 'conv_2',
        kerner_size = (5,5),
        padding_mode = 'VALID',
        n_output_channels = 64

    )
    h2_pool = tf.nn.max_pool(

        h2,
        ksize = [1, 2, 2, 1],
        strides = [1, 2, 2, 1],
        padding = 'SAME'

    )
    print('\n Layer 3!!!!')
    h3 = fc_layer(

        h2_pool,
        name = 'fc_3',
        n_output_units = 1024,
        activation_fn = tf.nn.relu

    )
    keep_prob = tf.compat.v1.placeholder(

        tf.float32,
        name = 'fc_keep_prob'

    )
    h3_drop = tf.compat.v1.nn.dropout(

        h3,
        keep_prob =  keep_prob,
        name = 'dropout_layer'

    )
    print('\n Layer 4: fully-connected, linear activation')
    h4 = fc_layer(

        h3_drop,
        name = 'fc_4',
        n_output_units = 10,
        activation_fn = None

    )

    predictions = {

        'probabilities': tf.nn.softmax(h4, name = 'probabilities'),
        'labels': tf.cast(tf.argmax(h4, axis = 1), tf.int32, name = 'labels')

    }

    cross_entropy_loss = tf.reduce_mean(

        tf.nn.softmax_cross_entropy_with_logits(logits = h4, labels = tf_y_onehot) , name = 'cross_entropy_loss'

    )
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    optimizer = optimizer.minimize(cross_entropy_loss, name = 'train_op')
    correct_predictions = tf.equal(

        predictions['labels'],
        tf_y,
        name = 'correct_pred'

    )
    accuracy = tf.reduce_mean(

        tf.cast(correct_predictions, tf.float32),
        name = 'accuracy'

    )


def save(saver, sess, epoch, path = './model/'):

    if not os.path.isdir(path):

        os.makedirs(path)

    print(f'saving model in {path}')
    saver.save(sess, os.path.join(path, 'cnn-model.ckpt'), global_step = epoch)


def load(saver, sess, path, epoch):

    print(f'loading model from {path}')
    saver.restore(sess, os.path.join(path, f'cnn-model.ckpt-{epoch}'))

def train(sess, training_set, validation_set =  None, initializer = True, epochs = 20, shuffle = True,
          dropout = 0.5, random_seed = None):

    X_data = np.array(training_set[0])
    y_data = np.array(training_set[1])
    training_loss = []

    if initializer:

        sess.run(tf.compat.v1.global_variables_initializer())

    np.random.seed(random_seed)

    for epoch in range(1, epochs + 1):

        batch_gen = batch_generator(

            X_data,
            y_data,
            shuffle = shuffle

        )
        avg_loss = 0.0

        for i, (batch_x, batch_y) in enumerate(batch_gen):

            feed = {

                'tf_x: 0': batch_x,
                'tf_y: 0': batch_y,
                'fc_keep_prob: 0': dropout

            }

            loss, _ = sess.run(

                ['cross_entropy_loss:0', 'train_op'],
                feed_dict = feed

            )
            avg_loss += loss

        training_loss.append(avg_loss/(i+1))
        print(f'epoch {epoch} vs training loss {avg_loss}')

        if validation_set is not None:

            feed = {

                'tf_x: 0': validation_set[0],
                'tf_y: 0': validation_set[1],
                'fc_keep_prob: 0': 1.0

            }
            valid_acc = sess.run('accuracy: 0', feed_dict = feed)
            print(f'validation acc is {valid_acc}')

        else:

            print()


def predict(sess, X_test, return_prob = False):

    feed = {

        'tf_x: 0': X_test,
        'fc_keep_prob: 0': 1.0

    }

    if return_prob:

        return sess.run('probabilities: 0', feed_dict = feed)

    else:

        return sess.run('labels: 0', feed_dict = feed)

learn_rate = 1e-4
random_seed = 5
g = tf.Graph()
with g.as_default():

    tf.compat.v1.set_random_seed(random_seed)
    build_cnn()
    saver = tf.compat.v1.train.Saver()

with tf.compat.v1.Session(graph = g) as sess:

    train(

        sess,
        training_set = (X_train_center, y_train),
        validation_set = (X_valid_center, y_valid),
        initializer = True,
        random_seed = 5

    )
    save(saver, sess, epoch = 20)


del g
g2 = tf.Graph()
with g2.as_default():

    tf.compat.v1.set_random_seed(random_seed)
    build_cnn()
    saver = tf.compat.v1.train.Saver()

with tf.compat.v1.Session(graph = g2) as sess:

    load(saver, sess, epoch = 20, path = './model/')
    preds = predict(sess, X_test_center, return_prob = False)
    print(f'test accuracy is {np.sum(preds == y_test)/len(y_test)}')
  
