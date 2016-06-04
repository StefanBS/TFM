from __future__ import print_function
import tensorflow as tf
import six.moves.cPickle as pickle
import tools
from time import time

pickle_file = 'Daimler.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

height = 96
width = 48
num_labels = 2
num_channels = 1 # grayscale

train_dataset, train_labels = tools.reformat_conv(train_dataset, train_labels)
valid_dataset, valid_labels = tools.reformat_conv(valid_dataset, valid_labels)
test_dataset, test_labels = tools.reformat_conv(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

batch_size = 128 # Original 128
num_hidden = 128 # Original 64
patch_size = 5
depth = 16      # Original 16


# Initialization
def initialization():
    # Input data. For the training data, we use a placeholder that will be fed at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, height, width, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Constants.
    beta = 0.00001
    global_step = tf.Variable(0)  # count the number of steps taken.
    dropout_keeprate = 0.7
    learning_rate = 0.75
    decay_rate = 0.98
    decay_steps = 1000

    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    layer3_weights = tf.Variable(tf.truncated_normal([height // 4 * width // 4 * depth, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    model_list = {
        1: model_1,
        2: model_2,
        3: model_3,
    }

# Model launcher.
def model_launch(model, model_list):
    if model in model_list:
        result = model_list[model]()
    else:
        print("Model not in model list")
    return result


# Model 1: Conv (stride 1) - ReLU - Max Pool (stride 2) - Conv (stride 2) - ReLU
def model_1(data, layer1_weights, layer1_biases, layer2_weights, layer2_biases,
            layer3_weights, layer3_biases, layer4_weights, layer4_biases, isTraining=False):
    if not isTraining:
        dropout_keeprate = 1
    else:
        dropout_keeprate = 0.7
    conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer1_biases)
    hidden = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer2_biases)
    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases), dropout_keeprate)
    return tf.matmul(hidden, layer4_weights) + layer4_biases


# Model 2: Conv (stride 1) - ReLU - Max Pool (stride 2) - Conv (stride 1) - ReLU - Max Pool (stride 2)
def model_2(data, layer1_weights, layer1_biases, layer2_weights, layer2_biases, layer3_weights, layer3_biases,
            layer4_weights, layer4_biases, isTraining=False):
    if not isTraining:
        dropout_keeprate = 1
    else:
        dropout_keeprate = 0.7
    conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer1_biases)
    hidden = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer2_biases)
    hidden = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')  # Capa adicional
    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases), dropout_keeprate)
    return tf.matmul(hidden, layer4_weights) + layer4_biases


# Model 3: Conv (stride 2) - ReLU - Conv (stride 2) - ReLU
def model_3(data, layer1_weights, layer1_biases, layer2_weights, layer2_biases, layer3_weights, layer3_biases,
            layer4_weights, layer4_biases, isTraining=False):
    if not isTraining:
        dropout_keeprate = 1
    else:
        dropout_keeprate = 0.7
    conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer1_biases)
    hidden = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer2_biases)
    hidden = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')  # Capa adicional
    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases), dropout_keeprate)
    return tf.matmul(hidden, layer4_weights) + layer4_biases

graph = tf.Graph()

with graph.as_default():

    # Training computation.
    logits = model(tf_train_dataset, True)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))