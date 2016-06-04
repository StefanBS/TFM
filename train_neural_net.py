from __future__ import print_function
import tensorflow as tf
import six.moves.cPickle as pickle
import tools

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

train_dataset, train_labels = tools.reformat_nn(train_dataset, train_labels)
valid_dataset, valid_labels = tools.reformat_nn(valid_dataset, valid_labels)
test_dataset, test_labels = tools.reformat_nn(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

batch_size = 128
hidden_nodes = 1024

graph = tf.Graph()

with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, height * width))
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

    # Variables in.
    weights_in = tf.Variable(tf.truncated_normal([height * width, hidden_nodes]))
    biases_in = tf.Variable(tf.zeros([hidden_nodes]))

    # Variables out.
    weights_out = tf.Variable(tf.truncated_normal([hidden_nodes, num_labels]))
    biases_out = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    layer1 = tf.matmul(tf_train_dataset, weights_in) + biases_in
    hidden_layer = tf.nn.dropout(tf.nn.relu(layer1), dropout_keeprate)
    logits = tf.matmul(hidden_layer, weights_out) + biases_out
    regularization = 0.5 * beta * (tf.nn.l2_loss(weights_in) + tf.nn.l2_loss(weights_out))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + regularization

    # Optimizer.
    learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights_in) + biases_in), weights_out) + biases_out)
    test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights_in) + biases_in), weights_out) + biases_out)

num_steps = 5001

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % tools.accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % tools.accuracy(valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % tools.accuracy(test_prediction.eval(), test_labels))