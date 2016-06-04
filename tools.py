import numpy as np
import six.moves.cPickle as pickle

height = 96
width = 48
num_channels = 1
num_labels = 2

def make_arrays(nb_rows, height, width):
    if nb_rows:
        dataset = np.ndarray((nb_rows, height, width), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels

def merge_datasets(pickle_files, train_size):
    num_classes = len(pickle_files)
    train_dataset, train_labels = make_arrays(train_size, height, width)
    tsize_per_class = train_size // num_classes
    start_t = 0
    end_t = tsize_per_class

    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)

                train_letter = letter_set[0:end_t, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
            del(f)
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return train_dataset, train_labels

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

def reformat_conv(dataset, labels):
    dataset = dataset.reshape((-1, height, width, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

def reformat_nn(dataset, labels):
    dataset = dataset.reshape((-1, height * width)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])