from __future__ import print_function
import format
import tools
import matplotlib.pyplot as plt
import numpy as np
import six.moves.cPickle as pickle

# Constants
height = 96             # Pixel height.
width = 48              # Pixel width.
pixel_depth = 255.0     # Number of levels per pixel.
train_size = 25000
test_size = 2500
valid_size = 2500

# Data folders
train_folders = ['DaimlerBenchmark/Data/TrainingData/NonPedestrians', 'DaimlerBenchmark/Data/TrainingData/Pedestrians']
test_folders = ['DaimlerBenchmark/Data/TestData/NonPedestrians', 'DaimlerBenchmark/Data/TestData/Pedestrians']
valid_folders = ['DaimlerBenchmark/Data/ValidationData/NonPedestrians', 'DaimlerBenchmark/Data/ValidationData/Pedestrians']

train_datasets = format.maybe_pickle(train_folders, train_size)
test_datasets = format.maybe_pickle(test_folders, test_size)
valid_datasets = format.maybe_pickle(valid_folders, valid_size)

for i in train_datasets:
    current = pickle.load(open(i, 'r'))
    print(i, len(current))

for i in test_datasets:
    current = pickle.load(open(i, 'r'))
    print(i, len(current))

for i in valid_datasets:
    current = pickle.load(open(i, 'r'))
    print(i, len(current))

train_dataset, train_labels = tools.merge_datasets(train_datasets, train_size)
test_dataset, test_labels = tools.merge_datasets(test_datasets, test_size)
valid_dataset, valid_labels = tools.merge_datasets(valid_datasets, valid_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

train_dataset, train_labels = tools.randomize(train_dataset, train_labels)
test_dataset, test_labels = tools.randomize(test_dataset, test_labels)
valid_dataset, valid_labels = tools.randomize(valid_dataset, valid_labels)

file = test_dataset[11]
print(test_labels[11])
plt.imshow(file[:, :])
plt.show()

print(np.unique(train_labels))
print(np.unique(test_labels))
print(np.bincount(train_labels))
print(np.bincount(test_labels))

# Save pickle
pickle_file = 'Daimler.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

