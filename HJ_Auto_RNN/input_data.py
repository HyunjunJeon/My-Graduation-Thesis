"""Functions for downloading and reading MNIST data."""
from __future__ import print_function
import numpy as np
import csv
import os

def read_data():
    filenames = os.listdir('./data')
    X = []
    for epoch in range(len(filenames) // 2 -1):
        csvdata = []
        f = open('./data/data_' + str(epoch+1) + '.csv', 'r')
        csvR = csv.reader(f)
        for row in csvR:
            csvdata.append([float(i) for i in row])
        if epoch == 0:
            X = np.asarray(csvdata, dtype=np.float32)
        else:
            X_temp = np.asarray(csvdata, dtype=np.float32)
            X = np.dstack([X, X_temp])

    csvydata = []
    fy = open('./data/data_y.csv', 'r')
    csvyR = csv.reader(fy)
    for row in csvyR:
        csvydata.append([float(i) for i in row])
    y = np.asarray(csvydata)
    return (X, y)

def read_test():
    filenames = os.listdir('./data')
    X = []
    for epoch in range(len(filenames) // 2 - 1):
        csvdata = []
        f = open('./data/test_' + str(epoch + 1) + '.csv', 'r')
        csvR = csv.reader(f)
        for row in csvR:
            csvdata.append([float(i) for i in row])
        if epoch == 0:
            X = np.asarray(csvdata, dtype=np.float32)
        else:
            X_temp = np.asarray(csvdata, dtype=np.float32)
            X = np.dstack([X, X_temp])

    csvydata = []
    fy = open('./data/test_y.csv', 'r')
    csvyR = csv.reader(fy)
    for row in csvyR:
        csvydata.append([float(i) for i in row])
    y = np.asarray(csvydata)
    return (X, y)

class DataSet(object):
  def __init__(self, datas, labels):
      assert datas.shape[0] == labels.shape[0], ("datas.shape: %s labels.shape: %s" % (datas.shape, labels.shape))
      self._num_examples = datas.shape[0]
      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      self._datas = datas
      self._labels = labels
      self._epochs_completed = 0
      self._index_in_epoch = 0
  @property
  def datas(self):
    return self._datas
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed
  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      # np.random.shuffle(perm)
      self._datas = self._datas[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._datas[start:end], self._labels[start:end]
def read_data_sets():
  class DataSets(object):
    pass
  data_sets = DataSets()
  train_datas, train_labels = read_data()
  test_datas , test_labels = read_test()
  data_sets.train = DataSet(train_datas, train_labels)
  data_sets.test = DataSet(test_datas, test_labels)
  return data_sets
