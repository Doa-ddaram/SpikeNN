import numpy as np


# To represent a list of spikes (sparse representation)
DATA_TYPE = [('indices', np.int32), ('timestamps', np.float32)]


# Iterator for SpikingDataset class
class SpikingDatasetIterator:
    def __init__(self, dataset):
        self._dataset = dataset
        self._index = 0

    def __next__(self):
        if self._index >= self._dataset.shape[0]:
            raise StopIteration
        else:
            sample = (self._dataset.data[self._index], self._dataset.labels[self._index])
            self._index += 1
            return sample


# A dataset class used to store input samples encoded with first-spike coding (one spike per neuron)
# Each sample is represented by an ordered sparse list of spikes (a spike is a tuple <index, timestamp>) and a label
# The dataset must be initialized with a dense numpy array of shape (N_samples, D1, ..., DN), 
# containing firing timestamps in the range [0; max_time], and a numpy array of shape (N_samples,) for labels
class SpikingDataset:

    __slots__ = ('shape', 'data', 'labels', 'max_time')

    @classmethod
    def from_file(cls, file, suffle=False):
        dataset_dict = np.load(file, allow_pickle=True).item()
        dataset = cls()
        dataset.shape = dataset_dict["shape"]
        dataset.max_time = dataset_dict["max_time"] if "max_time" in dataset_dict else 1 # default max time if not stored
        if suffle: inds_order = np.random.permutation(dataset.shape[0])
        else: inds_order = np.arange(dataset.shape[0])
        dataset.labels = dataset_dict["labels"][inds_order]
        dataset.data = [] 
        data = dataset_dict["data"]
        for ind in inds_order:
            dataset.data.append(np.rec.array(data[ind], dtype=DATA_TYPE))
        return dataset


    # Create a SpikingDataset from a dense numpy dataset of shape (N_samples, D1, ..., DN)
    # Should be done a single time, loading a SpikingDataset from file is more efficient
    @classmethod
    def from_numpy(cls, numpy_data, numpy_labels, max_time=1):        
        dataset = cls()

        # Flatten data (supports only fully-connected architectures)
        if len(numpy_data.shape) > 2: numpy_data = numpy_data.reshape(numpy_data.shape[0], -1)
        dataset.shape = numpy_data.shape

        # Map numpy data to a list of structured numpy array with sorted timestamps
        dataset.data = []
        for sample in numpy_data:
            # Sort by timestamps
            sorted_inds = sample.argsort()
            sorted_timestamps = sample[sorted_inds]
            # Mask the values not considered in the sparse representation
            # (i.e. if their firing timestamp is higher than the maximum firing time)
            sparse_mask = sorted_timestamps < max_time
            # Create a structured numpy array and apply the mask
            sample = np.recarray(np.count_nonzero(sparse_mask), dtype=DATA_TYPE)
            sample.indices = sorted_inds[sparse_mask]
            sample.timestamps = sorted_timestamps[sparse_mask]
            dataset.data.append(sample)

        # Map numpy labels to numpy array of integers ranging from 0 to N
        numpy_labels = np.searchsorted(np.unique(numpy_labels), numpy_labels).astype(np.int32)
        dataset.labels = numpy_labels
        
        dataset.max_time = max_time

        return dataset


    def save(self, file_path):
        np.save(file_path, {"data": self.data, "labels": self.labels, "shape": self.shape, "max_time": self.max_time})


    def to_dense_numpy(self, to_intensity=True):
        out = np.zeros(self.shape, dtype=np.float32)
        for i,sample in enumerate(self.data):
            dense = np.ones(self.shape[1], dtype=np.float32) * (1 - int(to_intensity)) * self.max_time
            for (indice,timestamp) in sample:
                dense[indice] = (self.max_time - timestamp) if to_intensity else timestamp
            out[i] = dense
        return out, self.labels


    def __iter__(self):
       return SpikingDatasetIterator(self)
