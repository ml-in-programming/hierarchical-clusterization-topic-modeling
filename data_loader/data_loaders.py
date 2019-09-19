import pandas as pd
import torch
import numpy as np

class DataProducer:
    def __init__(self, token_paths, n_tokens, token2id, token_counts, batch_size, num_neg_samples):
        self.token_paths = token_paths
        self.n_tokens = n_tokens
        self.token2id = token2id
        self.num_neg_samples = num_neg_samples
        self.batch_size = batch_size
        self.p_distr = self.get_distribution(token_counts)

    @staticmethod
    def get_distribution(token_counts):
        word_counts = torch.FloatTensor(token_counts)
        distr = word_counts / word_counts.sum()
        distr = distr.pow(3 / 4)
        distr = distr / distr.sum()
        return distr.numpy()

    def neg_sample(self, batch_size):
        return np.random.choice(self.n_tokens, size=(batch_size, self.num_neg_samples), p=self.p_distr)

    def generate_sample(self, token_paths):
        for token_path in token_paths:
            token_counts = pd.read_csv(token_path)
            token_indices = [self.token2id[token] for token in token_counts.token]
            for center in token_indices:
                for target in token_indices:
                    if center != target:
                        yield center, target

    def get_batch(self, iterator, batch_size):
        """ Group a numerical stream into batches and yield them as Numpy arrays. """
        while True:
            center_batch = np.zeros(batch_size, dtype=np.int32)
            target_batch = np.zeros(batch_size, dtype=np.int32)
            for index in range(batch_size):
                center_batch[index], target_batch[index] = next(iterator)
            yield center_batch, target_batch

    def batch_iterator(self):
        single_gen = self.generate_sample(self.token_paths)
        batch_gen = self.get_batch(single_gen, batch_size=self.batch_size)
        return batch_gen
