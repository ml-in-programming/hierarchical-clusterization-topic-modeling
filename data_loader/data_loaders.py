import os
from collections import Counter

import pandas as pd
import torch
import numpy as np

class TokenDataProducer:
    def __init__(self, n_tokens: int, batch_size: int, num_neg_samples: int):
        self.token_paths = None
        self.n_tokens = n_tokens
        self.token2id = None
        self.token_counts = None
        self.num_neg_samples = num_neg_samples
        self.batch_size = batch_size
        self.p_distr = None

    def set_initial_information(self, token_paths: np.ndarray, token2id: dict, all_token_counts: Counter):
        self.token_paths = token_paths
        self.token2id = token2id
        self.token_counts = np.zeros(self.n_tokens, dtype=np.int32)
        for token, count in all_token_counts.items():
            ind = token2id[token]
            if ind < self.n_tokens:
                self.token_counts[ind] = count

        self.p_distr = self.get_distribution(self.token_counts)

    @staticmethod
    def get_distribution(token_counts: np.ndarray):
        word_counts = torch.FloatTensor(token_counts)
        distr = word_counts / word_counts.sum()
        distr = distr.pow(3 / 4)
        distr = distr / distr.sum()
        return distr.numpy()

    def neg_sample(self, batch_size: int):
        return torch.tensor(np.random.choice(self.n_tokens, size=(batch_size, self.num_neg_samples), p=self.p_distr))

    def generate_sample(self, token_paths):
        while True:
            for token_path in token_paths:
                if os.stat(token_path).st_size == 0:
                    continue
                token_counts = pd.read_csv(token_path)
                token_indices = [self.token2id[token] for token in token_counts.token]
                for center in token_indices:
                    for target in token_indices:
                        if center != target:
                            yield center, target

    def get_batch(self, iterator, batch_size):
        """ Group a numerical stream into batches and yield them as Numpy arrays. """
        while True:
            center_batch = torch.zeros(batch_size, dtype=torch.long)
            target_batch = torch.zeros(batch_size, dtype=torch.long)
            for index in range(batch_size):
                center_batch[index], target_batch[index] = next(iterator)
            neg_batch = self.neg_sample(batch_size)
            yield center_batch, target_batch, neg_batch

    def batch_iterator(self):
        single_gen = self.generate_sample(self.token_paths)
        batch_gen = self.get_batch(single_gen, batch_size=self.batch_size)
        return batch_gen
