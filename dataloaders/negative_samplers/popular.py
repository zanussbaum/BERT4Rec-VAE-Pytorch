import numpy as np

from .base import AbstractNegativeSampler

from tqdm import trange

from collections import Counter


class PopularNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'popular'

    def generate_negative_samples(self):
        probabilities = self.items_by_popularity()

        negative_samples = {}
        print('Sampling negative items')
        for user in trange(self.user_count):
            seen = set(self.train[user])
            seen.update(self.val[user])
            seen.update(self.test[user])

            samples = []
            
            while len(samples) < self.sample_size:
                sampled_ids = np.random.choice(len(probabilities), 101, replace=False, p=probabilities)
                sampled_ids = [x for x in sampled_ids if x not in seen]
                samples.extend(sampled_ids)
                
            samples = samples[-self.sample_size:] 

            negative_samples[user] = samples

        return negative_samples

    def items_by_popularity(self):
        popularity = Counter()
        for user in range(self.user_count):
            popularity.update(self.train[user])
            popularity.update(self.val[user])
            popularity.update(self.test[user])

        counts = popularity.values()
        total = sum(counts)
        probabilities = [popularity.get(i)/total for i in range(len(popularity))]
        return probabilities
