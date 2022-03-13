import numpy as np
import scipy.sparse as sps

from RecSysFramework.DataManager import Dataset
from RecSysFramework.Utils import IncrementalSparseMatrix

from .Sampler import Sampler

class RandomUserSampler(Sampler):

    SamplerName = "RandomUserSampler"

    def __init__(self, sample_percentage=0.5, allow_cold_users=False, test_rating_threshold=0):
        super(RandomUserSampler, self).__init__(sample_percentage=sample_percentage, allow_cold_users=allow_cold_users,)
        self.test_rating_threshold = test_rating_threshold

    def get_name(self):
        return "random_user_sampler_{:.2f}_testtreshold_{:.1f}{}" \
               .format(self.sample_percentage, self.test_rating_threshold,
                       "" if self.allow_cold_users else "_no_cold_users")

    def sample_dataset(self, dataset, random_seed=42):

        super(RandomUserSampler, self).sample_dataset(dataset, random_seed=random_seed)

        print("RandomUserSampler: Sampling {:.2f}% of users".format(self.sample_percentage * 100))
        users_to_remove = np.random.choice(dataset.n_users, int(dataset.n_users * (1. - self.sample_percentage)), replace=False)

        new_dataset = dataset.copy()
        new_dataset.remove_users(users_to_remove, keep_original_shape=False)
        # new_dataset.add_postprocessing(self)

        return new_dataset
