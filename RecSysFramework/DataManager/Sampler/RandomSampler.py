import numpy as np
import scipy.sparse as sps

from RecSysFramework.DataManager import Dataset
from RecSysFramework.Utils import IncrementalSparseMatrix

from .Sampler import Sampler

class RandomSampler(Sampler):

    SamplerName = "RandomSampler"

    def __init__(self, sample_percentage=0.5, allow_cold_users=False, test_rating_threshold=0):
        super(RandomSampler, self).__init__(sample_percentage=sample_percentage, allow_cold_users=allow_cold_users,)
        self.test_rating_threshold = test_rating_threshold

    def get_name(self):
        return "random_sampler_{:.2f}_testtreshold_{:.1f}{}" \
               .format(self.sample_percentage, self.test_rating_threshold,
                       "" if self.allow_cold_users else "_no_cold_users")

    def sample_dataset(self, dataset, random_seed=42):

        super(RandomSampler, self).sample_dataset(dataset, random_seed=random_seed)

        print("RandomSampler: Sampling {:.2f}% of data".format(self.sample_percentage * 100))

        URM = sps.csr_matrix(dataset.get_URM())

        n_users, n_items = dataset.n_users, dataset.n_items
        user_indices = []
        URM_keep = {}

        # Select apriori how to randomly sort every user
        users_to_remove = []
        for user_id in range(n_users):
            assignment = np.random.choice(2, URM.indptr[user_id + 1] - URM.indptr[user_id], replace=True,
                                          p=[self.sample_percentage, (1 - self.sample_percentage)])
            assignments = [assignment == 0]

            if not self.allow_cold_users and assignments[0].sum() <= 0:
                # No interactions in the dataset kept
                users_to_remove.append(user_id)
            user_indices.append(assignments)

        for URM_name in dataset.get_URM_names():

            URM = dataset.get_URM(URM_name)
            URM = sps.csr_matrix(URM)

            URM_keep_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows=n_users,
                                                        auto_create_col_mapper=False, n_cols=n_items)

            users_to_remove_index = 0
            for user_id in range(n_users):

                if users_to_remove_index < len(users_to_remove) and user_id == users_to_remove[users_to_remove_index]:
                    users_to_remove_index += 1
                    continue

                indices = user_indices[user_id]

                start_user_position = URM.indptr[user_id]
                end_user_position = URM.indptr[user_id + 1]

                user_interaction_items = URM.indices[start_user_position:end_user_position]
                user_interaction_data = URM.data[start_user_position:end_user_position]

                # interactions to keep
                user_interaction_items_keep = user_interaction_items[indices[0]]
                user_interaction_data_keep = user_interaction_data[indices[0]]

                URM_keep_builder.add_data_lists([user_id] * len(user_interaction_items_keep),
                                                 user_interaction_items_keep, user_interaction_data_keep)

            URM_keep[URM_name] = URM_keep_builder.get_SparseMatrix()

        keep = Dataset(dataset.get_name(), base_folder=dataset.get_base_folder(),
                        postprocessings=dataset.get_postprocessings(),
                        URM_dict=URM_keep, URM_mappers_dict=dataset.get_URM_mappers_dict(),
                        ICM_dict=dataset.get_ICM_dict(), ICM_mappers_dict=dataset.get_ICM_mappers_dict(),
                        UCM_dict=dataset.get_UCM_dict(), UCM_mappers_dict=dataset.get_UCM_mappers_dict())
        keep.remove_users(users_to_remove)

        return keep
