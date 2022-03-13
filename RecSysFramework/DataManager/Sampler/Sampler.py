import numpy as np
import os

from RecSysFramework.DataManager import Dataset

class Sampler(object):

    def __init__(self, sample_percentage=0.5, allow_cold_users=False):
        super(Sampler, self).__init__()
        self.sample_percentage = sample_percentage
        self.allow_cold_users = allow_cold_users

    def get_name(self):
        pass

    def _get_dataset_names_in_split(self):
        datalist = ["optimization_dataset", "optimization_train", "optimization_validation"]
        return datalist

    def sample_dataset(self, dataset, random_seed=42):
        np.random.seed(random_seed)

    def get_default_save_folder_path(self, datareader, postprocessings=None):
        return datareader.get_complete_default_save_path(postprocessings)

    def save_split(self, split, save_folder_path=None, filename_suffix=""):

        datalist = self._get_dataset_names_in_split()
        assert len(split) == len(datalist), \
            "DataSplitter: expected {} datasets in split, given {}".format(len(datalist), len(split))

        if save_folder_path is None:
            save_folder_path = split[0].get_complete_folder() + self.get_name() + os.sep

        for i, dataset in enumerate(split):
            dataset.save_data(save_folder_path=save_folder_path,
                              filename_suffix="{}_{}".format(filename_suffix, datalist[i]))

    def load_split(self, datareader, save_folder_path=None, postprocessings=None, filename_suffix=""):

        tmp_save_folder_path = save_folder_path
        if tmp_save_folder_path is None:
            tmp_save_folder_path = self.get_default_save_folder_path(datareader, postprocessings=postprocessings)

        try:

            datalist = self._get_dataset_names_in_split()
            for i in datalist:
                if not datareader.all_files_available(tmp_save_folder_path + self.get_name() + os.sep,
                                                      filename_suffix="{}_{}".format(filename_suffix, i)):
                    raise Exception

            datasets = []
            for i in datalist:
                urm, urm_mappers, icm, icm_mappers, ucm, ucm_mappers = datareader.load_from_saved_sparse_matrix(
                    tmp_save_folder_path + self.get_name() + os.sep, filename_suffix="{}_{}".format(filename_suffix, i))
                datasets.append(Dataset(datareader.get_dataset_name(),
                                        base_folder=datareader.get_default_save_path(),
                                        postprocessings=postprocessings,
                                        URM_dict=urm, URM_mappers_dict=urm_mappers,
                                        ICM_dict=icm, ICM_mappers_dict=icm_mappers,
                                        UCM_dict=ucm, UCM_mappers_dict=ucm_mappers))

            return datasets

        except:

            print("DataSplitter: Preloaded data not found or corrupted, reading from original files...")
            dataset = datareader.load_data(save_folder_path=save_folder_path, postprocessings=postprocessings)
            return self.sample_dataset(dataset)