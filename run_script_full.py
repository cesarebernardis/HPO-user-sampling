import os
import csv
import sys

from RecSysFramework.DataManager.Reader import Movielens100KReader
from RecSysFramework.DataManager.Reader import LastFMHetrec2011Reader
from RecSysFramework.DataManager.Reader import Movielens10MReader
from RecSysFramework.DataManager.Reader.EpinionsReader import EpinionsReader

from RecSysFramework.Recommender.KNN.ItemKNNCFRecommender import ItemKNNCF
from RecSysFramework.Recommender.KNN.UserKNNCFRecommender import UserKNNCF
from RecSysFramework.Recommender.MatrixFactorization.PureSVD import PureSVD
from RecSysFramework.Recommender.GraphBased.RP3betaRecommender import RP3beta
from RecSysFramework.Recommender.KNN.EASE_R_Recommender import EASE_R

from RecSysFramework.ParameterTuning.Utils import run_parameter_search
from RecSysFramework.Recommender.DataIO import DataIO
from scripts.ExperimentalConfig import EXPERIMENTAL_CONFIG

if __name__ == '__main__':

    # initiate data readers and data pre processing
    datareader = getattr(sys.modules[__name__], sys.argv[1])()
    postprocessings = EXPERIMENTAL_CONFIG['postprocessings']

    # initiate datasets and apply postprocessing
    dataset = datareader.load_data(postprocessings=postprocessings)
    dataset.save_data()

    # random seeds
    random_seed = EXPERIMENTAL_CONFIG['random_seed']

    #
    # initial run to store sampled dataset
    #

    for seed_test in random_seed:

        # initiate splitter and split train/test
        splitter = EXPERIMENTAL_CONFIG['splits'][0]
        train, test = splitter.split(dataset, random_seed=seed_test)

        # save split, everything must be replicable
        splitter.save_split([train, test],
                            filename_suffix="_seed_sample=" + str(seed_test))

        for seed_sample in random_seed:
            for seed_validation in random_seed:

                for sample_percentage in EXPERIMENTAL_CONFIG['sample_percentages']:

                    # initiate sampler
                    sampler = EXPERIMENTAL_CONFIG['sampler'][0](sample_percentage=sample_percentage)

                    # sample dataset and create optimization dataset
                    optimization_dataset = sampler.sample_dataset(train, random_seed=seed_sample)
                    optimization_train, optimization_validation = splitter.split(optimization_dataset,
                                                                                 random_seed=seed_validation)

                    # save splits, everything must be replicable
                    sampler.save_split([optimization_dataset, optimization_train, optimization_validation],
                                       filename_suffix="_seed_sample=" + str(seed_sample)
                                                       + "_seed_validation=" + str(seed_validation))

        # check which sampler you are using
        sampler_name = sampler.SamplerName

        # create directory to store results
        cwd = os.getcwd()
        directory_pc = os.path.join(cwd, 'results/' + sampler_name + '/performanceComparison')
        # create directory if it does not exist
        if not os.path.exists(directory_pc):
            os.makedirs(os.path.join(directory_pc))

        # create directory to store results
        cwd = os.getcwd()
        directory_hpr = os.path.join(cwd, 'results/' + sampler_name + '/hpRanking')
        # create directory if it does not exist
        if not os.path.exists(directory_hpr):
            os.makedirs(os.path.join(directory_hpr))

        recommenders = getattr(sys.modules[__name__], sys.argv[2])

    #
    # find performances
    #
    print("Start second phase")

    for sample_percentage in EXPERIMENTAL_CONFIG['sample_percentages']:

        for seed_test in random_seed:

            # evaluation datasets -> only train and test needed
            splitter = EXPERIMENTAL_CONFIG['splits'][0]
            train, test = splitter.load_split(datareader, postprocessings=postprocessings,
                                              filename_suffix="_seed_sample=" + str(seed_test))

            for seed_sample in random_seed:

                for seed_validation in random_seed:

                    # skip seeds combination BEFORE starting_seed for the starting sample_percentage
                    if [seed_test, seed_sample, seed_validation] >= EXPERIMENTAL_CONFIG['starting_seed']\
                            or sample_percentage != EXPERIMENTAL_CONFIG['sample_percentages'][0]:

                        # initiate sampler with current sample_percentage
                        sampler = EXPERIMENTAL_CONFIG['sampler'][0](sample_percentage=sample_percentage)

                        # load optimization dataset
                        optimization_dataset, optimization_train, optimization_validation = \
                            sampler.load_split(datareader, postprocessings=postprocessings,
                                               filename_suffix="_seed_sample=" + str(seed_sample) +
                                                               "_seed_validation=" + str(seed_validation))

                        # hyper parameter optimization settings
                        fixed_keyword_args = EXPERIMENTAL_CONFIG["fixed_keyword_args"]
                        metrics_to_optimize = EXPERIMENTAL_CONFIG["metrics_to_optimize"]
                        output_folder_path = train.get_complete_folder() + \
                                             os.path.join(recommenders.RECOMMENDER_NAME) + os.sep

                        #
                        # first part of the script
                        # try all hp configurations and store the results obtained on the validation validation set
                        #

                        # hp optimization -> 50 random walks, seed set to 9000
                        run_parameter_search(
                            recommenders, splitter.get_name(), optimization_train, optimization_validation,
                            output_folder_path=output_folder_path,
                            metric_to_optimize=metrics_to_optimize,
                            cutoff_to_optimize=EXPERIMENTAL_CONFIG["cutoff_to_optimize"],
                            save_model="no",
                            n_cases=50, n_random_starts=50, fixed_keyword_args=fixed_keyword_args,
                            random_state=9000,
                            metric_to_optimize_early_stopping=EXPERIMENTAL_CONFIG["metrics_to_optimize_early_stopping"]
                        )

                        # recover best hyper parameters configurations
                        input_folder_path = train.get_complete_folder() + \
                                            os.path.join(recommenders.RECOMMENDER_NAME) + os.sep
                        filename = recommenders.RECOMMENDER_NAME + "_metadata"
                        dataIO = DataIO(folder_path=input_folder_path)
                        data_dict = dataIO.load_data(file_name=filename)

                        # for each of the hyper parameters found in the random walk
                        for i in range(50):

                            current_hyper_parameters = data_dict["hyperparameters_list"][i]

                            #
                            # write results into a csv file
                            #

                            # file path hp ranking
                            completePath = os.path.join(directory_hpr, 'HP_ranking_' + datareader.get_dataset_name()
                                                        + '_' + recommenders.RECOMMENDER_NAME + '.csv')

                            with open(completePath, mode='a', newline='') as file1:
                                employee_writer = csv.writer(file1, delimiter=',', quotechar='"',
                                                             quoting=csv.QUOTE_MINIMAL)

                                # initialise field names
                                fieldnames = ['Percentage', 'Seeds', 'HP']

                                # initialise dictionary to write on the file
                                results_dict = data_dict["result_on_validation_list"][i]
                                formatted_results_dict = {'Percentage': str(sample_percentage),
                                                          'Seeds': [seed_test, seed_sample, seed_validation],
                                                          'HP': current_hyper_parameters}

                                # update dictionary with results
                                for key, sub_dict in results_dict.items():
                                    for sub_key, value in sub_dict.items():
                                        fieldnames.append(sub_key + '@' + str(key))
                                        formatted_results_dict.update({sub_key + '@' + str(key): value})

                                # initialise writer
                                writer = csv.DictWriter(file1, fieldnames=fieldnames)

                                # check file is not empty and adds fields
                                if os.stat(completePath).st_size == 0:
                                    writer.writeheader()

                                # write results row
                                writer.writerow(formatted_results_dict)

                            file1.close()

                        #
                        # second half of the script
                        # try BEST hp configurations and store the results obtained on the test set
                        print("try BEST HP configs")
                        #

                        # initiate recommender with COMPLETE URM train and fit with BEST HP
                        recommender = recommenders(train.get_URM())
                        recommender.fit(**data_dict["hyperparameters_best"])

                        # evaluation
                        evaluator = EXPERIMENTAL_CONFIG['evaluators'][0]
                        metric_handler = evaluator.evaluateRecommender(recommender, URM_test=test.get_URM())

                        print("\nRecommender -> ", recommender.RECOMMENDER_NAME, "\nDataset -> ",
                              datareader.get_dataset_name(),
                              "\nSample percentage -> ", sample_percentage,
                              "\nSeed_test -> ", seed_test,
                              "\nSeed_sample -> ", seed_sample,
                              "\nSeed_validation -> ", seed_validation)
                        print("\nOriginal:\n" + metric_handler.get_results_string())

                        #
                        # write results into a csv file
                        #

                        # file path performance comparison
                        completePath = os.path.join(directory_pc, 'performance_comparison_' +
                                                    datareader.get_dataset_name() + "_" +
                                                    recommender.RECOMMENDER_NAME + '.csv')

                        # kill tf space
                        cs_op = getattr(recommender, "clear_session", None)
                        if cs_op is not None and callable(cs_op):
                            recommender.clear_session()
                        del recommender

                        with open(completePath, mode='a', newline='') as file2:
                            employee_writer = csv.writer(file2, delimiter=',', quotechar='"',
                                                         quoting=csv.QUOTE_MINIMAL)

                            # initialise field names
                            fieldnames = ['Percentage', 'Seeds', 'time_on_train_total', 'time_on_train_avg',
                                          'time_on_validation_total', 'time_on_validation_avg']

                            # initialise dictionary to write on the file
                            results_dict = metric_handler.get_results_dictionary()
                            formatted_results_dict = {'Percentage': str(sample_percentage),
                                                      'Seeds': [seed_test, seed_sample, seed_validation],
                                                      'time_on_train_total': data_dict['time_on_train_total'],
                                                      'time_on_train_avg': data_dict['time_on_train_avg'],
                                                      'time_on_validation_total': data_dict['time_on_validation_total'],
                                                      'time_on_validation_avg': data_dict['time_on_validation_avg']}

                            # create new dictionary with sample percentage, seed and results
                            for key, sub_dict in results_dict.items():
                                for sub_key, value in sub_dict.items():
                                    fieldnames.append(sub_key + '@' + str(key))
                                    formatted_results_dict.update({sub_key + '@' + str(key): value})

                            # initialise writer
                            writer = csv.DictWriter(file2, fieldnames=fieldnames)

                            # check file is not empty and adds fields
                            if os.stat(completePath).st_size == 0:
                                writer.writeheader()

                            # write results row
                            writer.writerow(formatted_results_dict)

                        file2.close()

    print("### END OF THE SCRIPT ###")
