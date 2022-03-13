import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

current_path = os.getcwd()
resultsPath = os.path.join(current_path, "../results/RandomUserSampler/performanceComparison/")

files = os.listdir(resultsPath)

dataset_dict = dict()
for file in files:
    dataset_used = file.split('_')[2]

    if dataset_used in dataset_dict:
        dataset_dict[dataset_used].append(file)
    else:
        dataset_dict[dataset_used] = [file]

for dataset in dataset_dict.keys():

    # set top and bot value equal to zero before each file
    top_value = 0
    bot_value = 1

    for file in dataset_dict[dataset]:
        filePath = os.path.join(resultsPath, file)
        file_name = file.split('.csv')[0]
        current_algorithm = file_name.split('_')[3]

        with open(filePath, mode='r') as csv_file:
            original_data = pd.read_csv(csv_file)

        # extrapolate main information from data frame
        seed_list = original_data['Seeds']
        feature_list = original_data['MAP@10']
        percentage_list = original_data['Percentage']

        # metrics currently analysed
        current_metrics = ['Percentage', 'Seeds', 'MAP@10']
        selected_data = original_data[current_metrics]

        plot_data = []
        for percentage in np.unique(percentage_list.tolist()):
            plot_data.append(selected_data[selected_data['Percentage'] == percentage]['MAP@10'].tolist())

        # ERROR BAR using var
        mean_vector = []
        var_vector = []
        for i in plot_data:
            mean_vector.append(np.mean(i))
            var_vector.append(np.var(i))

        # cast unique percentages to string to create equal distance between ticks
        unique_percantages = np.unique(percentage_list.tolist())
        x = ["".join(item) for item in unique_percantages.astype(str)]

        # add errorbar to plot
        plt.errorbar(x=x, y=mean_vector, yerr=var_vector,
                     uplims=False, lolims=False, label=current_algorithm, fmt='--o')

        # save top and bot value to set y range
        current_top = max(mean_vector)
        current_bot = min(mean_vector)
        if current_top > top_value:
            top_value = current_top
        if current_bot < bot_value:
            bot_value = current_bot

    # use only half of the unique_percentages and set x ticks
    del x[1::2]
    plt.xticks(x)
    # set y range
    range = top_value - bot_value
    plt.ylim([bot_value - 0.15 * range, top_value + 0.15 * range])
    # add legend
    plt.legend()
    # naming the x axis
    plt.xlabel('PERCENTAGES')
    # naming the y axis
    plt.ylabel("MAP@10")
    # add title
    title = dataset + '_MAP@10'
    plt.title(title)

    #
    # save the plot
    #

    cwd = os.getcwd()
    directory = os.path.join(cwd, '../plots/' + dataset)

    # create directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(os.path.join(directory))

    # save plot
    plt.savefig(directory + '/algorithm_MAP_comparison_' + dataset + '.png')

    # show the plot
    # plt.show()

    plt.close()

    print("Done with " + dataset)
