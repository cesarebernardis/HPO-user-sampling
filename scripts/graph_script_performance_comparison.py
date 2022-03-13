import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

current_path = os.getcwd()
resultsPath = os.path.join(current_path, '../results/RandomUserSampler/performanceComparison')

for file in os.listdir(resultsPath):
    filePath = os.path.join(resultsPath, file)

    # set top and bot value equal to zero before each file
    top_value = 0
    bot_value = 1

    with open(filePath, mode='r') as csv_file:
        original_data = pd.read_csv(csv_file)

    # extrapolate main information from data frame
    seed_list = original_data['Seeds']
    feature_list = original_data['MAP@10']
    percentage_list = original_data['Percentage']

    all_metrics = {'ranking_metrics': ['Percentage', 'Seeds', 'MAP@10', 'NDCG@10', 'Precision@10', 'Recall@10'],
                   'time_metrics': ['Percentage', 'Seeds', 'time_on_train_total', 'time_on_validation_total']}

    for key, current_metrics in all_metrics.items():

        # metrics currently analysed
        selected_data = original_data[current_metrics]

        # ignore percentage and seed since they are not metrics
        for metric in current_metrics[2:]:

            plot_data = []
            for percentage in np.unique(percentage_list.tolist()):

                plot_data.append(selected_data[selected_data['Percentage'] == percentage][metric].tolist())

            # ERROR BAR using std
            mean_vector = []
            std_vector = []
            for i in plot_data:
                mean_vector.append(np.mean(i))
                std_vector.append(np.std(i))

            # cast unique percentages to string to create equal distance between ticks
            unique_percantages = np.unique(percentage_list.tolist())
            x = ["".join(item) for item in unique_percantages.astype(str)]

            # add errorbar to plot
            plt.errorbar(x=x, y=mean_vector, yerr=std_vector,
                         uplims=False, lolims=False, label=metric, fmt='--o')

            # save top and bot value to set y range
            current_top = max(mean_vector)
            current_bot = min(mean_vector)
            if current_top > top_value:
                top_value = current_top
            if current_bot < bot_value:
                bot_value = current_bot

        # use only half of the unique_percentages -> redo ffs
        ax = x[1::2]
        labels = unique_percantages * 100
        labels = labels[1::2]
        # for some reason percentage 7% is broken so i just round it
        labels = np.around(labels)
        labels = labels.astype(int)
        plt.xticks(ax, labels)
        # set y range
        range = top_value - bot_value
        plt.ylim([bot_value - 0.15 * range, top_value + 0.15 * range])
        # add legend
        plt.legend()
        # naming the x axis
        plt.xlabel('PERCENTAGES')
        # naming the y axis
        if key == 'ranking_metrics':
            plt.ylabel(" ")
        else:
            plt.ylabel("SECONDS")
        # add title
        title = file.split('.csv')[0]
        plt.title(title)

        #
        # save the plot
        #

        temp = title.split('_')
        cwd = os.getcwd()
        directory = os.path.join(cwd, '../plots/' + temp[2] + '/' + temp[3] + '/performanceComparison')

        # create directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(os.path.join(directory))

        plt.savefig(directory + '/' + key + '.png')

        # show the plot
        # plt.show()

        plt.close()

        print("Done with file " + file)
