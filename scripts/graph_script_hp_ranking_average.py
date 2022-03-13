import matplotlib.pyplot as plt
import os, csv
import numpy as np
import pandas as pd
import plotly
import plotly.figure_factory as ff

current_path = os.getcwd()
resultsPath = os.path.join(current_path, '../results/RandomUserSampler/hpRanking/')

for file in os.listdir(resultsPath):

    filePath = os.path.join(resultsPath, file)
    rbo_stats = dict()

    # create data frame
    with open(filePath, mode='r') as csv_file:
        original_data = pd.read_csv(csv_file)

    # extrapolate main information from data frame
    seed_list = original_data['Seeds']
    feature_list = original_data['MAP@10']
    percentage_list = original_data['Percentage']
    hp_list = original_data['HP']

    # currently working with MAP@10
    selected_data = original_data[['Percentage', 'Seeds', 'HP', 'MAP@10']]

    # substitute HP with C#
    j = 0
    for i in np.unique(selected_data['HP'].tolist()):
        selected_data['HP'] = selected_data['HP'].replace([i], 'C' + str(j))
        j += 1

    values_to_plot, names_to_plot, best_index = [], [], []
    value_scale = dict()

    # different plot for each seed
    for percentage in np.unique(percentage_list):

        # filter by percentage
        data_filtered_by_percentage = selected_data.loc[selected_data['Percentage'] == percentage]

        # average
        data_filtered_by_percentage_averaged = data_filtered_by_percentage\
            .groupby(['Percentage', 'HP'], as_index=False).mean()

        # order HP ranking by the feature chosen
        data_filtered_by_percentage_and_hp_config_ordered = data_filtered_by_percentage_averaged\
            .sort_values(by=['MAP@10'], ascending=False)

        # select values to highlight in the plot
        if percentage == 1:

            base_value = 1
            best_index = data_filtered_by_percentage_and_hp_config_ordered['HP'].tolist()

            for index in best_index:
                base_value -= 1 / len(best_index)
                value_scale.update({index: base_value})

        # merge value and index ranking from different percentages
        names_to_plot.append(data_filtered_by_percentage_and_hp_config_ordered['HP'].tolist())
        values_to_plot.append(data_filtered_by_percentage_and_hp_config_ordered['MAP@10'].tolist())

    # over write values to highlight how the 5 "best" parameters behave
    '''for i in range(len(values_to_plot)):
        for j in range(len(values_to_plot[i])):

            # 1 if hp configuration is optimal, 0 otherwise
            if names_to_plot[i][j] in best_index[0:8]:
                values_to_plot[i][j] = 1
            else:
                values_to_plot[i][j] = 0'''

    # colour scale for results
    for i in range(len(names_to_plot)):
        for j in range(len(names_to_plot[i])):
            values_to_plot[i][j] = value_scale.get(names_to_plot[i][j])

    # preparing x,y label
    x = list(range(0, len(np.unique(hp_list))))
    x = [str(i + 1) + '°' for i in x]
    y = np.unique(list(original_data['Percentage'])).tolist()
    y = [('Percentage: ' + str(round(i * 100)) + '%') for i in y]

    # plot data
    fig = ff.create_annotated_heatmap(values_to_plot, x=x, y=y, colorscale='YlGnBu',
                                      annotation_text=names_to_plot)

    from plotly.graph_objs.layout import Font
    from plotly.graph_objs.layout import XAxis, YAxis
    from plotly.graph_objs.layout.xaxis import Tickfont as fontx
    from plotly.graph_objs.layout.yaxis import Tickfont as fonty

    import plotly.graph_objects as go
    fig.layout.update(
        go.Layout(
            autosize=False,
            yaxis=YAxis(tickfont=fonty(
                size=18
            )),
            xaxis=XAxis(tickfont=fontx(
                size=16
            ))
        )
    )

    # add title
    title = file.split('.csv')[0] + '_average'
    fig.update_layout(title_text=title)

    # show the plot
    fig.show()

    #
    # save the plot
    #

    temp = title.split('_')
    cwd = os.getcwd()
    directory = os.path.join(cwd, '../plots/RandomUserSampler/' + temp[2] + '/' + temp[3] + '/HpRanking')

    # create directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    if len(y) > 10:
        plotly.io.write_image(fig, file=os.path.join(directory, 'ranking_average.png'), width=2000, height=800)
    else:
        plotly.io.write_image(fig, file=os.path.join(directory, 'ranking_average.png'), width=2000, height=600)

    print("Done with file " + file)
