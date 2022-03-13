# On the Impact of Data Sampling on Hyper-parameter Optimisation of Recommendation Algorithms

This project was developed by Matteo Montanari and [Cesare Bernardis](https://scholar.google.it/citations?user=9fzJj_AAAAAJ), Ph.D. candidate at Politecnico di Milano. It is based on the recommendation framework used by the [RecSys@Polimi research group](https://recsys.deib.polimi.it/). 
The code allows reproducing the results of "On the Impact of Data Sampling on Hyper-parameter
Optimisation of Recommendation Algorithms", published on the 37th ACM/SIGAPP Symposium On Applied Computing.

An article on this work will soon be published!

## Installation

The setup to run this experiment can be accomplished in one of two ways: 
by using docker (A) or by working directly on your machine (B).

### (A) Docker setup

Install docker on your local machine.

Open a terminal and move into the root folder containing the Dockerfile.

Afterwards run the following code to create a Docker image:

```python
docker build --tag hpoContaienr .
```

To then run a container from the created image, run the following code:

```python
docker run -ti -v $(pwd):/hpo hpoContaienr
```

Once the container has started, you should have a terminal open in your docker container. Install the framework by running:
```python
sh install.sh
```

Your docker container now has all the requirements to run an experiment.

### (B) Local Setup

Install python on your local machine.

Once python is installed, install the necessary packages by running:

```python
pip install Cython
pip install -r requirements.txt
```
And finally, install the framework by running:
```python
sh install.sh
```

Now, your local machine has all the requirements to run an experiment.

## Run the experiments

To run an experiment, move into the root folder of this project and run the following code:

```python
python3 run_script_full.py arg1 arg2
```
where arg1 is the dataset reader and arg2 is the algorithm.

Options regarding the dataset reader are:
	    Movielens100KReader, 
        LastFMHetrec2011Reader,
        EpinionsReader,
        Movielens10MReader (the dataset will be automatically downloaded if necessary)

Options regarding the algorithm are:
        ItemKNNCF,
        UserKNNCF,
        RP3beta,
        PureSVD,
        EASE_R

The experiments using the algorithm EASE_R requires a high amount of RAM and time to run, especially when used with a sizable dataset.

The algorithm is tested on the dataset at percentages (100%, 90%, 80%, 70%, 60%, 50%, 40%, 30%, 20%, 10%).

## Results Analysis 

Once one or more .csv file have been produced by the script above, it is possible to plot the data by using the following Python scripts, contained in the folder "scripts":

To generate a graph for each dataset in which algorithms are compared to each other, run:

```python
python3 graph_script_dataset_comparison.py
```

To generate a table with the average ranking of the 50 hyper-parameter configurations for each dataset-algorithm combination, run:

```python
python3 graph_script_hp_ranking_average.py
```

To generate a graph in which model and time performance are compared for each dataset-algorithm combination, run:

```python
python3 graph_script_performance_comparison.py
```

The generated graphs will appear in the folder "plots".
