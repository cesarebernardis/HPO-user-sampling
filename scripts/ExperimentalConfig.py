from RecSysFramework.DataManager.Reader import Movielens100KReader
from RecSysFramework.DataManager.Reader import LastFMHetrec2011Reader
from RecSysFramework.DataManager.Reader import Movielens10MReader
from RecSysFramework.DataManager.Reader.EpinionsReader import EpinionsReader

from RecSysFramework.DataManager.Splitter import Holdout
from RecSysFramework.DataManager.Sampler import RandomSampler
from RecSysFramework.DataManager.Sampler import RandomUserSampler
from RecSysFramework.DataManager.DatasetPostprocessing import URMKCore, ImplicitURM

from RecSysFramework.Recommender.KNN.ItemKNNCFRecommender import ItemKNNCF
from RecSysFramework.Recommender.KNN.UserKNNCFRecommender import UserKNNCF
from RecSysFramework.Recommender.MatrixFactorization.PureSVD import PureSVD
from RecSysFramework.Recommender.GraphBased.RP3betaRecommender import RP3beta
from RecSysFramework.Recommender.KNN.EASE_R_Recommender import EASE_R

from RecSysFramework.Evaluation import EvaluatorHoldout

EXPERIMENTAL_CONFIG = {
    'splits': [
        Holdout(train_perc=0.8, test_perc=0.2),
    ],
    'sampler': [
        # RandomSampler,
        RandomUserSampler
    ],
    'datasets':
        # Movielens100KReader
        LastFMHetrec2011Reader
        # EpinionsReader
        # Movielens10MReader
    ,
    'postprocessings': [
        ImplicitURM(min_rating_threshold=3.),
        URMKCore(user_k_core=5, item_k_core=5, reshape=True),
    ],
    'recommenders':
        ItemKNNCF
        # UserKNNCF
        # RP3beta
        # PureSVD
        # EASE_R
    ,
    'recap_metrics': ["Precision", "Recall", "MAP", "NDCG"],
    'cutoffs': [10],
    'evaluators': [
        EvaluatorHoldout([10], exclude_seen=True)
    ],
    'sample_percentages': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    'random_seed': [1, 2, 3],
    'starting_seed': [1, 1, 1],
    'metrics_to_optimize': ["Precision", "Recall", "MAP", "NDCG"],
    'cutoff_to_optimize': 10,
    'metrics_to_optimize_early_stopping': "MAP",
    'fixed_keyword_args': {}  # "similarity": "cosine"}
}
