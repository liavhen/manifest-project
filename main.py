import utils
from manifest import *
from utils import *
from adaboost import AdaBoost
from ReliefF import ReliefF
import logging, sys

def run_adaboost(data, adaboost_timesteps, weak_learners):
    adaboost = AdaBoost(data, T=adaboost_timesteps, weak_learners=weak_learners)
    adaboost.train()
    train_error, test_error = adaboost.get_losses()
    return train_error, test_error


# def run_adaboost(data, adaboost_timesteps, weak_learners):
#     adaboost = AdaBoostClassifier(n_estimators=len(weak_learners))
#     model = adaboost.fit()

def apply_relief(data):
    n_neighbors = min(data['train_samples'].shape[0] - 1, 100)
    n_features_to_keep = data['train_samples'].shape[0] # keep all features (they will later be selected)
    relief = ReliefF(n_neighbors=n_neighbors, n_features_to_keep=n_features_to_keep)
    relief.fit(data['train_samples'], data['train_labels'])
    scores = -relief.feature_scores.copy()  # take negative of this implementation - the higher the score, the better
    features = np.argsort(scores)[::-1]
    return scores, features


def run_experiment(logger, adaboost_timesteps=30, nof_selected_features=20, nof_samples=2000, class_balance=0.5):
    # results = dict()
    data = utils.load_data(
        classes=(0, 8),
        nof_samples_per_class=(int(nof_samples*class_balance), int(nof_samples*class_balance))
    )
    manifest_scores, manifest_features, _ = ManiFeSt(data['train_samples'], data['train_labels'])
    relief_scores, relief_features = apply_relief(data)

    # Normalize scores to the same scales
    manifest_scores = normalize(manifest_scores)
    relief_scores = normalize(relief_scores)

    # Collect features
    manifest_top_features = manifest_features[:nof_selected_features]
    relief_top_features = relief_features[:nof_selected_features]

    # Show Scores
    utils.show_scores(
        manifest_scores, 'ManiFeSt Score',
        relief_scores, 'ReliefF Score',
        nof_features=nof_selected_features, nof_samples=nof_samples
    )

    manifest_weak_learners = ([above_threshold_classifier(j, 128) for j in manifest_top_features]
                              + [below_threshold_classifier(j, 128) for j in manifest_top_features])
    relief_weak_learners = ([above_threshold_classifier(j, 128) for j in relief_top_features]
                            + [below_threshold_classifier(j, 128) for j in relief_top_features])
    random_weak_learners = ([above_threshold_classifier(j, 128) for j in np.random.randint(0, 28 * 28, nof_selected_features)]
                            + [below_threshold_classifier(j, 128) for j in np.random.randint(0, 28 * 28, nof_selected_features)])

    manifest_train_error, manifest_test_error = run_adaboost(data, adaboost_timesteps, manifest_weak_learners)
    relief_train_error, relief_test_error = run_adaboost(data, adaboost_timesteps, relief_weak_learners)
    random_train_error, random_test_error = run_adaboost(data, adaboost_timesteps, random_weak_learners)

    utils.plot_errors(manifest_train_error, manifest_test_error, 'ManiFeSt',
                      adaboost_timesteps, nof_features=nof_selected_features, nof_samples=nof_samples)
    utils.plot_errors(relief_train_error, relief_test_error, 'ReliefF',
                      adaboost_timesteps, nof_features=nof_selected_features, nof_samples=nof_samples)
    utils.plot_errors(random_train_error, random_test_error, 'Random',
                      adaboost_timesteps, nof_features=nof_selected_features, nof_samples=nof_samples)
    return manifest_test_error, relief_test_error, random_test_error


def main():
    adaboost_timesteps = 30
    features_samples_ratios = [2, 5, 10, 50]
    nof_selected_features_s = [5, 50, 200]

    logging.basicConfig(
        filename=f'plots/run.log', filemode='w',
        format='%(asctime)s %(levelname)s --> %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))

    for features_samples_ratio in features_samples_ratios:
        for nof_selected_features in nof_selected_features_s:

            nof_samples = nof_selected_features * features_samples_ratio

            manifest_test_error, relief_test_error, random_test_error = run_experiment(
                logger,
                adaboost_timesteps=adaboost_timesteps,
                nof_selected_features=nof_selected_features,
                nof_samples=nof_samples
            )
            logger.info('=================================================================')
            logger.info(f'\nResults | {nof_samples} Samples, {nof_selected_features} Weak Learners:')
            logger.info('=================================================================')
            logger.info(f'Random ManiFest Error: {manifest_test_error[-1]}')
            logger.info(f'Random ReliefF Error: {relief_test_error[-1]}')
            logger.info(f'Random AdaBoost Error: {random_test_error[-1]}')
            logger.info('=================================================================')


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
