import utils
from manifest import *
from utils import *
from adaboost import AdaBoost
from tqdm import tqdm
import logging, sys
import pandas as pd
from relief import reliefF, feature_ranking


def run_adaboost(data, adaboost_timesteps, weak_learners):
    adaboost = AdaBoost(data, T=adaboost_timesteps, weak_learners=weak_learners)
    adaboost.train()
    train_error, test_error = adaboost.get_losses()
    return train_error, test_error


def relieff(data, nof_selected_features):
    scores = reliefF(data['train_samples'], data['train_labels'])
    features = feature_ranking(scores)[:nof_selected_features]
    return scores, features


def run_experiment(
        adaboost_timesteps=30,
        nof_selected_features=20,
        nof_samples=2000,
        save_imgs=False,
        class_balance=0.5
):

    data = utils.load_data(
        classes=(4, 9),
        nof_samples_per_class=(int(nof_samples*class_balance), int(nof_samples*class_balance))
    )
    manifest_scores, manifest_features, _ = ManiFeSt(data['train_samples'], data['train_labels'])
    relief_scores, relief_features = relieff(data, nof_selected_features)

    # Collect features
    manifest_top_features = manifest_features[:nof_selected_features]
    relief_top_features = relief_features[:nof_selected_features]

    if save_imgs:
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

    if save_imgs:
        utils.plot_errors(manifest_train_error, manifest_test_error, 'ManiFeSt',
                          adaboost_timesteps, nof_features=nof_selected_features, nof_samples=nof_samples)
        utils.plot_errors(relief_train_error, relief_test_error, 'ReliefF',
                          adaboost_timesteps, nof_features=nof_selected_features, nof_samples=nof_samples)
        utils.plot_errors(random_train_error, random_test_error, 'Random',
                          adaboost_timesteps, nof_features=nof_selected_features, nof_samples=nof_samples)

    return manifest_test_error, relief_test_error, random_test_error


def main():
    adaboost_timesteps = 30
    SAFETY_KNOB = True
    FEW_SAMPLES = True

    if SAFETY_KNOB:
        if FEW_SAMPLES:
            nof_samples_range = list(range(20, 401, 20))
            nof_selected_features_range = list(range(10, 201, 10))
        else:
            nof_samples_range = list(range(100, 1000, 100)) + list(range(1000, 6001, 500))
            nof_selected_features_range = list(range(10, 201, 10))
    else:
        nof_samples_range = []
        nof_selected_features_range = []

    repeat = 10

    manifest_results = dict()
    relief_results = dict()
    random_results = dict()
    manifest_results.update(nof_selected_features=[], nof_samples=[])
    relief_results.update(nof_selected_features=[], nof_samples=[])
    random_results.update(nof_selected_features=[], nof_samples=[])

    log_name = f"plots/{'run_few_samples' if FEW_SAMPLES else 'run'}.log"

    logging.basicConfig(
        filename=log_name, filemode='w',
        format='%(asctime)s %(levelname)s --> %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger()
    # logger.addHandler(logging.StreamHandler(sys.stdout))

    for i in tqdm(range(repeat), "Initializing Dicts"):
        manifest_results[f'min_test_error_{i}'] = []
        relief_results[f'min_test_error_{i}'] = []
        random_results[f'min_test_error_{i}'] = []

    for nof_samples in tqdm(nof_samples_range, "For nof_samples"):
        for nof_selected_features in tqdm(nof_selected_features_range, "For nof_selected_features"):

            manifest_results['nof_samples'].append(nof_samples)
            relief_results['nof_samples'].append(nof_samples)
            random_results['nof_samples'].append(nof_samples)

            manifest_results['nof_selected_features'].append(nof_selected_features)
            relief_results['nof_selected_features'].append(nof_selected_features)
            random_results['nof_selected_features'].append(nof_selected_features)

            for i in range(repeat):

                save_imgs = (i == 0 and nof_selected_features % 20 == 0)

                manifest_test_error, relief_test_error, random_test_error = run_experiment(
                    adaboost_timesteps=adaboost_timesteps,
                    nof_selected_features=nof_selected_features,
                    nof_samples=nof_samples,
                    save_imgs=save_imgs
                )

                logger.info('=================================================================')
                logger.info(f'\nResults #{i} | {nof_selected_features} Features, {nof_samples} Samples:')
                logger.info('=================================================================')
                logger.info(f'ManiFest AdaBoost Error: {manifest_test_error[-1]}')
                logger.info(f'ReliefF AdaBoost Error: {relief_test_error[-1]}')
                logger.info(f'Random AdaBoost Error: {random_test_error[-1]}')
                logger.info('=================================================================')

                manifest_results[f'min_test_error_{i}'].append(manifest_test_error[-1])
                relief_results[f'min_test_error_{i}'].append(relief_test_error[-1])
                random_results[f'min_test_error_{i}'].append(random_test_error[-1])

            pd.DataFrame.from_dict(manifest_results).to_csv(f"manifest_results{'_few_samples' if FEW_SAMPLES else ''}.csv")
            pd.DataFrame.from_dict(relief_results).to_csv(f"relief_results{'_few_samples' if FEW_SAMPLES else ''}.csv")
            pd.DataFrame.from_dict(random_results).to_csv(f"random_results{'_few_samples' if FEW_SAMPLES else ''}.csv")


if __name__ == '__main__':
    main()
