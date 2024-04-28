import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import seaborn as sns
import scipy


def heatmap(vals, ax1, ax2, label1="", label2="", method="", key="", vmax=0.5):
    ax1_uniq = sorted(np.unique(ax1))
    ax2_uniq = sorted(np.unique(ax2))[::-1]
    X, Y = np.meshgrid(ax1_uniq, ax2_uniq)

    Z = np.zeros_like(X, dtype=np.float64)
    for i, x in enumerate(ax1_uniq):
        for j, y in enumerate(ax2_uniq):
            Z[j, i] = vals[np.where((ax1 == x) & (ax2 == y))[0]][0]

    zoom_rate = 1
    Z = scipy.ndimage.zoom(Z, zoom_rate, order=3)
    xlabels = [str(x) for x in scipy.ndimage.zoom(ax1_uniq, zoom_rate, order=0)[::zoom_rate]]
    ylabels = [str(x) for x in scipy.ndimage.zoom(ax2_uniq, zoom_rate, order=0)[::zoom_rate]]

    plt.figure(figsize=(8, 5))
    ax = sns.heatmap(Z, xticklabels=xlabels, yticklabels=ylabels, vmin=0, vmax=vmax)
    ax.set_xlabel(label1)
    ax.set_ylabel(label2)
    fig_title = f"{method} FS-based classification error {key}"
    ax.set_title(fig_title)
    plt.savefig(f'analysis/{fig_title.replace(" ", "_")}.png')
    plt.cla()
    plt.close()


def violin_plots(df, col, method):
    accuracies = df[col].to_numpy()
    nof_selected_features = df['nof_selected_features'].to_numpy()
    nof_samples = df['nof_samples'].to_numpy()

    uniq_nof_features = np.unique(nof_selected_features)[::2]
    uniq_nof_samples = np.unique(nof_samples)[::2]

    acc_per_nof_features = [accuracies[np.where(nof_selected_features == n)] for n in uniq_nof_features]
    acc_per_nof_samples = [accuracies[np.where(nof_samples == m)] for m in uniq_nof_samples]

    fig, (ax1, ax3) = plt.subplots(nrows=1, ncols=2, figsize=(20, 4), sharey=True)

    ax1.violinplot(acc_per_nof_features, showmeans=True)
    ax1.set_title(f"{method} FS Classification Errors\nPer # of Features")
    ax1.set_ylabel("Error")
    ax1.set_xlabel("# of Features")
    ax1.set_xticks(np.arange(1, len(uniq_nof_features)+1), labels=[str(l) for l in uniq_nof_features])

    ax3.violinplot(acc_per_nof_samples, showmeans=True)
    ax3.set_title(f"{method} FS Classification Errors\nPer # of Samples")
    ax3.set_ylabel("Error")
    ax3.set_xlabel("# of Samples")
    ax3.set_xticks(np.arange(1, len(uniq_nof_samples)+1), labels=[str(l) for l in uniq_nof_samples])

    plt.savefig(f"analysis/{method}_violin_plots.png")
    plt.cla()
    plt.close()


def heatmap_df(df, method):
    heatmap(df['mean_test_error'].to_numpy(), df['nof_selected_features'].to_numpy(), df['nof_samples'].to_numpy(),
            label1='nof_selected_features', label2='nof_samples',method=method, key='mean', vmax=0.4)
    heatmap(df['std_test_error'].to_numpy(), df['nof_selected_features'].to_numpy(), df['nof_samples'].to_numpy(),
            label1='nof_selected_features', label2='nof_samples', method=method, key='std', vmax=0.1)


def compare_methods(df1, df2, settings='hard'):
    manifest_accuracies = df1['mean_test_error'].to_numpy()
    relief_accuracies = df2['mean_test_error'].to_numpy()
    nof_samples = df1['nof_samples'].to_numpy()
    nof_features = df1['nof_selected_features'].to_numpy()
    # ratios = df['features_samples_ratio'].to_numpy()

    assert settings in ['hard', 'soft']

    if settings == 'soft':
        nof_features_ref = 150
        nof_samples_ref = 1000
    else:
        nof_features_ref = 20
        nof_samples_ref = 100

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharey=True)

    # Build data for examining dependency on #samples, fixing the number of features
    manifest_accs_per_nof_samples = manifest_accuracies[np.where(nof_features == nof_features_ref)]
    relief_accs_per_nof_samples = relief_accuracies[np.where(nof_features == nof_features_ref)]
    nof_samples_reduced = nof_samples[np.where(nof_features == nof_features_ref)]
    ax1.plot(nof_samples_reduced, manifest_accs_per_nof_samples, label=f"Manifest {nof_features_ref} Features")
    ax1.plot(nof_samples_reduced, relief_accs_per_nof_samples, label=f"ReliefF {nof_features_ref} Features")
    ax1.set_title(f"Supervised FS Dependency on # of Training Samples")
    ax1.legend()
    ax1.set_ylabel("Error")
    ax1.set_xlabel("Number of Training Samples")

    # Build data for examining dependency on #features, fixing the number of training samples
    manifest_accs_per_nof_features = manifest_accuracies[np.where(nof_samples == nof_samples_ref)]
    relief_accs_per_nof_features = relief_accuracies[np.where(nof_samples == nof_samples_ref)]
    nof_features_reduced = nof_features[np.where(nof_samples == nof_samples_ref)]
    ax2.plot(nof_features_reduced, manifest_accs_per_nof_features, label=f"Manifest {nof_samples_ref} samples")
    ax2.plot(nof_features_reduced, relief_accs_per_nof_features, label=f"ReliefF {nof_samples_ref} samples")
    ax2.set_title(f"Supervised FS Dependency on # of Features")
    ax2.legend()
    ax2.set_ylabel("Error")
    ax2.set_xlabel("Number of Selected Features")

    plt.savefig(f"analysis/methods_dependency_comparison_plots_{settings}.png")
    plt.cla()
    plt.close()


def main():
    # df = pd.read_csv("results.csv")

    manifest_df = pd.read_csv("manifest_results.csv")
    relief_df = pd.read_csv("relief_results.csv")
    random_df = pd.read_csv("random_results.csv")

    manifest_df['mean_test_error'] = manifest_df.iloc[:, -10:].mean(axis=1)
    manifest_df['std_test_error'] = manifest_df.iloc[:, -10:].std(axis=1)
    relief_df['mean_test_error'] = relief_df.iloc[:, -10:].mean(axis=1)
    relief_df['std_test_error'] = relief_df.iloc[:, -10:].std(axis=1)
    random_df['mean_test_error'] = random_df.iloc[:, -10:].mean(axis=1)
    random_df['std_test_error'] = random_df.iloc[:, -10:].std(axis=1)

    # Manifest
    heatmap_df(manifest_df,'ManiFest')
    violin_plots(manifest_df, 'mean_test_error', 'ManiFest')

    # ReliefF
    heatmap_df(relief_df, 'ReliefF')
    violin_plots(relief_df, 'mean_test_error', 'ReliefF')

    # Random
    heatmap_df(random_df, 'Random')
    violin_plots(random_df, 'mean_test_error', 'Random')

    # Comparison
    compare_methods(manifest_df, relief_df, settings='hard')
    compare_methods(manifest_df, relief_df, settings='soft')


if __name__ == "__main__":
    main()