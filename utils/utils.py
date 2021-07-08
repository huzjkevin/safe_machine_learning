import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter


def split_wine_dataset(wine_data_dir, train_set_dir, test_set_dir):

    if not os.path.exists(wine_data_dir):
        raise FileNotFoundError('{} file is not found. need to download and place the file in the mentioned directory'.format(wine_data_dir))

    else: # when the dataset file 'winequality-white.csv' exists
        if not os.path.exists(train_set_dir): # when the dataset file exists but not has been splitted.
            wine_dataset = pd.read_csv(wine_data_dir, sep=';')
            wine_dataset = (wine_dataset - wine_dataset.min())/(wine_dataset.max()-wine_dataset.min())

            #std_scaler = StandardScaler() # define standard normalizer

            #wine_dataset = pd.DataFrame(std_scaler.fit_transform(wine_dataset), columns=wine_dataset.columns)

            X = wine_dataset.drop(labels= 'quality', axis = 1)


            X = (X-X.min())/(X.max() - X.min()) # min-max normalization

            Y = wine_dataset['quality']

            train_x, test_x, train_y, test_y = train_test_split(X,Y, test_size=0.1, stratify=Y)

            df_train = pd.concat([train_x, train_y], axis=1) # concatanate training set with corresponding labels
            df_test = pd.concat([test_x, test_y], axis=1) # concatanate test set with corresponding labels

            # write splitted dataset to csv
            df_train.to_csv(train_set_dir, index=False)
            df_test.to_csv(test_set_dir, index=False)

def get_args_from_yaml(file):
    with open(file) as f:
        conf = yaml.load(f,Loader=yaml.FullLoader)
    return conf


def draw_loss_trend_figure(title, train_loss_lost, test_loss_list, epoch, output_dir):
    """
    Draw and Save loss figure for given parameters

    :param title:
    :param train_loss_lost:
    :param test_loss_list:
    :param epoch:
    :return:
    """
    epochs_train = np.arange(0, epoch, 1)
    epochs_test = np.arange(1, epoch+1,1)
    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(epochs_train, train_loss_lost, label = 'train loss')
    plt.plot(epochs_test, test_loss_list, label= 'test loss')
    plt.legend(loc='best')

    os.makedirs(output_dir, exist_ok=True)
    figure_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figure_dir, exist_ok=True)

    figure_dir = os.path.join(figure_dir, '{}_loss_fig.png'.format(title))
    plt.savefig(figure_dir)
    plt.show()


def plot_and_save_histograms(NLL_list, RMSE_list, output_dir, title="fig"):
    fig, axs = plt.subplots(1, 2, tight_layout=True)
    num_bins = 60

    # N is the count in each bin, bins is the lower-limit of the bin
    N, bins, patches = axs[0].hist(NLL_list, bins=num_bins, weights=np.ones(len(NLL_list)) / len(NLL_list))
    axs[0].yaxis.set_major_formatter(PercentFormatter(1))

    axs[0].set_title(title + ': NLL Histogram')
    axs[0].set_xlabel('NLL value')
    # We'll color code by height, but you could use any scalar
    fracs = N / N.max()

    # we need to normalize the data to 0..1 for the full range of the colormap
    norm = colors.Normalize(fracs.min(), fracs.max())

    # Now, we'll loop through our objects and set the color of each accordingly
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)


    N, bins, patches = axs[1].hist(RMSE_list, bins=num_bins, weights=np.ones(len(RMSE_list)) / len(RMSE_list))
    axs[1].set_title(title + ': RMSE Histogram')
    axs[1].set_xlabel('RMSE value')
    axs[1].yaxis.set_major_formatter(PercentFormatter(1))

    fracs = N / N.max()
    norm = colors.Normalize(fracs.min(), fracs.max())
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)

    os.makedirs(output_dir, exist_ok=True)
    figure_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figure_dir, exist_ok=True)

    figure_dir = os.path.join(figure_dir, title + '_NLL_RMSE_histogram.png')
    plt.savefig(figure_dir)
    plt.show()

def plot_scatter(NLL_list, RMSE_list, output_dir, title="fig"):
    plt.scatter(NLL_list, RMSE_list)
    plt.title(title + ': NLL and RMSE Scatter')
    plt.xlabel('NLL value')
    plt.ylabel('RMSE value')

    os.makedirs(output_dir, exist_ok=True)
    figure_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figure_dir, exist_ok=True)

    figure_dir = os.path.join(figure_dir, title + '_NLL_RMSE_scatter.png')
    plt.savefig(figure_dir)
    plt.show()


### TODO add code below when merge conflict happens

def plot_Mahalanobis_distance(sample_M_distance_list, gt_M_distance_list, output_dir, title="fig"):
    num_bins = 100

    bins=np.histogram(np.hstack((sample_M_distance_list,gt_M_distance_list)), bins=num_bins)[1] # to get the equal bin width

    plt.hist(gt_M_distance_list, bins,   alpha=0.5, label='Ground Truth', weights=np.ones(len(gt_M_distance_list)) / len(gt_M_distance_list))
    plt.hist(sample_M_distance_list, bins,  alpha=0.5, label='Sample', weights=np.ones(len(sample_M_distance_list)) / len(sample_M_distance_list))


    plt.legend(loc='upper right')
    plt.xlabel('Mahalanobis_distance')
    plt.ylabel('frequency')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title(title + ': Assessment of uncertainty realism')

    os.makedirs(output_dir, exist_ok=True)
    figure_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figure_dir, exist_ok=True)
    figure_dir = os.path.join(figure_dir, title + '_Assesment of Uncertainty Realism.png')
    plt.savefig(figure_dir)

    plt.show()

####################################################################################