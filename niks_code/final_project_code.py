import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from numpy import linspace, dot, transpose
from numpy.linalg import svd, pinv
from matplotlib.pyplot import plot
from sklearn.decomposition import PCA


curr_dir = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(curr_dir, 'imZML')


def find_rank(S):
    aggregate_sum = sum(S)
    cum_sum = 0
    for i, v in enumerate(S):
        cum_sum = cum_sum + v
        cum_sum_percent = cum_sum / aggregate_sum
        if cum_sum_percent > .99:
            return i


def pca_with_sklearn():
    df_control_feature_set = pd.read_csv('csvData/aggregated_feature_table_control.csv')
    df_injured_feature_set = pd.read_csv('csvData/aggregated_feature_table_injured.csv')
    num_control_pixels = len(df_control_feature_set)
    num_injured_pixels = len(df_injured_feature_set)
    vec_control_label = np.ones(num_control_pixels)
    vec_injured_label = -1 * np.ones(num_injured_pixels)
    vec_combined_labels = np.concatenate((vec_control_label, vec_injured_label), axis=0)
    df_combined_feature_set = pd.concat([df_control_feature_set, df_injured_feature_set])

    df_combined_feature_set_standardized = (df_combined_feature_set - df_combined_feature_set.mean()) / df_combined_feature_set.std()
    df_combined_feature_set_standardized = df_combined_feature_set_standardized.fillna(0)

    reduced_data = PCA(n_components=2).fit_transform(df_combined_feature_set_standardized.values)
    kmeans = KMeans(init='k-means++', n_clusters=3, n_init=10)
    kmeans.fit(reduced_data)
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = 0.1  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def kmeans_plot(df_feature_set, type):
    if type=="control":
        _shape = (30, 30)
    else:
        _shape = (32, 31)

    df_control_feature_set_standardized = (df_feature_set - df_feature_set.mean()) / df_feature_set.std()
    df_control_feature_set_standardized = df_control_feature_set_standardized.fillna(0)
    [U, S, V] = svd(df_control_feature_set_standardized.values)
    S_DIAG = np.diag(S)
    dim_red_data = dot(S_DIAG[:2, :2], U[:, :2].transpose()).transpose()
    kmeans = KMeans(n_clusters=3, n_init=15)
    kmeans.fit(dim_red_data)
    y_kmeans = kmeans.predict(dim_red_data)
    df_clusters = pd.DataFrame(dim_red_data)
    df_clusters['clusters'] = y_kmeans
    plt.imshow(y_kmeans.reshape(33, 30))
    plt.show()


def imaging_different_molecules(df_feature_set):
    del df_feature_set['Unnamed: 0']
    del df_feature_set[df_feature_set.columns[-1]]
    df_control_feature_set_standardized = (df_feature_set - df_feature_set.mean()) / df_feature_set.std()
    df_control_feature_set_standardized = df_control_feature_set_standardized.fillna(0)
    [U, S, V] = svd(df_control_feature_set_standardized.values)
    num_cols = len(df_control_feature_set_standardized.columns)
    images = {}
    for i in range(num_cols):
        image = df_control_feature_set_standardized.iloc[:, i].values.reshape(30, 30)
        max_intensity = image.max()
        norm_image = np.multiply(np.divide(image, max_intensity), 255)
        images[i] = norm_image
        plt.imshow(norm_image)
        plt.show()


def select_tissue_region(df_feature_set):
    del df_feature_set['Unnamed: 0']
    del df_feature_set[df_feature_set.columns[-1]]
    df_control_feature_set_standardized = (df_feature_set - df_feature_set.mean()) / df_feature_set.std()
    df_control_feature_set_standardized = df_control_feature_set_standardized.fillna(0)
    [U, S, V] = svd(df_control_feature_set_standardized.values)
    S_DIAG = np.diag(S)
    dim_red_data = dot(S_DIAG[:2, :2], U[:, :2].transpose()).transpose()
    kmeans = KMeans(n_clusters=3, n_init=15)
    kmeans.fit(dim_red_data)
    y_kmeans = kmeans.predict(dim_red_data)
    df_clusters = df_control_feature_set_standardized
    df_clusters['clusters'] = y_kmeans
    unique, counts = np.unique(y_kmeans, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    white_space_key = max(cluster_counts, key=cluster_counts.get)
    df_remove_white_space = df_clusters[df_clusters['clusters'] != white_space_key]
    del df_remove_white_space['clusters']
    return df_remove_white_space


def pca():
    df_control_feature_set = pd.read_csv('csvData/aggregated_feature_table_control.csv')
    df_injured_feature_set = pd.read_csv('csvData/aggregated_feature_table_injured.csv')
    num_control_pixels = len(df_control_feature_set)
    num_injured_pixels = len(df_injured_feature_set)
    vec_control_label = np.ones(num_control_pixels)
    vec_injured_label = -1*np.ones(num_injured_pixels)
    vec_combined_labels = np.concatenate((vec_control_label, vec_injured_label), axis=0)
    df_combined_feature_set = pd.concat([df_control_feature_set, df_injured_feature_set])
    del df_combined_feature_set[df_combined_feature_set.columns[-1]]
    del df_combined_feature_set['Unnamed: 0']
    # df_combined_feature_set.to_excel('combined_feature_set.xlsx')
    df_combined_feature_set_standardized = (df_combined_feature_set - df_combined_feature_set.mean()) / df_combined_feature_set.std()
    df_combined_feature_set_standardized = df_combined_feature_set_standardized.fillna(0)

    # kmeans_plot(df_injured_feature_set, "injured")

    df_control_no_ws = select_tissue_region(df_control_feature_set)
    vec_control_label = np.ones(len(df_control_no_ws))
    df_injured_no_ws = select_tissue_region(df_injured_feature_set)
    vec_injured_label = -1 * np.ones(len(df_injured_no_ws))
    df_combined_feature_set = pd.concat([df_control_no_ws, df_injured_no_ws])
    vec_combined_labels = np.concatenate((vec_control_label, vec_injured_label), axis=0)
    vec_combined_feature_set = df_combined_feature_set.values

    ATA = dot(transpose(vec_combined_feature_set), vec_combined_feature_set)
    ATA_inv = pinv(ATA)
    w = dot(dot(ATA_inv, transpose(vec_combined_feature_set)), vec_combined_labels)
    print

    """
    END SEGMENT
    """

    # FIND THE SVD, NOTE V IS RETURNED TRANSPOSED
    # [U, S, V] = svd(df_control_feature_set_standardized.values)
    # # CREATE A DIAG MATRIX
    # S_DIAG = np.diag(S)
    # # COMPUTE THE RANK 2 APPOIMATION TO FIND THE COORDINATES
    # dim_red_data = dot(S_DIAG[:2, :2], U[:, :2].transpose()).transpose()
    #
    # kmeans = KMeans(n_clusters=3, n_init=15)
    # kmeans.fit(dim_red_data)
    # y_kmeans = kmeans.predict(dim_red_data)
    # df_clusters = pd.DataFrame(dim_red_data)
    # df_clusters['clusters'] = y_kmeans
    #
    # x = df_clusters[0]
    # y = df_clusters[1]
    # plt.scatter(x, y, c=y_kmeans, s=50, cmap='viridis')
    # centers = kmeans.cluster_centers_
    # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    # plt.title('Clusters Wattage vs Duration')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()
    # print





def main():
    # # control_set = 'Control_Day14_01.imzML'
    # injured_set = 'Injured_Day03_03.imzML'
    # # save_data_to_csv(control_set, 'control')
    # save_data_to_csv(injured_set, 'injured')
    # # build_feature_table('control')
    # build_feature_table('injured')
    # pass
    # pca()
    #plt.scatter(x, y)
    # plt.show()


if __name__ == '__main__':
    main()