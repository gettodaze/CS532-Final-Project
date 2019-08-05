import pandas as pd
import numpy as np
import os
from pyimzml.ImzMLParser import ImzMLParser
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from numpy import linspace, dot
import scipy.signal as signal
from numpy.linalg import svd
from matplotlib.pyplot import plot
from sklearn.decomposition import PCA

curr_dir = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(curr_dir, 'data', 'data_files', 'imzmLs')


def read_data(dataset_type):
    """
    HELPER METHOD FOR READING DATA FROM CSV
    :return:
    """
    df_intensities = pd.read_csv('csvData/intensities_{dataset_type}.csv'.format(dataset_type=dataset_type))
    df_mass = pd.read_csv('csvData/mass_data_{dataset_type}.csv'.format(dataset_type=dataset_type))
    return df_mass, df_intensities


def build_feature_table(dataset_type):
    """
    FUNCTION FOR AGGREGATING PIXELS OF A SINGLE IMAGE INTO A FEATURE TABLE
    :param dataset_type:
    :return:
    """

    # READ THE DATA
    df_mass, df_intensities = read_data(dataset_type)
    # GET NUMBER OF PIXELS, NOTE COLUMN VECTORS ARE FEATURES (IE PIXELS)
    pixel_count = len(df_mass.columns)
    master_df = pd.DataFrame()

    for i in range(pixel_count - 1):
        pixel_x = df_mass[str(i)].values
        pixel_y = df_intensities[str(i)].values
        # PUT X AND Y INTO A 2 COLUMN MATRIX
        df_train = pd.DataFrame({'mzs': pixel_x, 'intensities': pixel_y})
        # FIND NUMBER OF PEAKS, RETURNS THE INDEX OF VALUE MZS
        peaks = signal.find_peaks(pixel_y, height=50000)
        # GET THE REDUCED DATA SET, FILTER BY PEAK INDICES
        df_train = df_train[df_train.index.isin(peaks[0])]
        df_train['pixel_id'] = i
        master_df = master_df.append(df_train, ignore_index=True)
    master_df = master_df.sort_values('mzs')
    number_of_bins = 250
    min_mzs = 450
    max_mzs = 1000
    bins = linspace(min_mzs, max_mzs, number_of_bins)
    col_set = list(range(pixel_count - 1))

    aggregated_df = pd.DataFrame(columns=list(bins))

    for i in col_set:
        curr_pixel = master_df[master_df['pixel_id'] == i]
        pixel_binned = {}
        for index in range(1, len(bins)):
            lower_bound = bins[index - 1]
            upper_bound = bins[index]
            curr_bin = curr_pixel[curr_pixel['mzs'].between(lower_bound, upper_bound)]
            bin_intensity = curr_bin['intensities'].sum()
            pixel_binned[lower_bound] = bin_intensity
        temp = pd.DataFrame([pixel_binned])
        aggregated_df = aggregated_df.append(temp)
    aggregated_df.to_csv('csvData/aggregated_feature_table_{dataset_type}.csv'.format(dataset_type=dataset_type))


def save_data_to_csv(filename, type):
    data_control_day_03 = os.path.join(data_path, filename)
    p = ImzMLParser(data_control_day_03)
    mass_data = {}
    intensity_data = {}
    coords = {}
    for idx, (x, y, z) in enumerate(p.coordinates):
        # mzs are masses over charge of 1 ion
        # intensities correspond to the abundance of the particular ion
        mzs, intensities = p.getspectrum(idx)
        mass_data[idx] = mzs
        intensity_data[idx] = intensities
        coords[idx] = {"x": x, "y": y, "z": z}
    df1 = pd.DataFrame(mass_data)
    df2 = pd.DataFrame(intensity_data)
    df3 = pd.DataFrame.from_dict(coords, orient="index")
    df1.to_csv('csvData/mass_data_{type}.csv'.format(type=type))
    df2.to_csv('csvData/intensities_{type}.csv'.format(type=type))
    df3.to_csv('csvData/coords_{type}.csv'.format(type=type))


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
    del df_combined_feature_set[df_combined_feature_set.columns[-1]]
    del df_combined_feature_set['Unnamed: 0']

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


def pca():
    df_control_feature_set = pd.read_csv('aggregated_feature_table_control.csv')
    df_injured_feature_set = pd.read_csv('aggregated_feature_table_injured.csv')
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

    [U, S, V] = svd(df_combined_feature_set_standardized.values)
    S_DIAG = np.diag(S)

    # THIS IS THE DIM REDUCTION USING FIRST TWO COLS OF V
    #dim_red_data = dot(df_combined_feature_set_standardized.values, V[:2, :].transpose())

    dim_red_data = dot(S_DIAG[:2, :2], U[:, :2].transpose()).transpose()

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(dim_red_data)
    y_kmeans = kmeans.predict(dim_red_data)

    df_clusters = pd.DataFrame(dim_red_data)
    df_clusters['clusters'] = y_kmeans

    x = df_clusters[0]
    y = df_clusters[1]
    plt.scatter(x, y, c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.title('Clusters Wattage vs Duration')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    print


def main():
    control_set = 'Control_Day03_01.imzML'
    injured_set = 'Injured_Day03_01.imzML'
    save_data_to_csv(control_set, 'control')
    save_data_to_csv(injured_set, 'injured')
    build_feature_table('control')
    build_feature_table('injured')

    # pca()
    # plt.scatter(x, y)
    # plt.show()


if __name__ == '__main__':
    main()