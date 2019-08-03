import pandas as pd
import numpy as np
import os
from pyimzml.ImzMLParser import ImzMLParser
from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
from numpy import linspace
import scipy.signal as signal
from numpy.linalg import svd

curr_dir = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(curr_dir, 'data', 'data_files', 'imzmLs')


def read_data():
    """
    HELPER METHOD FOR READING DATA FROM CSV
    :return:
    """
    df_intensities = pd.read_csv('intensities.csv')
    df_mass = pd.read_csv('mass_data.csv')
    return df_mass, df_intensities


def build_feature_table(dataset_type):
    """
    FUNCTION FOR AGGREGATING PIXELS OF A SINGLE IMAGE INTO A FEATURE TABLE
    :param dataset_type:
    :return:
    """

    # READ THE DATA
    df_mass, df_intensities = read_data()
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
    aggregated_df.to_csv('aggregated_feature_table_{dataset}.csv'.format(dataset=dataset_type))


def save_data_to_csv(filename):
    data_control_day_03 = os.path.join(data_path, filename)
    p = ImzMLParser(data_control_day_03)
    mass_data = {}
    intensity_data = {}
    for idx, (x, y, z) in enumerate(p.coordinates):
        # mzs are masses over charge of 1 ion
        # intensities correspond to the abundance of the particular ion
        mzs, intensities = p.getspectrum(idx)
        mass_data[idx] = mzs
        intensity_data[idx] = intensities
    df1 = pd.DataFrame(mass_data)
    df2 = pd.DataFrame(intensity_data)
    df1.to_csv('mass_data.csv')
    df2.to_csv('intensities.csv')


def pca():
    df_control_feature_set = pd.read_csv('aggregate_feature_table_control')
    df_injured_feature_set = pd.read_csv('aggregate_feature_table_injured')
    num_control_pixels = len(df_control_feature_set)
    num_injured_pixels = len(df_injured_feature_set)
    vec_control_label = np.ones(num_control_pixels)
    vec_injured_label = -1*np.ones(num_injured_pixels)
    vec_combined_labels = np.concatenate((vec_control_label,vec_injured_label), axis=0)
    df_combined_feature_set = pd.concat([df_control_feature_set, df_injured_feature_set])
    [U, S, V] = svd(df_combined_feature_set.values)
    print


def main():
    control_set = 'Control_Day03_01.imzML'
    injured_set = 'Injured_Day03_01.imzML'
    save_data_to_csv(control_set)
    build_feature_table('control')


    # plt.scatter(x, y)
    # plt.show()


if __name__ == '__main__':
    main()