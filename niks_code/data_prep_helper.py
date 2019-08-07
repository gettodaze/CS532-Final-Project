__author__ = "Nik Burmeister"
import pandas as pd
import scipy.signal as signal
from numpy import linspace
import os
from os import listdir
from os.path import isfile, join
from pyimzml.ImzMLParser import ImzMLParser

curr_dir = os.path.dirname(os.path.realpath(__file__))
data_path_imzml = os.path.join(curr_dir, 'imZML')
data_path_csv = os.path.join(curr_dir, 'rawData')


def read_spectrometry_data(mass_file_name, intensity_file_name):
    """
    HELPER METHOD FOR READING DATA FROM CSV
    :return:
    """
    df_intensities = pd.read_csv('rawData/{intensity_file_name}'.format(intensity_file_name=intensity_file_name), index_col=0)
    df_mass = pd.read_csv('rawData/{mass_file_name}'.format(mass_file_name=mass_file_name), index_col=0)
    return df_mass, df_intensities


def find_pixel_peaks(x, y):
    pixel = pd.DataFrame({'mzs': x.values, 'intensities': y.values})
    peaks = signal.find_peaks(y, height=50000)
    pixel = pixel[pixel.index.isin(peaks[0])]
    return pixel


def build_feature_table(mass_file_name, intensity_file_name, dataset_type=None, bins=250, flag=0, shape_x=None, shape_y=None):

    if dataset_type is None:
        raise Exception

    df_mass, df_intensities = read_spectrometry_data(mass_file_name, intensity_file_name)

    # GET NUMBER OF PIXELS, NOTE COLUMN VECTORS ARE FEATURES (IE PIXELS)
    pixel_count = len(df_mass.columns)

    # CREATE A DATAFRAME THAT WILL HOLD ALL FILTER PIXELS
    master_df = pd.DataFrame()

    # FIND PEAKS FOR EACH PIXEL, ADD TO MASTER_DF
    for i in range(pixel_count):
        pixel_x = df_mass.iloc[:, i]
        pixel_y = df_intensities.iloc[:, i]
        df = find_pixel_peaks(pixel_x, pixel_y)
        df['pixel_id'] = i
        master_df = master_df.append(df, ignore_index=True)

    # NUMBER OF BINS FOR THE MZS AXIS
    number_of_bins = bins
    # PHYSICAL MIN AND MAX OF AXIS
    min_mzs = 450
    max_mzs = 1000
    # MAKE THE LINSPACE
    bin_linspace = linspace(min_mzs, max_mzs, number_of_bins)
    col_set = list(range(pixel_count))

    # CREATE NEW DATAFRAME WHERE COLUMNS ARE THE BINS
    aggregated_df = pd.DataFrame(columns=list(bin_linspace[:-1]))

    for i in range(2):
        curr_pixel = master_df[master_df['pixel_id'] == i]
        pixel_binned = {}
        for index in range(1, len(bin_linspace)):
            lower_bound = bin_linspace[index - 1]
            upper_bound = bin_linspace[index]
            curr_bin = curr_pixel[curr_pixel['mzs'].between(lower_bound, upper_bound)]
            bin_intensity = curr_bin['intensities'].sum()
            pixel_binned[lower_bound] = bin_intensity
        temp = pd.DataFrame([pixel_binned])
        aggregated_df = aggregated_df.append(temp, ignore_index=True)

    if flag == 0:
        aggregated_df.to_csv('featureData/aggregated_feature_table_{dataset_type}_b{bins}.csv'.format(
            dataset_type=dataset_type, bins=bins, shape_x=shape_x, shape_y=shape_y))

    if flag == 1:
        return aggregated_df


def output_imzml_data_to_csv(data):
    f_name_mass = "Mass_" + data["f_name"]
    data["mass"].to_csv('rawData/{f_name}.csv'.format(f_name=f_name_mass))
    f_name_intensity = "Intensity_" + data["f_name"]
    data["intensity"].to_csv('rawData/{f_name}.csv'.format(f_name=f_name_intensity))


def load_imzml_data_set(file):
    """

    FLAG=0: SEND TO CSV, RETURN NOTHING
    FLAG=1: RETURN DICT OF DATAFRAMES
    FLAG=2: SEND TO CSV, RETURN DICT OF DATAFRAMES

    :param file:
    :param flag:
    :return:
    """
    imzml_data_path = os.path.join(data_path_imzml, file)
    p = ImzMLParser(imzml_data_path)
    mass_data = {}
    intensity_data = {}
    x_cord, y_cord = p.coordinates[-1][0], p.coordinates[-1][1]
    for idx, (x, y, z) in enumerate(p.coordinates):
        # mzs are masses over charge of 1 ion
        # intensities correspond to the abundance of the particular ion
        mzs, intensities = p.getspectrum(idx)
        mass_data[idx] = mzs
        intensity_data[idx] = intensities

    # CONVERT DICTS TO DATA FRAMES
    df_mass_data = pd.DataFrame(mass_data)
    df_intensity_data = pd.DataFrame(intensity_data)
    f_name = file.split('.')[0]

    return {"mass": df_mass_data, "intensity": df_intensity_data, "x": x_cord, "y":  y_cord, "f_name": f_name}


def load_imzml_data_from_directory(imzml_file_list, flag=0):
    """
    LOADS IMZL FILES. RETURNS THREE DATAFRAMES: INTENSITY, MZS, AND COORDINATES

    FLAG=0: SEND TO CSV, RETURN NOTHING
    FLAG=1: RETURN DICT OF DATAFRAMES
    FLAG=2: SEND TO CSV, RETURN DICT OF DATAFRAMES

    :param filename:
    :param type:
    :return:
    """

    if not isinstance(imzml_file_list, list):
        print("** You must pass a list of imzml file names **")
        raise Exception

    all_data = {}
    for file in imzml_file_list:
        data = load_imzml_data_set(file)
        if flag == 1 or flag == 2:
            f_name = file.split('.')[0]
            all_data[f_name] = data
        if flag == 0 or flag == 2:
            output_imzml_data_to_csv(data)

    return all_data


def get_imzml_file_list():
    """
    GETS ALL FILES FROM IMZML DIRECTORY
    :return:
    """

    all_files = [f for f in listdir(data_path_imzml) if isfile(join(data_path_imzml, f))]
    accepted_files = [file for file in all_files if file.split('.')[1] != 'ibd']
    return accepted_files


def get_csv_file_list():

    all_files = [f for f in listdir(data_path_csv) if isfile(join(data_path_csv, f))]
    accepted_files = [file for file in all_files]
    return accepted_files


def main():
    # 1. GET LIST OF FILES THAT WE WANT TO CONVERT TO CSV
    # imzml_file_list = get_imzml_file_list()

    # 2. GO THROUGH FILE LIST AND LOAD DATA. SEND FLAG=1 TO EXPORT IT TO CSV.
    # all_imzml_data = load_imzml_data_from_directory(imzml_file_list, flag=0)

    csv_file_list = get_csv_file_list()
    control_intensity_day03_01 = csv_file_list[0]
    control_mass_day03_01 = csv_file_list[16]
    dataset_type = "control_day03_01"
    build_feature_table(control_mass_day03_01, control_intensity_day03_01, dataset_type=dataset_type, bins=100)

    injured_intensity_day03_01 = csv_file_list[7]
    injured_mass_day03_01 = csv_file_list[23]
    dataset_type = "injured_day03_01"
    build_feature_table(injured_intensity_day03_01, injured_mass_day03_01, dataset_type=dataset_type, bins=100)




    pass


if __name__ == "__main__":
    main()
    from sklearn.cluster.
