import glob, os
from pyimzml.ImzMLParser import ImzMLParser
import pandas as pd
import scipy.signal as signal
from numpy import linspace, dot
import numpy as np
from os import path

def idxs_to_bool(idxs, length=None):
    length = length or idxs[-1]+1
    ret = np.zeros(length,dtype=bool)
    for idx in idxs:
        ret[idx] = True;
    return ret

output_directory = r"C:\Users\John\Documents\School\19 Summer\532 ML\CS 532 Project\Data\csvData\aggregates"
input_directory = r"C:\Users\John\Documents\School\19 Summer\532 ML\CS 532 Project\Data\Restenosis Files From Jill\imzmLs"
out = []


os.chdir(input_directory)
files = [file for file in glob.glob("*.imzML")]
for f in files:
    print(f)
    p = ImzMLParser(f)
    shape = (p.imzmldict['max count of pixels x'], p.imzmldict['max count of pixels y'])
    spectrums = [p.getspectrum(i) for i in range(len(p.coordinates))]
    all_mzs,  all_intensities = zip(*spectrums)
    peaks, peak_intensities = [], []
    for i,intensities in enumerate(all_intensities):
        print(f'Getting Intensities: {i}/{len(all_intensities)}')
        t = signal.find_peaks(intensities, 50*1000)
        peaks.append(all_mzs[i][idxs_to_bool(t[0], len(intensities))])
        peak_intensities.append(t[1]['peak_heights'])
    number_of_bins = 250
    min_mzs = 450
    max_mzs = 1000
    bins = linspace(min_mzs, max_mzs, number_of_bins)
    col_set = list(range(len(p.coordinates) - 1))
    aggregated_df = pd.DataFrame(columns=list(bins))
    master_df = pd.DataFrame()
    for pixel, (peak_l, intensity_l) in enumerate(zip(peaks, peak_intensities)):
        print(f'Binning: {pixel}/{len(p.coordinates)}')
        curr_pixel = pd.DataFrame({'mzs': peak_l, 'intensities': intensity_l})
        pixel_binned = {}
        for index in range(1, len(bins)):
            lower_bound = bins[index - 1]
            upper_bound = bins[index]
            curr_bin = curr_pixel[curr_pixel['mzs'].between(lower_bound, upper_bound)]
            bin_intensity = curr_bin['intensities'].sum()
            pixel_binned[lower_bound] = bin_intensity
        temp = pd.DataFrame([pixel_binned])
        aggregated_df = aggregated_df.append(temp)

    data_name = path.splitext(f)[0]
    outfile = path.join(output_directory,data_name)
    aggregated_df.to_csv(f'{outfile}_{shape[0]}x{shape[1]}_aggregated.csv')
