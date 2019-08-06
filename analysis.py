import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from numpy import linspace, dot
import scipy.signal as signal
from numpy.linalg import svd
from collections import defaultdict
from matplotlib.pyplot import plot
from sklearn.decomposition import PCA
NEGATIVE_LABEL = "control"
POSITIVE_LABEL = 'injured'


def extract_meta(fpath):
    regex_search = re.search(r'(.*)_Day(\d\d)_(\d\d)_(\d\d)x(\d\d)_', fpath)
    label, day, experiment, size_y, size_x = regex_search.groups()
    assert label.lower() in [NEGATIVE_LABEL, POSITIVE_LABEL]
    return {
        'filepath': fpath,
        'label': 1 if label.lower() == POSITIVE_LABEL else -1,
        'day': int(day),
        'experiment': int(experiment),
        'x': int(size_x),
        'y': int(size_y)
    }

def get_filepaths_and_meta(csv_folder):
    ret = [extract_meta(f) for f in os.listdir(csv_folder) if f.endswith('.csv')]
    for x in ret:
        x['filepath'] = os.path.join(csv_folder, x['filepath'])
    return ret

def read_and_process_csv(csv_fpath):
    df = pd.read_csv(csv_fpath)
    del df[df.columns[-1]]
    del df['Unnamed: 0']
    return df

def pca(df):
    num_pixels = len(df)
    df_standardized = (df - df.mean()) / df.std()
    df_standardized = df_standardized.fillna(0)
    covariance_mat = np.cov(df_standardized.values)
    # img = (covariance_mat-covariance_mat.min()) *255 / (covariance_mat.max()-covariance_mat.min())
    # plt.imshow(img)
    # plt.show()
    return svd(covariance_mat, full_matrices=False)

def kmeans(df, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(df.values)
    return kmeans

def graph_ndim_kmeans(df, n_clusters):
    U, S, Vh = pca(df)
    xs = dot(S[0],Vh[0,:])
    ys = dot(S[1],Vh[1,:])
    labels, centers = kmeans(df, n_clusters)
    plt.scatter(xs, ys, c=labels, s=20, cmap='viridis')
    # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.title('K-means of Pixel MS data')
    plt.xlabel('Singular Vector 1')
    plt.ylabel('Singular Vector 2')
    plt.show()

def day_groups(files_and_meta):
    ret = defaultdict(lambda: [])
    for fam in files_and_meta:
        ret[(fam['label'], fam['day'])].append((fam, read_and_process_csv(fam['filepath'])))
    return ret

if __name__ == '__main__':
    input_folder = r"C:\Users\John\Documents\School\19 Summer\532 ML\CS 532 Project\Data\csvData\aggregates (bin=250)"
    files_and_meta = get_filepaths_and_meta(input_folder)
    day_grouped = day_groups(files_and_meta)
    # for i in range(len(files_and_meta)):
    #     df = read_and_process_csv(files_and_meta[i]['filepath'])
    for key in day_grouped.keys():
        dfs = day_grouped[key]
        label, day = key
        full_df = pd.DataFrame()
        for (meta,df) in dfs:
            full_df = full_df.append(df)
        # U, S, Vh = pca(full_df)
        # graph_ndim_kmeans(full_df,2)
        kmeans_obj = kmeans(full_df, 2)
        for (meta,df) in dfs:
            clusters = kmeans_obj.predict(df)
            plt.imshow(clusters.reshape(meta['x'], meta['y']))
            plt.show()
            plt.title(f'Label: {label} | Day: {day}')