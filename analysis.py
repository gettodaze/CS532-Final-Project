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

def dim_reduction(df):
    U, S, Vh = pca(df)
    total_s = sum(S)
    cumsum = 0
    for i in range(len(S)):
        cumsum += S[i]/total_s
        if cumsum > .99:
            break;
    return dot(dot(U,S),Vh)

def select_tissue_region(df_feature_set):
    [U, S, V] = pca(df_feature_set)
    S_DIAG = np.diag(S)
    dim_red_data = dot(U[:, :2], S_DIAG[:2, :2])
    kmeans = KMeans(n_clusters=3, n_init=15)
    kmeans.fit(df_feature_set)
    y_kmeans = kmeans.predict(df_feature_set)
    df_clusters = df_feature_set
    df_clusters['clusters'] = y_kmeans
    unique, counts = np.unique(y_kmeans, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    white_space_key = max(cluster_counts, key=cluster_counts.get)
    df_remove_white_space = df_clusters[df_clusters['clusters'] != white_space_key]

    del df_remove_white_space['clusters']
    return df_remove_white_space


def kmeans(A, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(A)
    return kmeans

def graph_ndim_kmeans(df, n_clusters):
    U, S, Vh = np.linalg.svd(np.cov(df.values))
    xs = dot(S[0],Vh[0,:])
    ys = dot(S[1],Vh[1,:])
    kmeans_obj = kmeans(dot(U[:,:2],np.diag(S)[:2,:2]), n_clusters)
    labels, centers = kmeans_obj.labels_, kmeans_obj.cluster_centers_
    return labels
    # plt.scatter(xs, ys, c=labels, s=20, cmap='viridis')
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
def dim_reduction(A):
    U, S, Vh = svd(A, full_matrices=False)
    total_s = sum(S)
    cumsum = 0
    for i in range(len(S)):
        cumsum += S[i]/total_s
        if cumsum > .99:
            break;
    return dot(dot(U[:,:i],np.diag(S[:i])),Vh[:i,:])
def filter_meta_day_exp(day, experiment):
    experiments = get_filepaths_and_meta(input_folder);
    ret = filter(lambda x: x['day'] == day and x['experiment'] == experiment,experiments)
    return list(ret)

def get_x_y(day, experiment):
    meta = filter_meta_day_exp(day, experiment)
    x0 = select_tissue_region(read_and_process_csv(meta[0]['filepath'])).values
    y0 = np.ones(len(x0)) * meta[0]['label']
    x1 = select_tissue_region(read_and_process_csv(meta[1]['filepath'])).values
    y1 = np.ones(len(x1)) * meta[1]['label']
    return np.vstack((x0,x1)), np.append(y0,y1)

if __name__ == '__main__':
    input_folder = r"\\wfs1\users$\mccloskey\Downloads\csvData-20190806T210426Z-001\csvData\aggregates (bin=500)"
    outdir = r"\\wfs1\users$\mccloskey\My Documents\CS 532\CS532-Final-Project\2dim kmeans"
    X, y = get_x_y(3,1)
    # from sklearn import linear_model
    # reg = linear_model.LinearRegression()
    # reg.fit(X,y)
    # print(reg.coef_)
    # X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.4, random_state = 0)
    #
    # clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    # clf.score(X_test, y_test)
    files_and_meta = get_filepaths_and_meta(input_folder)
    day_grouped = day_groups(files_and_meta)
    # # for i in range(len(files_and_meta)):
    # #     df = read_and_process_csv(files_and_meta[i]['filepath'])
    # data
    for key in day_grouped.keys():
        dfs = day_grouped[key]
        label, day = key
        # U, S, Vh = pca(full_df)
        # graph_ndim_kmeans(full_df,2)
        for (meta,df) in dfs:
            df_standardized = (df - df.mean()) / df.std()
            df_standardized = df_standardized.fillna(0)
            # tissue = select_tissue_region(df)
            labels = graph_ndim_kmeans(df_standardized, 3)
            plt.imshow(labels.reshape(meta['x'], meta['y']))
            # # dim_reduction(df)
            # kmeans_obj = kmeans(df, 4)
            # clusters = kmeans_obj.predict(df)
            # plt.imshow(clusters.reshape(meta['x'], meta['y']))
            title = f'Label{label} Day{day}.{meta["experiment"]} bins=500'
            plt.title(title)
            # plt.show()
            plt.savefig(os.path.join(outdir, title+'.png'))