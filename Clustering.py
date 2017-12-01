from utils import read_jsonl, color_zero_white, color_red, color_blue, highlight_max_red, magnify
from analysis import ma_transform_by_spacing, get_matrix
from sklearn.cluster import MiniBatchKMeans, KMeans
from collections import OrderedDict
import pandas as pd
import numpy as np

def Kmeans_clustering(load_file_path, vectorizer, n_cluster=None):
    app_id_list, app_name_list, cate_list, rating_list, ma_list = read_jsonl(load_file_path)
    # word_idx = load_json('./data/word_idx_json_by_app_id.txt')
    spaced_ma_list = ma_transform_by_spacing(ma_list)
    coo_matrix, words, _ = get_matrix(vectorizer, spaced_ma_list)

    if not n_cluster:
        n_cluster = len(set(cate_list))

    m = coo_matrix.toarray()
    model = MiniBatchKMeans(n_clusters=n_cluster, verbose=0, n_init=10, batch_size=100)
    model.fit(m)

    return model, words, app_id_list, app_name_list, cate_list, rating_list, ma_list


def get_cent_words(model, words, n_cluster, n_max):
    ordered_centroids = model.cluster_centers_.argsort()[:, ::-1]
    cluster_dict = OrderedDict()
    for cluster_num in range(n_cluster):
        center_word_nums = [word_num for word_num in ordered_centroids[cluster_num, :n_max]]
        center_words = [words[word_num].split('/')[0] for word_num in center_word_nums]
        cluster_dict[cluster_num] = center_words

    table = pd.DataFrame(cluster_dict).T

    return table


def get_clusters_df(model, category_list, app_name_list):
    df = pd.DataFrame({'cluster': model.labels_,
                      'category': category_list,
                      'app_name': app_name_list})

    df2 = df.groupby(df.cluster).agg({'app_name': 'size', 'category': 'unique'})
    df2.columns = ['total_apps_nums', 'categories']
    return df, df2


def main_cluster(vectorizer, n_cluster=24, n_max=10):

    load_file_path = './data/docs_jsonl_ma_by_app_id.txt'

    model, words, app_id_list, app_name_list, cate_list, rating_list, ma_list = \
        Kmeans_clustering(load_file_path, vectorizer, n_cluster=n_cluster)
    t = get_cent_words(model, words, n_cluster=n_cluster, n_max=n_max)
    df, df2 = get_clusters_df(model, cate_list, app_name_list)

    re_df = pd.DataFrame(np.hstack([t.values, df2.values]))
    re_df.columns = ['키워드'+str(i) for i in range(1, n_max+1)] + ['앱의 갯수', '실제 카테고리들']
    print('Done!')

    df_ = df.pivot_table('app_name', 'cluster', 'category', aggfunc='count').T
    re_df.index = re_df.index + 1
    df_.columns = df_.columns + 1
    return re_df, df_, df
