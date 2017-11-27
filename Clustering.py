from analysis import *
from sklearn.cluster import MiniBatchKMeans, KMeans

def Kmeans_clustering(n_apps, jsonl_file_path, flat_file_path, cate_name_path, choose_pos, n_cluster=None):
    make_dict, app_id_list, ratings_lists, ma_lists = make_keyword_dict(jsonl_file_path, choose_pos=choose_pos,
                                                                        print_keywords=False)
    boundary = filter_docs_by_dict(make_dict, ma_lists[:n_apps], ratings_lists, save_file_path=flat_file_path,
                                   by_app_id=True)

    flat_docs, flat_ratings = data_loading(save_file_path=flat_file_path, make_dict=make_dict)
    category_list, app_name_list = get_app_name_category(cate_name_path, app_id_list)
    coo_matrix, words = get_tfidf_matrix(flat_docs, make_dict)
    # category 에 nan가 섞여 있음
    category_list = ['None' if type(i) == float else i for i in category_list]
    if not n_cluster:
        n_cluster = len(set(category_list))
    m = coo_matrix.toarray()
    # model = KMeans(n_clusters=n_cluster, verbose=0)
    model = MiniBatchKMeans(n_clusters=n_cluster, verbose=0, n_init=10, batch_size=100)

    model.fit(m)

    return model, words, boundary, flat_docs, flat_ratings, app_id_list[:n_apps], category_list[:n_apps], app_name_list[:n_apps]


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


def main_cluster(n_cluster=24, n_max=10, option_major_pos=True, n_apps=2985):

    jsonl_file_path = './data/train_jsonl_ma_komoran_after_processing.txt'
    flat_file_path = ['./data/train_flat_docs_by_app_id.txt', './data/train_flat_ratings_by_app_id.txt']
    cate_name_path = './data/cate_app_name_train.txt'

    if option_major_pos:
        major_pos = ["NNG", "NNP", "NP", "XR", "VV", "VA", "MAG", "MAJ"]
        choose_pos = major_pos
    else:
        full_morph = ['NNG', 'NNP', 'NNB', 'NR', 'NP', 'VV', 'VA', 'VX', 'VCP', 'VCN', 'MM', 'MAG', 'MAJ', 'IC']
        empty_morph = ['JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC', 'EP', 'EF', 'EC', 'ETN', 'ETM',
                       'XPN','XSN', 'XSV', 'XSA']
        egun = ['XR']
        signs = ['SF', 'SE', 'SS', 'SP', 'SO', 'SW']
        not_koreans = ['SL', 'SH', 'SN']
        choose_pos = full_morph + egun + not_koreans # + empty_morph
    #
    model, words, boundary, flat_docs, flat_ratings, app_id_list, category_list, app_name_list = \
        Kmeans_clustering(n_apps, jsonl_file_path, flat_file_path, cate_name_path, choose_pos, n_cluster=n_cluster)
    t = get_cent_words(model, words, n_cluster=n_cluster, n_max=n_max)
    df, df2 = get_clusters_df(model, category_list, app_name_list)

    re_df = pd.DataFrame(np.hstack([t.values, df2.values]))
    re_df.columns = ['키워드'+str(i) for i in range(1, n_max+1)] + ['앱의 갯수', '실제 카테고리들']
    print('Done!')

    df_ = df.pivot_table('app_name', 'cluster', 'category', aggfunc='count').T
    re_df.index = re_df.index + 1
    df_.columns = df_.columns + 1
    return re_df, df_

# utils
