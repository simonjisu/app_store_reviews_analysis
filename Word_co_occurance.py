from analysis import *
from itertools import combinations
from operator import itemgetter
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_extraction.text import CountVectorizer
import sys

def Word_co_occurance_data_processing():
    file_path = './data/docs_jsonl_ma_komoran_after_processing.txt'
    save_file_path = './data/docs_jsonl_ma_wordco.txt'
    major_pos = ["NNG", "NNP", "NP", "XR"]
    make_dict, app_id_list, app_name_list, cate_list, rating_list, ma_list = \
        make_keyword_dict(file_path, major_pos, print_keywords=False, komoran_option=True, only_ma=False)

    filter_docs_by_dict(make_dict, app_id_list, app_name_list, cate_list, rating_list, ma_list,
                        save_file_path=save_file_path, by_app_id=False)


def get_matrix_by_countvec(spaced_ma_list, ngram_range, max_feature):
    vec = CountVectorizer(analyzer=str.split, ngram_range=ngram_range, binary=True, max_features=max_feature)
    coo_matrix = vec.fit_transform(spaced_ma_list)
    words = vec.get_feature_names()

    return coo_matrix, words, vec


def get_tt_matrix(coo_matrix):
    tt_mat = coo_matrix.T * coo_matrix
    tt_mat.setdiag(0)

    return tt_mat


def get_word_sim_mat(tt_mat):
    word_sim_mat = pdist(tt_mat.toarray(), metric='correlation')
    word_sim_mat = squareform(word_sim_mat)

    return word_sim_mat


def get_sorted_words_by_similarity(output_file_path, word_sim_mat, words, option_reverse):
    word_sims = []

    for i, j in combinations(range(len(words)), 2):
        sim = word_sim_mat[i, j]
        if sim == 0:
            continue

        word_sims.append((words[i], words[j], sim))

    sorted_word_sims = sorted(word_sims, key=itemgetter(2), reverse=option_reverse)

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for word_i, word_j, sim in sorted_word_sims:
            print('{}\t{}\t{}'.format(word_i, word_j, sim), file=output_file)

    return word_sims, sorted_word_sims


# Net word 그림그리기
def build_word_sim_network(word_sims, n_words_coocs):

    G = nx.Graph()

    for word1, word2, sim in word_sims[:n_words_coocs]:
        G.add_edge(word1, word2, weight=sim)

    T = nx.minimum_spanning_tree(G)

    return T


def get_font_name():
    """플랫폼별로 사용할 글꼴 이름을 돌려준다."""

    if sys.platform in ["win32", "win64"]:
        font_name = "malgun gothic"
    elif sys.platform == "darwin":
        font_name = "AppleGothic"

    return font_name


def draw_network(G):
    """어휘 공기 네트워크를 화면에 표시한다."""
    font_name = get_font_name()
    nx.draw_networkx(G,
                     pos=nx.spring_layout(G, k=0.8),
                     node_size=1000,
                     node_color="yellow",
                     font_family=font_name,
                     label_pos=0,  # 0=head, 0.5=center, 1=tail
                     with_labels=True,
                     font_size=13)

    plt.axis("off")
    # plt.savefig("graph.png")
    plt.show()


def word_co(load_file_path, output_file_path, n_words_coocs, max_feature=500, ngram_range=(1, 1), option_reverse=True):

    app_id_list, app_name_list, cate_list, rating_list, ma_list = read_jsonl(load_file_path)
    spaced_ma_list = ma_transform_by_spacing(ma_list)

    coo_matrix, words, vec = get_matrix_by_countvec(spaced_ma_list, ngram_range, max_feature)
    # term-term matrix
    tt_mat = get_tt_matrix(coo_matrix)

    # 거리 및 유사도 계산
    word_sim_mat = get_word_sim_mat(tt_mat)

    # 유사도에 의한 유사단어 추출
    word_sims, sorted_word_sims = get_sorted_words_by_similarity(output_file_path, word_sim_mat, words, option_reverse)

    # Social Network Analysis !!
    T = build_word_sim_network(sorted_word_sims, n_words_coocs)
    draw_network(T)