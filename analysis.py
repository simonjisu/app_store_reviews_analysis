from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import pandas as pd
from pprint import pprint

##
# full_morph = ['NNG', 'NNP', 'NNB', 'NR', 'NP', 'VV', 'VA', 'VX', 'VCP', 'VCN', 'MM', 'MAG', 'MAJ', 'IC']
# empty_morph = ['JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC', 'EP', 'EF', 'EC', 'ETN', 'ETM', 'XPN',
#                'XSN', 'XSV', 'XSA']
# egun = ['XR']
# signs = ['SF', 'SE', 'SS', 'SP', 'SO', 'SW']
# not_koreans = ['SL', 'SH', 'SN']

def make_keyword_dict(file_path, choose_pos, print_keywords=True, komoran_option=False, only_ma=False):
    """주요 품사별로 사전 만들기"""
    if only_ma:
        app_id_list, ma_list = read_jsonl(file_path, only_ma=only_ma)
    else:
        app_id_list, app_name_list, cate_list, rating_list, ma_list = read_jsonl(file_path, only_ma=only_ma)

    make_dict = Make_dictionay(komoran_option=komoran_option)
    make_dict.fit(ma_list, choose_pos=choose_pos, normal_dict_option=1)
    if print_keywords:
        print('최다 빈도수 단어 TOP 10')
        print('='*20)
        pprint(make_dict.word_count.most_common(10))
        print('='*20)
        print('총 단어수:', len(make_dict.word_idx))

    if only_ma:
        return make_dict, app_id_list, ma_list
    else:
        return make_dict, app_id_list, app_name_list, cate_list, rating_list, ma_list


def filter_docs_by_dict(make_dict, app_id_list, app_name_list, cate_list, rating_list, ma_list, by_app_id,
                        save_file_path=None):
    """
    사전을 기준으로 문서 단어를 추출, by_app_id 로하면 app 하나로 통합됨
    """
    ma_list, deleted_doc_index = make_dict.document_transform(ma_list)
    df = pd.DataFrame({'app_id': app_id_list,
                       'app_name': app_name_list,
                       'category': cate_list,
                       'rating': rating_list})
    df = df.loc[~df.index.isin(deleted_doc_index), :]
    app_id_list = df.iloc[:, 0].values.reshape(-1).tolist()
    app_name_list = df.iloc[:, 1].values.reshape(-1).tolist()
    cate_list = df.iloc[:, 2].values.reshape(-1).tolist()
    rating_list = df.iloc[:, 3].values.reshape(-1).tolist()

    if by_app_id:
        app_id_list, app_name_list, cate_list, rating_list, ma_list = \
            make_dict.transform_by_app_id(app_id_list, app_name_list, cate_list, rating_list, ma_list)

    save_jsonl(save_file_path, app_id_list, app_name_list, cate_list, rating_list, ma_list)


def data_processing_keyword_extract(file_path, save_file_path, major_pos):
    make_dict, app_id_list, app_name_list, cate_list, rating_list, ma_list = \
        make_keyword_dict(file_path, major_pos, print_keywords=False, komoran_option=True, only_ma=False)

    filter_docs_by_dict(make_dict, app_id_list, app_name_list, cate_list, rating_list, ma_list,
                        save_file_path=save_file_path, by_app_id=True)
    # save_json('./data/word_idx_json_by_app_id.txt', make_dict.word_idx)


def ma_transform_by_spacing(ma_list):
        documents = []
        for doc in ma_list:
            documents.append(' '.join(doc))

        return documents


def extract_keyword(matrix, vocab_idx, app_id_list, n_rank=10):
    """키워드 추출"""
    doc_len, word_len = matrix.shape
    vocab_inv = {v: k for k, v in vocab_idx.items()}

    keyword_dict = defaultdict()
    for i in tqdm(range(doc_len), desc='Extracting...', total=doc_len):
        rank_list = matrix.getrow(i).toarray().reshape(-1).argsort()[::-1]
        keyword_dict[app_id_list[i]] = [vocab_inv[r] for r in rank_list[:n_rank]]

    return keyword_dict


def get_matrix(vectorizer, spaced_ma_list, ngram_range=(1, 1), mode_NB=False, test_spaced_ma_list=None):
    vec = vectorizer(analyzer=str.split, ngram_range=ngram_range)
    if not mode_NB:
        coo_matrix = vec.fit_transform(spaced_ma_list)
        words = vec.get_feature_names()
        return coo_matrix, words, vec
    else:
        train_coo_matrix = vec.fit_transform(spaced_ma_list)
        test_coo_matrix = vec.transform(test_spaced_ma_list)
        words = vec.get_feature_names()
        return train_coo_matrix, test_coo_matrix, words, vec


def get_key_word(load_file_path, vectorizer, save_keyword_path='./data/keyword_dict_by_app_id'):
    app_id_list, app_name_list, cate_list, rating_list, ma_list = read_jsonl(load_file_path)
    spaced_ma_list = ma_transform_by_spacing(ma_list)
    coo_matrix, words, vec = get_matrix(vectorizer, spaced_ma_list, ngram_range=(1, 1), mode_NB=False)
    keyword_dict = extract_keyword(coo_matrix, vec.vocabulary_, app_id_list, n_rank=10)
    save_json(save_keyword_path, keyword_dict)

    return keyword_dict, app_id_list, cate_list, app_name_list, ma_list


# specific rate docs analysis

def create_specific_rate_docs(load_file_path, save_file_path, rate):
    app_id_list, app_name_list, cate_list, rating_list, ma_list = read_jsonl(load_file_path)
    df = pd.DataFrame({'app_id': app_id_list,
                       'app_name': app_name_list,
                       'category': cate_list,
                       'rating': rating_list,
                       'review': ma_list})

    df = df.loc[df.rating == str(rate), :]
    app_id_list = df.iloc[:, 0].values.reshape(-1).tolist()
    app_name_list = df.iloc[:, 1].values.reshape(-1).tolist()
    cate_list = df.iloc[:, 2].values.reshape(-1).tolist()
    rating_list = df.iloc[:, 3].values.reshape(-1).tolist()
    ma_list = df.iloc[:, 4].values.reshape(-1).tolist()

    save_jsonl(save_file_path, app_id_list, app_name_list, cate_list, rating_list, ma_list)


def get_specific_rate_Counter(load_file_path, vectorizer, save_keyword_path):
    keyword_dict, app_id_list, cate_list, app_name_list, ma_list = get_key_word(load_file_path, vectorizer,
                                                                                save_keyword_path=save_keyword_path)
    word_counter = Counter()
    for doc in ma_list:
        word_counter.update(doc)

    return word_counter


def main_specific(n_common):
    load_file_path = './data/docs_jsonl_ma_NB.txt'
    save_file_path_1 = './data/docs_jsonl_ma_NB_rate1'
    save_file_path_5 = './data/docs_jsonl_ma_NB_rate5'

    create_specific_rate_docs(load_file_path, save_file_path_1, 1)
    create_specific_rate_docs(load_file_path, save_file_path_5, 5)

    docs1 = get_specific_rate_Counter(save_file_path_1, vectorizer=TfidfVectorizer, save_keyword_path='./data/keyword_doc1.txt')
    docs5 = get_specific_rate_Counter(save_file_path_5, vectorizer=TfidfVectorizer, save_keyword_path='./data/keyword_doc5.txt')

    a, b = list(zip(*docs5.most_common(n_common)))
    c, d = list(zip(*docs1.most_common(n_common)))
    df = pd.DataFrame([a, b, c, d]).T
    df.columns = ['5점 문서', 'Count_5', '1점 문서', 'Count_1']
    df.index = df.index + 1

    mask1 = df.iloc[:, 0].isin(df.iloc[:, 2])  # 1점 문서에 5점짜리 단어가 있는지
    mask2 = df.iloc[:, 2].isin(df.iloc[:, 0])  # 5점 문서에 1점짜리 단어가 있는지
    a = df.loc[~mask1, ['5점 문서', 'Count_5']].reset_index(drop=True)
    b = df.loc[~mask2, ['1점 문서', 'Count_1']].reset_index(drop=True)
    table = pd.concat((a, b), axis=1)
    table.index = table.index + 1

    return table, df