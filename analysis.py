from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np

##
# full_morph = ['NNG', 'NNP', 'NNB', 'NR', 'NP', 'VV', 'VA', 'VX', 'VCP', 'VCN', 'MM', 'MAG', 'MAJ', 'IC']
# empty_morph = ['JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC', 'EP', 'EF', 'EC', 'ETN', 'ETM', 'XPN',
#                'XSN', 'XSV', 'XSA']
# egun = ['XR']
# signs = ['SF', 'SE', 'SS', 'SP', 'SO', 'SW']
# not_koreans = ['SL', 'SH', 'SN']

def make_keyword_dict(file_path, choose_pos, print_keywords=True, mecab_option=False):
    """주요 품사별로 사전 만들기"""
    app_id_list, ratings_lists, ma_lists = read_jsonl(file_path)
    make_dict = Make_dictionay(mecab_option=mecab_option)
    make_dict.fit(ma_lists, choose_pos=choose_pos, normal_dict_option=1)
    if print_keywords:
        print('최다 빈도수 단어 TOP 10')
        print('===============')
        print(make_dict.word_count.most_common(10))

    return make_dict, app_id_list, ratings_lists, ma_lists

##

def filter_docs_by_dict(make_dict, ma_lists, ratings_lists=None, save_file_path=None, by_app_id=False):
    """
    사전을 기준으로 문서 단어를 추출
    save_file_path가 rating_list있을 때 list로 넣을 것
    """
    if ratings_lists:
        filtered_docs, filtered_ratings = make_dict.document_transform(ma_lists, ratings_lists)
        boundary = get_boundary(filtered_docs)
        flat_docs = make_dict.flatten_all_docs(filtered_docs, by_app_id=by_app_id)
        flat_ratings = make_dict.flatten_all_docs(filtered_ratings, by_app_id=by_app_id)
        make_dict.save_as_file(save_file_path[0], flat_docs)
        make_dict.save_as_file(save_file_path[1], flat_ratings)

        return boundary

    else:
        filtered_docs = make_dict.document_transform(ma_lists)
        flat_docs = make_dict.flatten_all_docs(filtered_docs, by_app_id=by_app_id)
        make_dict.save_as_file(save_file_path, flat_docs)
        boundary = get_boundary(filtered_docs)

        return boundary

def extract_key_word(matrix, vocab_idx, app_id_list, n_rank=10):
    """키워드 추출"""
    doc_len, word_len = matrix.shape
    vocab_inv = {v: k for k, v in vocab_idx.items()}

    keyword_dict = defaultdict()
    for app_id, i in tqdm(zip(app_id_list, range(doc_len)), desc='Extracting...', total=len(app_id_list)):
        rank_list = sorted(range(word_len), key=lambda k: matrix[i, :][k], reverse=True)
        keyword_dict[app_id] = [vocab_inv[r] for r in rank_list[:n_rank]]

    return keyword_dict

def get_key_word(cate_name_path, file_path_by_app_id, app_id_list, make_dict, ma_lists):

    category_list, app_name_list = get_app_name_category(cate_name_path, app_id_list)
    _ = filter_docs_by_dict(make_dict, ma_lists, save_file_path=file_path_by_app_id, by_app_id=True)

    flat_docs = make_dict.load_file(file_path_by_app_id, option_split=False)
    tfidf_vec = TfidfVectorizer(analyzer=str.split, vocabulary=make_dict.word_idx)
    keyword_dict = extract_key_word(tfidf_vec.fit_transform(flat_docs).toarray(), make_dict.word_idx,
                                    app_id_list, n_rank=10)

    return keyword_dict, category_list, app_name_list

##

def data_processing(jsonl_file_path, save_file_path, choose_pos, option_by_app_id=True):
    """
    jsonl_file_path: 형태소 처리된 파일 경로
    save_file_path: flat한 document 저장경로
    choose_pos: 원하는 형태소 선택, 하나의 리스트
    """
    make_dict, app_id_list, ratings_lists, ma_lists = make_keyword_dict(jsonl_file_path, choose_pos, print_keywords=False)
    filter_docs_by_dict(make_dict, ma_lists, ratings_lists, save_file_path=save_file_path, by_app_id=option_by_app_id)

    return make_dict, app_id_list

def data_loading(save_file_path, make_dict):
    flat_docs = make_dict.load_file(save_file_path[0], option_split=False)
    flat_ratings = make_dict.load_file(save_file_path[1], option_split=True)

    return flat_docs, flat_ratings


def get_tfidf_matrix(flat_docs, make_dict):
    tfidf_vec = TfidfVectorizer(analyzer=str.split, vocabulary=make_dict.word_idx)
    coo_matrix = tfidf_vec.fit_transform(flat_docs)
    words = tfidf_vec.get_feature_names()

    return coo_matrix, words

def get_boundary(filtered_docs):
    li = [0]
    k = 0
    for i in range(len(filtered_docs)):
        k += len(filtered_docs[i])
        li.append(k)

    return li

# specific rate docs analysis
def get_app_id_dict_from_boundary(boundary, app_id_list):
    app_id_dict = defaultdict()
    for app_id, (b1, b2) in zip(app_id_list, zip(boundary, boundary[1:]+[None])):
        app_id_dict[app_id] = (b1, b2)

    reverse_ = {v: k for k, v in app_id_dict.items()}

    return app_id_dict, reverse_

def check_isin_id(reverse_app_id_dict, i):
    for (b1, b2), v in reverse_app_id_dict.items():
        if (i < b2) & (i >= b1):
            break
    return v

def create_specific_rate_docs(rate, boundary, app_id_list, test_y, test_X):
    app_id_dict, reverse_app_id_dict = get_app_id_dict_from_boundary(boundary, app_id_list)
    mask = test_y.astype(np.int).reshape(-1) == rate
    docs = []
    docs_app_id = []
    orderd_set_docs_app_id = []
    for i in tqdm(np.arange(len(mask))[mask], desc='Creating Docs...', total=sum(mask)):
        docs.append(test_X[i])
        app_id = check_isin_id(reverse_app_id_dict, i)
        docs_app_id.append(app_id)

    for app_id in docs_app_id:
        if app_id not in orderd_set_docs_app_id:
            orderd_set_docs_app_id.append(app_id)

    return docs, orderd_set_docs_app_id

def main_choose_specific_rate(rate, test_y, test_X, test_boundary, test_app_id_list, option=True):

    docs, docs_app_id = create_specific_rate_docs(rate, test_boundary, test_app_id_list, test_y, test_X)
    word_counter = Counter()
    for doc in docs:
        word_counter.update(doc.split(' '))

    word_idx = {k: i for i, k in enumerate(word_counter.keys())}
    tfidf = TfidfVectorizer(analyzer=str.split, vocabulary=word_idx)
    matrix = tfidf.fit_transform(docs)
    keyword_dict = extract_key_word(matrix.toarray(), tfidf.vocabulary_, docs_app_id)
    if option:
        choose_pos = ["NNG", "NNP", "NP", "XR", "VV", "VA", "MAG", "MAJ"]
    else:
        full_morph = ['NNG', 'NNP', 'NNB', 'NR', 'NP', 'VV', 'VA', 'VX', 'VCP', 'VCN', 'MM', 'MAG', 'MAJ', 'IC']
        empty_morph = ['JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC', 'EP', 'EF', 'EC', 'ETN', 'ETM',
                       'XPN',
                       'XSN', 'XSV', 'XSA']
        egun = ['XR']
        not_koreans = ['SL', 'SH', 'SN']
        choose_pos = full_morph + egun + not_koreans + empty_morph

    total_words = list(set([w for words in keyword_dict.values() for w in words]))
    choosen_words = [w for w in total_words if w.split('/')[1] in choose_pos]
    choosen_words = [w+'다' if w.split('/')[1] in ['VV', 'VA'] else w for w in choosen_words ]
    choosen_words_count = Counter({w: word_counter[w] for w in choosen_words})

    return choosen_words_count

def main_specific(n_common, test_y, test_X, test_boundary, test_app_id_list, option=True):

    docs5 = main_choose_specific_rate(5, test_y, test_X, test_boundary, test_app_id_list, option=option)
    docs1 = main_choose_specific_rate(1, test_y, test_X, test_boundary, test_app_id_list, option=option)

    a, b = list(zip(*docs5.most_common(n_common)))
    c, d = list(zip(*docs1.most_common(n_common)))
    df = pd.DataFrame([a, b, c, d]).T
    df.columns = ['5점 문서', 'Count', '1점문서', 'Count']
    df.index = df.index + 1
    table = df.loc[~df.iloc[:, 0].isin(df.iloc[:, 2]), :]

    return table, df