from analysis import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def NB_data_processing(jsonl_file_path, flat_file_path, choose_pos):

    make_dict, app_id_list, ratings_lists, ma_lists = make_keyword_dict(jsonl_file_path, choose_pos=choose_pos, print_keywords=False, mecab_option=True)
    boundary = filter_docs_by_dict(make_dict, ma_lists, ratings_lists, save_file_path=flat_file_path, by_app_id=False)
    # X, y
    flat_docs, flat_ratings = data_loading(save_file_path=flat_file_path, make_dict=make_dict)

    return make_dict, flat_docs, flat_ratings, boundary, app_id_list

def NB_model(vectorizer, make_dict, X, y, ngram_range=(1, 2)):
    model = Pipeline([
                ('vect', vectorizer(analyzer=str.split, vocabulary=make_dict.word_idx, ngram_range=ngram_range)),
                ('mb', MultinomialNB()),
            ])

    model.fit(X, np.array(y).ravel())

    return model

def NB_train(vectorizer, ngram_range=(1, 2), option_major_pos=True):
    jsonl_file_path = './data/train_jsonl_ma_komoran_after_processing.txt'
    flat_file_path = ['./data/train_flat_docs.txt', './data/train_flat_ratings.txt']

    if option_major_pos:
        major_pos = ["NNG", "NNP", "NP", "XR", "VV", "VA", "MAG", "MAJ"]
        choose_pos = major_pos
    else:
        full_morph = ['NNG', 'NNP', 'NNB', 'NR', 'NP', 'VV', 'VA', 'VX', 'VCP', 'VCN', 'MM', 'MAG', 'MAJ', 'IC']
        empty_morph = ['JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC', 'EP', 'EF', 'EC', 'ETN', 'ETM', 'XPN',
                       'XSN', 'XSV', 'XSA']
        egun = ['XR']
        not_koreans = ['SL', 'SH', 'SN']
        choose_pos = full_morph + egun + not_koreans + empty_morph

    # build model
    print('** Training Model **')
    make_dict, flat_docs, flat_ratings, boundary, _ = NB_data_processing(jsonl_file_path, flat_file_path, choose_pos)
    model = NB_model(vectorizer, make_dict, flat_docs, flat_ratings, ngram_range=ngram_range)

    # flat_docs = [doc.split(' ') for doc in flat_docs]

    return model, boundary, flat_docs, np.array(flat_ratings)

def NB_test(model, option_major_pos=True):
    test_file_path = './data/test_jsonl_ma_komoran_after_processing.txt'
    flat_file_path = ['./data/test_flat_docs.txt', './data/test_flat_ratings.txt']

    if option_major_pos:
        major_pos = ["NNG", "NNP", "NP", "XR", "VV", "VA", "MAG", "MAJ"]
        choose_pos = major_pos
    else:
        full_morph = ['NNG', 'NNP', 'NNB', 'NR', 'NP', 'VV', 'VA', 'VX', 'VCP', 'VCN', 'MM', 'MAG', 'MAJ', 'IC']
        empty_morph = ['JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC', 'EP', 'EF', 'EC', 'ETN', 'ETM', 'XPN',
                       'XSN', 'XSV', 'XSA']
        egun = ['XR']
        not_koreans = ['SL', 'SH', 'SN']
        choose_pos = full_morph + egun + not_koreans + empty_morph

    print('** Testing Model **')

    _, test_flat_docs, test_flat_ratings, test_boundary, test_app_id_list = NB_data_processing(test_file_path, flat_file_path, choose_pos)
    pred_y = model.predict(test_flat_docs)

    # test_flat_docs = [doc.split(' ') for doc in test_flat_docs]

    print('Done!\n')
    return pred_y, test_boundary, test_flat_docs, np.array(test_flat_ratings), test_app_id_list


def compare_pos(vectorizer, report=True):
    print('==================================\n', '** Major Pos **')
    print('==================================\n')

    model1, _, _, _ = NB_train(vectorizer, ngram_range=(1, 2), option_major_pos=True)
    pred_y1, _, _, test_y1, _ = NB_test(model1, option_major_pos=True)
    ac1 = accuracy_score(test_y1, pred_y1) * 100

    print('==================================\n', '** All Pos **')
    print('==================================\n')

    model2, _, _, _ = NB_train(vectorizer, ngram_range=(1, 2), option_major_pos=False)
    pred_y2, _, _, test_y2, _ = NB_test(model2, option_major_pos=False)
    ac2 = accuracy_score(test_y2, pred_y2) * 100

    print('==================================\n')
    print('Using Major Pos, Accuracy Score is {0:.2f}%,\n Using All Pos, Accuracy Score is {1:.2f}%.\n'.format(ac1, ac2))
    if report:
        print('==================================\n', 'Classification Report: Major Pos')
        print('==================================\n')
        print(classification_report(test_y1, pred_y1))
        print('==================================\n', 'Classification Report: All Pos')
        print('==================================\n')
        print(classification_report(test_y2, pred_y2))

