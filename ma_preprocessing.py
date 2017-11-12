# -*- coding: utf-8 -*-

from collections import defaultdict
from konlpy.tag import Twitter, Komoran, Mecab
import ujson
import sys


def get_data_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        texts = file.read()
        data = ujson.loads(texts)
        app_ids_list = list(data.keys())
    return data, app_ids_list


def get_pos(tokenizer, doc, twi_norm=True, twi_stem=True):
    if tokenizer.__class__ == Twitter:
        tokens = tokenizer.pos(doc, norm=twi_norm, stem=twi_stem)
    else:
        tokens = tokenizer.pos(doc)
    return tokens


def save_pos_json(filename, docs, app_id_list, tokenizer):
    with open(filename, 'w', encoding='utf-8') as output_file:
        json_doc = defaultdict(list)
        for app_id in app_id_list:
            reviews = docs[app_id]['reviews']
            for review in reviews:
                ma = get_pos(tokenizer, review, twi_norm=True, twi_stem=False)
                json_doc[app_id].append(ma)
            print(app_id)
        json_str = ujson.dumps(json_doc, ensure_ascii=False)
        print(json_str, file=output_file)


def main():
    twitter = Twitter()
    komoran = Komoran()
    mecab = Mecab()

    train_data, train_app_id_list = get_data_json('./data/train_json.txt')
    test_data, test_app_id_list = get_data_json('./data/test_json.txt')

    argv_dict = {'tokenizer': {'twitter': twitter,
                               'komoran': komoran,
                               'mecab': mecab},
                 'data_type': {'train': [train_data, train_app_id_list],
                               'test': [test_data, test_app_id_list]},
                 }

    if len(sys.argv) < 3:
        print('please insert sys_argv \n',
              '1) train or test \n',
              '2) tokenizer selection: twitter, komoran, mecab')
    else:
        name = input('text_file_name:')
        file_path = './data/' + name + '.txt'
        save_pos_json(filename=file_path,
                      docs=argv_dict['data_type'][sys.argv[1]][0],
                      app_id_list=argv_dict['data_type'][sys.argv[1]][1],
                      tokenizer=argv_dict['tokenizer'][sys.argv[2]])

main()
