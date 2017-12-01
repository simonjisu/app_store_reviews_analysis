# -*- coding: utf-8 -*-

from konlpy.tag import Twitter, Komoran, Mecab
from utils import read_jsonl, save_jsonl
from tqdm import tqdm
import sys


def get_pos(tokenizer, doc, twi_norm=True, twi_stem=True):
    if tokenizer.__class__ == Twitter:
        tokens = tokenizer.pos(doc, norm=twi_norm, stem=twi_stem)
    else:
        tokens = tokenizer.pos(doc)
    return tokens


def main():
    twitter = Twitter()
    komoran = Komoran()
    mecab = Mecab()

    argv_dict = {
                 'twitter': twitter,
                 'komoran': komoran,
                 'mecab': mecab
                }

    if len(sys.argv) < 2:
        print('please insert sys_argv \n',
              '1) tokenizer selection: twitter, komoran, mecab \n',
              '2) twitter_tokenizer_option: norm [no argv means True] \n',
              '3) twitter_tokenizer_option: stemming [no argv means True]')
    else:
        output_file_path = input('output_text_file_name: ')
        output_file_path = './data/' + output_file_path
        input_file_path = input('input_text_file_name: ')
        input_file_path = './data/' + input_file_path
        twitter_option = [bool(sys.argv[2]), bool(sys.argv[3])] if len(sys.argv) == 4 else [True, True]
        app_id_list, app_name_list, cate_list, rating_list, review_list = read_jsonl(input_file_path, key_ma=False)

        ma_list = []
        for review in tqdm(review_list, desc='tokenizing', total=len(review_list)):
            ma_tokens = get_pos(tokenizer=argv_dict[sys.argv[1]], doc=review, twi_norm=twitter_option[0], twi_stem=twitter_option[1])
            ma_list.append(ma_tokens)

        save_jsonl(output_file_path, app_id_list, app_name_list, cate_list, rating_list, ma_list)

main()
