from konlpy.tag import Mecab, Komoran
from utils import *

mecab = Mecab()

def space_jamo(print_util, filename):
    train_docs, app_id_list = get_data_json('./data/train_json.txt')
    space_model = Spacing(print_util)
    new_train_docs = space_model.fit(train_docs)
    save_json(filename, new_train_docs)

space_jamo(print_util=commandline_print, filename='./data/train_json_space_jamo')

