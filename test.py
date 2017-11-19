from collections import Counter, defaultdict
from konlpy.tag import Mecab, Komoran
from utils import *
mecab = Mecab()
komoran = Komoran()
file_path = './data/train_jsonl_ma_komoran.txt'

app_id_list, ratings_lists, ma_lists = read_jsonl(file_path)

major_pos = ["NNG", "NNP", "NP", "XR", "VV", "VA", "MAG", "MAJ"]

make_dict = Make_dictionay()

make_dict.fit(ma_lists, choose_pos=major_pos, normal_dict_option=1)