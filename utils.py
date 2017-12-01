# -*- coding: utf-8 -*-

from itertools import chain, islice
from collections import deque, defaultdict, Counter
from tqdm import tqdm
import pandas as pd
import numpy as np
import mmap
import os
import ujson
import re

class Post_ma(object):
    def __init__(self, file_path, tokenizer, auto_load=True):
        """
        사용법:
        1. update를 사용해 규칙사전을 등록, 2번째 부터는 자동으로 불러오기함
        2. 규칙을 로드하면 change하면됨
        """
        self.file_path = file_path
        self.post_ma_pairs = []
        self.tokenizer = tokenizer
        if auto_load:
            if os.path.exists(self.file_path):
                self.load()
                print('Load Complete! total {} rules'.format(len(self.post_ma_pairs)))
            else:
                print('Can not load file: {}! '.format(self.file_path),
                      'There is no file, please create file by update method(take option write=True)')

    # update part
    def update(self, ma_pairs, write=False):
        """[([before_ma, before_pos], [after_ma, after_pos]), (), ()...]"""
        button = 'w' if write else 'a'
        with open(self.file_path, button, encoding='utf-8') as output_file:
            output_file.write('')
            duplicated_index = []
            for ma in ma_pairs:
                if not self.check_duplicate(ma):
                    text = self.split_before_after(ma)
                    print(text, file=output_file)
                else:
                    duplicated_index.append(ma_pairs.index(ma))

            if len(duplicated_index) != 0:
                print('There are duplicated rules! find you input pairs in index of {}'.format(duplicated_index))

    def check_duplicate(self, ma):
        if ma in self.post_ma_pairs:
            return True
        else:
            return False

    def split_before_after(self, ma):
        before, after = ma
        before = list(chain.from_iterable(before))
        after = list(chain.from_iterable(after))
        text = '\t'.join(before) + '\t>>\t' + '\t'.join(after)
        return text

    # delete part
    def delete(self, ma_pairs):
        pass

    # load part
    def load(self):
        """[([[before_ma, before_pos]], [[after_ma, after_pos]]), (), ()...]"""
        with open(self.file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            result = []
            for line in lines:
                result.append(self.combinate_before_after(line))
        self.post_ma_pairs = result

    def combinate_before_after(self, line):
        """before >> after"""
        before, after = line.strip().split('>>')
        return tuple((self.tab_str_to_double_list(before),
                      self.tab_str_to_double_list(after)))

    def tab_str_to_double_list(self, string):
        """string = ma \t pos \t ma \t pos..."""
        tokens = string.strip().split('\t')
        n = len(tokens)
        result = []
        for i in range(0, n + 1, 2):
            if i == 0:
                continue
            else:
                pair = list(deque(islice(tokens, i), maxlen=2))
                result.append(pair)
        return result

    # change part post_ma
    # 1. 바꾸고 싶은 단어를 등록한다. [( [[단어, 형태소]] , [[바뀐 단어, 형태소]] )]
    # 2. NA 중에 바꾸고 싶은 단어가 있는지 확인하고 스페이싱을 한다.
    # 3. 형태소 분석을 한다.
    # 4. 분리된 형태소에서 단어를 바꿔준다.

    def find_sublists(self, seq, sublist):
        length = len(sublist)
        for index, value in enumerate(seq):
            if value == sublist[0] and seq[index:index + length] == sublist:
                yield index, index + length

    def replace_sublist(self, seq, target, replacement, maxreplace=None):
        sublists = self.find_sublists(seq, target)
        if maxreplace:
            sublists = islice(sublists, maxreplace)
        for start, end in sublists:
            seq[start:end] = replacement

    def search_word_and_spacing(self, target_list, target_idx, replacement):
        repl_len = len(replacement[0])
        partial_start_idx = [m.start() for m in re.finditer(replacement[0], target_list[target_idx][0])]  # start_idxes
        partial_end_idx = [i + repl_len for i in partial_start_idx]
        partial_idx = sorted(partial_start_idx + partial_end_idx)
        partial_words = [target_list[target_idx][0][i:j] for i, j in zip(partial_idx, partial_idx[1:]+[None])]
        target_list[target_idx][0] = ' '.join(partial_words)

    def try_tokenizing(self, target_list, target_idx):
        tokens = self.tokenizer.pos(target_list[target_idx][0])
        self.replace_list_elem(target_list, target_idx, tokens)

    def replace_list_elem(self, target_list, target_idx, replacement):
        target_list.pop(target_idx)
        for i, repl in enumerate(replacement, target_idx):
            target_list.insert(i, list(repl))

    def get_location_dict_by_ma(self, ma, location_dict):
        """
        ma: 업뎃  ->  location_dict key에 있는지 확인
        loc_dict: {'업뎃해주세요': [위치], '업뎃좀':[위치1, 위치2]}
        """
        keys = [k for k in location_dict.keys()]
        loc_dict = {k: location_dict[k] for k in keys if ma in k}

        return loc_dict

    def find_ma_loc(self, ma_doc, key):
        for i, ma in enumerate(ma_doc):
            if ma[0] == key:
                return i

    def split_method(self, ma_docs, ma_set, loc_dict):
        """
        ma_set = ([ma, pos], [ma, pos])
        앞에서 loc_dict에 key는 문자열 하나기 때문에 loc_info에는 doc_loc와 ma_loc둘다 저장됨
        """
        before = ma_set[0][0]
        after = ma_set[1]
        for key in tqdm(loc_dict.keys(), desc='Processings', total=len(loc_dict)):
            for doc_loc in loc_dict[key]:
                ma_loc = self.find_ma_loc(ma_docs[doc_loc], key)
                self.search_word_and_spacing(ma_docs[doc_loc], ma_loc, before)
                self.try_tokenizing(ma_docs[doc_loc], ma_loc)
                self.replace_list_elem(ma_docs[doc_loc], ma_loc, after)

    def merge_method(self, ma_docs, ma_set, loc_info):
        """
        앞에서 loc_dict에 key로 list를 쓸수 없기 때문에 loc_info에는 doc_loc 만 들어가게됨,
        """
        for app_loc, doc_loc in loc_info:
            self.replace_sublist(ma_docs[app_loc][doc_loc], ma_set[0], ma_set[1])

    def replace_ma_docs(self, ma_docs, location_dict):  ## 이미 처리했던거 기록할 필요가 있음
        """
        location_dict: 바꾸고 싶은 단어, 모르는 단어들의 위치(전체)
        split: {'ma': [doc_loc1, doc_loc2], 'ma2':...]
        merge: {'ma1/ma2': [doc_loc1, doc_loc2], ...}
        """
        if not self.post_ma_pairs:
            print('Plead load post_ma_rules first')
            return None

        for ma_set in self.post_ma_pairs:
            if len(ma_set[0]) == 1:  # before -> after: split 일때
                ma_key = ma_set[0][0][0]
                loc_dict = self.get_location_dict_by_ma(ma_key, location_dict)
                self.split_method(ma_docs, ma_set, loc_dict)
            elif len(ma_set[0]) > 1:  # before -> after: merge 일때
                ma_key = '/'.join([word for word in ma_set[0]])
                self.merge_method(ma_docs, ma_set, location_dict[ma_key])
            else:
                print('error, no before word')

        return ma_docs

###############################################################

class Unknown_words(object):
    def __init__(self, ):
        self.unknown_loc_dict = defaultdict(list)

    def get_unknown_words(self, ma_docs):
        unknown_list = []
        total_ma_count = 0
        for doc_loc, doc in tqdm(enumerate(ma_docs), desc='Extracting unknowns:', total=len(ma_docs)):
            for ma in doc:
                if ma[1] in ['UNKNOWN', 'NA']:
                    unknown_list.append(ma[0])
                    self.unknown_loc_dict[ma[0]].append(doc_loc)
                total_ma_count += len(doc)

        unique_list = list(set(unknown_list))

        print('Unknowns(중복제거): {}, Total Unknown: {}, Total MA: {}'.format(len(unique_list),
                                                                            len(unknown_list), total_ma_count))

        return unique_list

    # shape changer
    def shape_changer(self, dictionary, tokenizer_name='komoran'):

        unk_pos = 'UNKNOWN' if tokenizer_name == 'twitter' else 'NA'
        changed_ma_list = []
        for before, after in dictionary.items():
            changed_ma_list.append(tuple([[tuple([before, unk_pos])], after]))

        return changed_ma_list

###############################################################
# Dictionary
###############################################################

class Make_dictionay(object):
    def __init__(self, komoran_option=True):
        self.maxwords = None
        self.word_count = None
        self.choose_pos_list = None
        self.min_word_length = 1
        self.komoran_option = komoran_option
        self.normal_dict_option = None
        self.word_idx = None


    def get_word_count(self, ma_list):
        """단어를 품사별로 묶어서 사전을 만든다 '단어, 품사' """

        # dictionary 종류 고르기
        self.set_normal_dict_option()
        for ma_doc in tqdm(ma_list, desc="Counting Words...", total=len(ma_list)):
            ma_doc = self.filter_ma(ma_doc)
            self.count_words(ma_doc)

        if (len(self.word_count.keys()) > self.maxwords) & self.normal_dict_option:  # normal_dict 에만 적용
            self.word_count = Counter({vals[0]: vals[1] for vals in self.word_count.most_common(self.maxwords)})

        return self

    def set_normal_dict_option(self):
        if self.normal_dict_option == 0:
            self.word_count = Counter()
        elif self.normal_dict_option == 1:
            self.word_count = Counter()
        elif self.normal_dict_option == 2:
            self.word_count = defaultdict(Counter)

        return self

    def count_words(self, ma_doc):
        if self.normal_dict_option == 0:
            self.word_count.update(ma_doc)
        elif self.normal_dict_option == 1:
            ma_doc = ['/'.join(ma) for ma in ma_doc]
            self.word_count.update(ma_doc)
        elif self.normal_dict_option == 2:
            for lex, pos in ma_doc:
                self.word_count[lex][pos] += 1

        return self

    def filter_ma(self, ma_doc):
        if self.choose_pos_list:
            ma_doc = [(lex, pos) for (lex, pos) in ma_doc if self.is_pos_in(pos)]
            # 띄어써진 단어들 붙여주기
            ma_doc = [(''.join(lex.split(' ')), pos) if ' ' in lex else (lex, pos) for (lex, pos) in ma_doc]

        if self.komoran_option:  # komoran 품사단어 뒤에 +다 붙여주기
            ma_doc = [(lex + '다', pos) if pos in ['VV', 'VA'] else (lex, pos) for (lex, pos) in ma_doc]

        if self.min_word_length >= 2:  # 2글자 이상만 고르기
            ma_doc = [(lex, pos) for (lex, pos) in ma_doc if len(lex) >= 2]

        #  NA 값은 NULL로 넣기
        ma_doc = [('NULL', pos) if pos in ['UNKNOWN', 'NA'] else (lex, pos) for (lex, pos) in ma_doc]

        return ma_doc

    def is_pos_in(self, pos):
        if pos in self.choose_pos_list:
            return True
        else:
            return False

    def get_word_idx(self):
        self.word_idx = {w: i for i, w in enumerate(self.word_count.keys())}

    def fit(self, ma_list, choose_pos=None, min_word_length=1, maxwords=100000, normal_dict_option=1):
        """
        delete_pos: 지우고 싶은 품사는 지울 수 있음 list형태
        maxwords: 설정하고자 하는 최대 단어 수, most_common 순으로 추출된다, 안하려면 None 으로 설정할 것
        inverse: 사전idx에 대한 역인덱싱 사전도 구함
        normal_dict_option:
            - 0 {(단어, 품사): 카운트}
            - 1 {(단어/품사}: 카운트}
            - 2 {단어: {품사: 카운트}}
        """
        self.maxwords = maxwords
        self.normal_dict_option = normal_dict_option
        if choose_pos:
            self.choose_pos_list = choose_pos
        if min_word_length >= 2:
            self.min_word_length = min_word_length

        self.get_word_count(ma_list)
        self.get_word_idx()

    # def update(self, new_dict):

    ####### only for dictionary option 1
    def filter_by_dict(self, doc):
        if self.normal_dict_option == 0:
            doc = [ma for ma in doc if ma in self.word_count.keys()]
        elif self.normal_dict_option == 1:
            doc = ['/'.join(ma) for ma in doc if '/'.join(ma) in self.word_count.keys()]

        return doc

    def document_transform(self, ma_list):
        """사전에 존재하는 것들만 문서로 바꿔준다 app_id별로 [[word1, word2, ...], [...] 순서는 상관 없음"""
        documents = []
        deleted_doc_index = []
        for i, doc in tqdm(enumerate(ma_list), desc="Filtering...", total=len(ma_list)):
            filtered_doc = self.filter_by_dict(doc)
            if len(filtered_doc) > 0:
                documents.append(filtered_doc)
            else:
                deleted_doc_index.append(i)

        return documents, deleted_doc_index

    def transform_by_app_id(self, app_id_list, app_name_list, cate_list, rating_list, ma_list):
        trans_docs = []
        trans_app_id_list = []
        trans_app_name_list = []
        trans_cate_list = []
        trans_rating_list = []

        for i, app_id in tqdm(enumerate(app_id_list), desc='Transforming by app_id...', total=len(app_id_list)):
            if i == 0:
                temp = ma_list[i]
                temp2 = [rating_list[i]]
                trans_app_id_list.append(app_id_list[i])
                trans_app_name_list.append(app_name_list[i])
                trans_cate_list.append(cate_list[i])
                previous_app_id = app_id
                continue

            if previous_app_id == app_id:
                temp = temp + ma_list[i]
                temp2 = temp2 + [rating_list[i]]

            else:
                trans_docs.append(temp)
                trans_rating_list.append(temp2)
                temp = ma_list[i]
                temp2 = [rating_list[i]]
                trans_app_id_list.append(app_id_list[i])
                trans_app_name_list.append(app_name_list[i])
                trans_cate_list.append(cate_list[i])
                previous_app_id = app_id

            if i == len(app_name_list) - 1:
                trans_docs.append(temp)
                trans_rating_list.append(temp2)

        return trans_app_id_list, trans_app_name_list, trans_cate_list, trans_rating_list, trans_docs

###################################################################
# Spacing model
###################################################################

class Spacing(object):
    def __init__(self):
        self.app_id_list = None
        self.check_list = None
        self.basic_check_list = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ',
                                 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ',
                                 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ', 'ㄳ', 'ㄵ',
                                 'ㄶ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅄ']

    def fit(self, review_list, add_check_list=None):
        if add_check_list:
            self.check_list = self.basic_check_list + add_check_list
        else:
            self.check_list = self.basic_check_list

        for i, review in tqdm(enumerate(review_list), desc='Processing spacing', total=len(review_list)):
            review_list[i] = self.space_jamo(review)

        return review_list

    def space_jamo(self, review):
        if not self.check_list:
            self.check_list = self.basic_check_list

        review_splited = [char for char in review]
        space_idx = self.get_spacing_index(review_splited)
        tokens_list = self.split_doc_to_words(review, space_idx)
        new_text = ' '.join(tokens_list)

        return new_text

    def split_doc_to_words(self, review, space_idx):
        tokens_list = []
        for i in range(2, len(space_idx) + 1):
            start, stop = deque(islice(space_idx, i), maxlen=2)
            tokens_list.append(review[start:stop])

        return tokens_list

    def get_spacing_index(self, review_splited):
        space_idx = []
        doc_len = len(review_splited)
        for i in range(1, doc_len+1):
            window = deque(islice(review_splited, i), maxlen=2)  # 번째 원소 전까지 window가 움직이는거
            if self.check_jamo(window):
                space_idx.append(i - 1)
        # slicing위한 작업
        if 0 not in space_idx:
            space_idx.insert(0, 0)
        if doc_len not in space_idx:
            space_idx.insert(len(space_idx), doc_len)

        return space_idx

    def check_jamo(self, window):

        mask = [True if char in self.check_list else False for char in window]
        if sum(mask) == 1:
            return True
        else:
            return False

#### utils:

def save_json(filepath, docs):
    with open(filepath, 'w', encoding='utf-8') as file:
        texts = ujson.dumps(docs, ensure_ascii=False)
        print(texts, file=file)


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        texts = file.read()
        data = ujson.loads(texts)
    return data


def read_jsonl(filepath, key_ma=True, only_ma=False):
    key_app_id = 'app_id'
    key_rating = 'rating'
    key_cate = 'category'
    key_app_name = 'app_name'
    if key_ma:
        key_ma = 'ma'
    else:
        key_ma = 'review'
    app_id_list = []
    rating_list = []
    ma_list = []
    cate_list = []
    app_name_list = []
    if only_ma:
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in tqdm(file, desc='Reading Files', total=get_num_lines(filepath)):
                doc = ujson.loads(line)
                app_id_list.append(doc[key_app_id])
                ma_list.append(doc[key_ma])
        return app_id_list, ma_list

    else:
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in tqdm(file, desc='Reading Files', total=get_num_lines(filepath)):
                doc = ujson.loads(line)
                app_id_list.append(doc[key_app_id])
                rating_list.append(doc[key_rating])
                ma_list.append(doc[key_ma])
                cate_list.append(doc[key_cate])
                app_name_list.append(doc[key_app_name])

        return app_id_list, app_name_list, cate_list, rating_list, ma_list


def save_jsonl(filepath, app_id_list, app_name_list, cate_list, rating_list, ma_list, key_ma=True, only_ma=False):
    key_app_id = 'app_id'
    key_rating = 'rating'
    key_cate = 'category'
    key_app_name = 'app_name'
    if key_ma:
        key_ma = 'ma'
    else:
        key_ma = 'review'

    if only_ma:
        with open(filepath, 'w', encoding='utf-8') as file:
            doc_len = len(app_id_list)
            for i in tqdm(range(doc_len), desc='Saving documents', total=doc_len):
                json_dict = {key_app_id: app_id_list[i],
                             key_ma: ma_list[i]}
                line = ujson.dumps(json_dict, ensure_ascii=False)
                print(line, file=file)
    else:
        with open(filepath, 'w', encoding='utf-8') as file:
            doc_len = len(app_id_list)
            for i in tqdm(range(doc_len), desc='Saving documents', total=doc_len):
                json_dict = {key_app_id: app_id_list[i],
                             key_rating: rating_list[i],
                             key_ma: ma_list[i],
                             key_cate: cate_list[i],
                             key_app_name: app_name_list[i]}
                line = ujson.dumps(json_dict, ensure_ascii=False)
                print(line, file=file)


def get_num_lines(filename):
    """빠른 속도로 텍스트 파일의 줄 수를 세어 돌려준다.
    https://blog.nelsonliu.me/2016/07/29/progress-bars-for-python-file-reading-with-tqdm/
    """

    fp = open(filename, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


# dataframe utils
def color_zero_white(val):
    color = 'white' if val == 0 else 'black'
    return 'color: %s' % color


def color_red(val):
    color = 'red' if val > 10 else 'black'
    return 'color: %s' % color


def color_blue(val):
    color = 'blue' if val in ['구매', '내역', '목록', '삭제', '부탁'] else 'black'
    return 'color: %s' % color


def color_red_if_duplicated(data, words):
    return ['color: red' if v in words else '' for v in data]


def highlight_max_red(data, color='red'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)


def magnify():
    return [dict(selector="th",
                 props=[("font-size", "4pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])]