# -*- coding: utf-8 -*-

from itertools import chain, islice
from collections import deque, defaultdict, Counter
from IPython.display import clear_output
from tqdm import tqdm
import mmap
import time
import os
import sys
import ujson

class Post_ma(object):
    def __init__(self, file_path, auto_load=True):
        """
        사용법:
        1. update를 사용해 규칙사전을 등록, 2번째 부터는 자동으로 불러오기함
        2. 규칙을 로드하면 change하면됨
        """
        self.file_path = file_path
        self.post_ma_pairs = []
        if auto_load:
            if os.path.exists(self.file_path):
                self.load()
                print('Load Complete! total {} rules'.format(len(self.post_ma_pairs)))
            else:
                print('Can not load file: {}! '.format(self.file_path),
                      'There is no file, please create file by update method(take option write=True)')

    def __doc__(self):
        """
        post processing class rules,
        rules are saved at data file
        """
    # update part
    def update(self, ma_pairs, write=False):
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
        with open(self.file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            result = []
            for line in lines:
                result.append(self.combinate_before_after(line))
        self.post_ma_pairs = result

    def combinate_before_after(self, line):
        before, after = line.strip().split('>>')
        return tuple((self.tab_str_to_double_list(before),
                      self.tab_str_to_double_list(after)))

    def tab_str_to_double_list(self, string):
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

    # change part
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

    def replace_list_elem(self, target_list, target_idx, replacement):
        target_list.pop(target_idx)
        for i, repl in enumerate(replacement, target_idx):
            target_list.insert(i, repl)

    def split_method(self, ma_docs, ma_set, loc_info):
        """
        앞에서 loc_dict에 key는 문자열 하나기 때문에 loc_info에는 doc_loc와 ma_loc둘다 저장됨
        """
        for app_loc, doc_loc, ma_loc in loc_info:
            self.replace_list_elem(ma_docs[app_loc][doc_loc], ma_loc, ma_set[1])

    def merge_method(self, ma_docs, ma_set, loc_info):
        """
        앞에서 loc_dict에 key로 list를 쓸수 없기 때문에 loc_info에는 doc_loc 만 들어가게됨,
        """
        for app_id, doc_loc in loc_info:
            self.replace_sublist(ma_docs[app_id][doc_loc], ma_set[0], ma_set[1])

    def replace_ma_docs(self, ma_docs, location_dict):
        """
        location_list의 형태: 통일 할 것
        split: {'ma': [[app_id, doc_loc, ma_loc], ...]
        merge: {'ma1/ma2': [[app_id, doc_loc], ...]}
        """
        if not self.post_ma_pairs:
            print('Plead load post_ma_rules first')
            return None

        for ma_set in self.post_ma_pairs:
            if len(ma_set[0]) == 1:  # before -> after: split 일때
                ma_key = ma_set[0][0][0]
                self.split_method(ma_docs, ma_set, location_dict[ma_key])
            elif len(ma_set[0]) > 1:  # before -> after: merge 일때
                ma_key = '/'.join([word for word in ma_set[0]])
                self.merge_method(ma_docs, ma_set, location_dict[ma_key])
            else:
                print('error, no before word')

        return ma_docs

###############################################################

class Unknown_words(object):
    def __init__(self, ):
        self.unknown_loc_dict = None

    def get_unknown_words(self, ma_docs):
        unknown_list = []
        total_ma_count = 0
        for app_loc, docs in tqdm(enumerate(ma_docs), desc='Extracting unknowns:', total=len(ma_docs)):
            for doc_loc, doc in enumerate(docs):
                unknowns = [(ma[0], (app_loc, doc_loc, ma_loc)) for ma_loc, ma in enumerate(doc) if ma[1] in ['UNKNOWN', 'NA']]
                unknown_list.append(unknowns)
                total_ma_count += 1

        unknown_words = [ma for doc in unknown_list for ma, loc in doc]
        unique_list = list(set(unknown_words))

        self.unknown_loc_dict = {ma: loc for doc in unknown_list for ma, loc in doc}
        print('Unknown(중복제거): {}, Total Unknown: {}, Total MA: {}'.format(len(unique_list),
                                                                            len(unknown_words), total_ma_count))

        return unique_list

    # shape changer
    def shape_changer(self, dictionary, tokenizer_name='komoran'):
        """
        {'ㅜㅜ부탁이예요': [('부탁', 'NNG'), ('이', 'VCP'), ('예요', 'EC')],}
        이런 사전형을 리스트로 업데이트 할 수 있게 이쁘게 바꿔줌
        """
        unk_pos = 'UNKNOWN' if tokenizer_name == 'twitter' else 'NA'
        changed_ma_list = []
        for before, after in dictionary.items():
            changed_ma_list.append(tuple([[tuple([before, unk_pos])], after]))

        return changed_ma_list

###############################################################
# After data Processing
###############################################################

class Make_dictionay(object):
    def __init__(self, mecab_option=False):
        self.maxwords = None
        self.word_count = None
        self.choose_pos_list = None
        self.min_word_length = 1
        self.mecab_option = mecab_option
        self.normal_dict_option = None
        self.word_idx = None


    def get_word_count(self, ma_lists):
        """단어를 품사별로 묶어서 사전을 만든다 '단어, 품사' """

        # dictionary 종류 고르기
        self.set_normal_dict_option()
        for doc in tqdm(ma_lists, desc="Counting Words...", total=len(ma_lists)):  # 1 개의 app에 대해서
            for ma_doc in doc:
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

        if not self.mecab_option:  # komoran 품사단어 뒤에 +다 붙여주기
            ma_doc = [(lex + '다', pos) if pos in ['VV', 'VA'] else (lex, pos) for (lex, pos) in ma_doc]
        if self.min_word_length >= 2:
            ma_doc = [(lex, pos) for (lex, pos) in ma_doc if len(lex) >= 2]
        ma_doc = [('NULL', 'NULL') if pos in ['UNKNOWN', 'NA'] else (lex, pos) for (lex, pos) in ma_doc]

        return ma_doc

    def is_pos_in(self, pos):
        if pos in self.choose_pos_list:
            return True
        else:
            return False

    def get_word_idx(self):
        self.word_idx = {w: i for i, w in enumerate(self.word_count.keys())}

        return self

    def fit(self, ma_lists, choose_pos=None, min_word_length=1, maxwords=100000, normal_dict_option=0):
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

        self.get_word_count(ma_lists)
        self.get_word_idx()

    # def update(self, new_dict):

    ####### only for dictionary option 1
    def filter_by_dict(self, doc):
        if self.normal_dict_option == 0:
            doc = [ma for ma in doc if ma in self.word_count.keys()]
        elif self.normal_dict_option == 1:
            doc = ['/'.join(ma) for ma in doc if '/'.join(ma) in self.word_count.keys()]

        return doc

    def document_transform(self, ma_lists, ratings_lists=None):
        """사전에 존재하는 것들만 문서로 바꿔준다 app_id별로 [[word1, word2, ...], [...] 순서는 상관 없음"""
        documents = []
        ratings = []
        for i, docs in tqdm(enumerate(ma_lists), desc="Filtering...", total=len(ma_lists)):
            filtered_docs = []
            filtered_ratings = []
            for j, doc in enumerate(docs):
                new_doc = self.filter_by_dict(doc)
                if len(new_doc) > 0:
                    filtered_docs.append(new_doc)
                    if ratings_lists:
                        filtered_ratings.append(ratings_lists[i][j])

            documents.append(filtered_docs)
            ratings.append(filtered_ratings)

        if ratings:
            return documents, ratings
        else:
            return documents

    def flatten_all_docs(self, all_docs, by_app_id=False):
        flattened_docs = []
        for docs in tqdm(all_docs, desc="Processing...", total=len(all_docs)):
            if by_app_id:
                flattened_docs.append([val for doc in docs for val in doc])
            else:
                for doc in docs:
                    flattened_docs.append(doc)

        return flattened_docs


    def save_as_file(self, filepath, docs):
        with open(filepath, 'w', encoding='utf-8') as outputfile:
            for doc in tqdm(docs, desc='saving', total=len(docs)):
                print(' '.join(doc), file=outputfile)

    def load_file(self, file_path, option_split=True):
        with open(file_path, 'r', encoding='utf-8') as file:
            docs = []
            for line in file:
                if option_split:
                    doc = line.split(' ').strip()
                else:
                    doc = line.strip()
                docs.append(doc)

        return docs
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

    def fit(self, docs, add_check_list=None):
        if add_check_list:
            self.check_list = self.basic_check_list + add_check_list
        else:
            self.check_list = self.basic_check_list
        self.app_id_list = [k for k in docs.keys()]
        for app_id in tqdm(self.app_id_list, desc='Processing spacing', total=len(self.app_id_list)):
            for i, doc in enumerate(docs[app_id]['reviews']):
                docs[app_id]['reviews'][i] = self.space_jamo(doc)
        return docs

    def space_jamo(self, doc):
        if not self.check_list:
            self.check_list = self.basic_check_list

        doc_splited = [char for char in doc]
        space_idx = self.get_spacing_index(doc_splited)
        tokens_list = self.split_doc_to_words(doc, space_idx)
        new_text = ' '.join(tokens_list)

        return new_text

    def split_doc_to_words(self, doc, space_idx):
        tokens_list = []
        for i in range(2, len(space_idx) + 1):
            start, stop = deque(islice(space_idx, i), maxlen=2)
            tokens_list.append(doc[start:stop])

        return tokens_list

    def get_spacing_index(self, doc_splited):
        space_idx = []
        doc_len = len(doc_splited)
        for i in range(1, doc_len+1):
            window = deque(islice(doc_splited, i), maxlen=2)  # 번째 원소 전까지 window가 움직이는거
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

####### utils
def notebook_print(text, sleep_time=0.01):
    clear_output(wait=True)
    print(text)
    time.sleep(sleep_time)


def commandline_print(text, sleep_time=0.01):
    sys.stdout.write('processing: {} \r'.format(text))
    sys.stdout.flush()
    time.sleep(sleep_time)


def get_data_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        texts = file.read()
        data = ujson.loads(texts)
        app_ids_list = list(data.keys())
    return data, app_ids_list


def save_json(filepath, docs):
    with open(filepath, 'w', encoding='utf-8') as file:
        texts = ujson.dumps(docs, ensure_ascii=False)
        print(texts, file=file)


def read_jsonl(filepath):
    key_app_id = 'app_id'
    key_ratings = 'ratings'
    key_ma = 'ma'
    app_id_list = []
    ratings_lists = []
    ma_lists = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc="Reading documents", total=get_num_lines(filepath)):
            doc = ujson.loads(line)
            app_id_list.append(doc[key_app_id])
            ratings_lists.append(doc[key_ratings])
            ma_lists.append(doc[key_ma])

    return app_id_list, ratings_lists, ma_lists

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


###############################################################
# word visualization
###############################################################