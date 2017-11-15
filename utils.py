# -*- coding: utf-8 -*-

from itertools import chain, islice
from collections import deque, defaultdict, Counter
import os

class Post_ma(object):
    def __init__(self, auto_load=True, tokenizer_name='komoran'):
        """
        사용법:
        1. update를 사용해 규칙사전을 등록, 2번째 부터는 자동으로 불러오기함
        2. 규칙을 로드하면 change하면됨
        """
        self.filename = './data/post_ma_pairs_' + tokenizer_name + '.txt'
        self.post_ma_pairs = []
        if auto_load:
            if os.path.exists(self.filename):
                self.load()
                print('Load Complete! total {} rules'.format(len(self.post_ma_pairs)))
            else:
                print('Can not load file: {}! '.format(self.filename),
                      'There is no file, please create file by update method(take option write=True)')

    def __doc__(self):
        """
        post processing class rules,
        rules are saved at data file
        """
    # update part
    def update(self, ma_pairs, write=False):
        button = 'w' if write else 'a'
        with open(self.filename, button, encoding='utf-8') as output_file:
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
        with open(self.filename, 'r', encoding='utf-8') as file:
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
        for app_id, (doc_loc, ma_loc) in loc_info:
            self.replace_list_elem(ma_docs[app_id][doc_loc], ma_loc, ma_set[1])

    def merge_method(self, ma_docs, ma_set, loc_info):
        """
        앞에서 loc_dict에 key로 list를 쓸수 없기 때문에 loc_info에는 doc_loc 만 들어가게됨,
        """
        for app_id, doc_loc in loc_info:
            self.replace_sublist(ma_docs[app_id][doc_loc], ma_set[0], ma_set[1])

    def replace_ma_docs(self, ma_docs, location_dict):
        """
        location_dict의 형태: 통일 할 것
        split: {'ma': [[app_id, (doc_loc, ma_loc)], ...]
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
    def __init__(self, ma_docs, app_id_list):
        self.ma_docs = ma_docs
        self.app_id_list = app_id_list
        self.unknown_loc = defaultdict(list)  # inversed index {ma: [(app_id, loc), (app_id, loc)...]}

    def get_unknown_words(self):
        unknown_list = []
        total_ma_count = 0
        for app_id in self.app_id_list:
            for doc_loc, doc in enumerate(self.ma_docs[app_id]):
                for ma_loc, ma in enumerate(doc):
                    total_ma_count += 1
                    if ma[1] in ['UNKNOWN', 'NA']:
                        unknown_list.append(ma[0])
                        self.unknown_loc[ma[0]].append([app_id, tuple([doc_loc, ma_loc])])
        print('Unknown(중복제거): {}, Total Unknown: {}, Total MA: {}'.format(len(set(unknown_list)), len(unknown_list), total_ma_count))
        return list(set(unknown_list))

    def remove_no_meanings(self, unknown_list, tokenizer):
        """UNKOWN 단어들의 앞뒤에 ㅡ,ㅉ,ㅠ,ㅜ,ㅋ 등등 를 떼어내고 형태소 분석 결과 내보냄"""
        rest_unknown_list = []
        unknown_rule = defaultdict()
        for unknown_word in unknown_list:
            new_word, rest = self.check_word(unknown_word)
            tokens = tokenizer.pos(new_word)
            if len(tokens) > 1:
                unknown_rule[unknown_word] = tokens
            elif (len(tokens) == 1) and (tokens[0][1] not in ['UNKNOWN', 'NA']):
                unknown_rule[unknown_word] = tokens
            else:
                rest_unknown_list.append(unknown_word)

        return unknown_rule, rest_unknown_list

    def check_word(self, word):
        new_word = []
        rest = []
        for char in word:
            if self.check_jamo(char):
                rest.append(char)
            else:
                new_word.append(char)
        new_word = ''.join(new_word)

        return new_word, rest

    def check_jamo(self, char):
        CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ',
                         'ㅢ', 'ㅣ']
        JONGSUNG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ',
                         'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        if (char in CHOSUNG_LIST) | (char in JUNGSUNG_LIST) | (char in JONGSUNG_LIST):
            return True
        else:
            return False

    # shape changer
    def shape_changer(self, dictionary, tokenizer_name='komoran'):
        """
        {'ㅜㅜ부탁이예요': [('부탁', 'NNG'), ('이', 'VCP'), ('예요', 'EC')],}
        이런 사전형을 리스트로 업데이트 할 수 있게 이쁘게 바꿔줌
        """
        unk_pos = 'UNKNOWN' if tokenizer_name == 'twitter' else 'NA'
        changed_ma_list = []
        for items in dictionary.items():
            changed_ma_list.append(tuple([[tuple([items[0], unk_pos])], items[1]]))

        return changed_ma_list

###############################################################
# After data Processing
###############################################################

class Make_dictionay(object):
    def __init__(self):
        self.maxwords = None
        self.word_count = Counter()
        self.word_dict_idx = defaultdict(int)
        self.word_dict_idx['NULL'] = 0
        self.word_dict_idx_inverse = None
        self.delete_pos_list = None
        self.min_word_length = 1


    def get_word_count(self, ma_docs):
        """단어를 품사별로 묶어서 사전을 만든다 '단어/품사' """
        for app_id in self.app_id_list:
            for doc in ma_docs[app_id]:
                if self.delete_pos_list:
                    doc = [word for word in doc if word[1] not in self.delete_pos_list]
                if self.min_word_length >= 2:
                    doc = [word for word in doc if len(word[0]) >= 2]
                words = ['NULL' if word[1] in ['UNKNOWN', 'NA'] else '/'.join(word) for word in doc]  # NA 단어는 NULL 처리
                self.word_count.update(words)
        if self.maxwords:
            self.word_count = Counter({vals[0]: vals[1] for vals in self.word_count.most_common(self.maxwords)})
            print('total words: {}'.format(len(self.word_count)))

    def get_word_dict_idx(self, inverse=False):
        """단어 마다 인덱싱을 해준다"""
        for i, word in enumerate(list(self.word_count.keys()), 1):
            self.word_dict_idx[word] = i
        if inverse:
            self.word_dict_idx_inverse = {v: k for k, v in self.word_dict_idx.items()}

    def fit(self, ma_docs, delete_pos=None, min_word_length=1, maxwords=100000, inverse=True):
        """
        delete_pos: 지우고 싶은 품사는 지울 수 있음 list형태
        maxwords: 설정하고자 하는 최대 단어 수, most_common 순으로 추출된다, 안하려면 None 으로 설정할 것
        inverse: 사전idx에 대한 역인덱싱 사전도 구함
        """
        self.app_id_list = list(ma_docs.keys())
        self.maxwords = maxwords
        if delete_pos:
            self.delete_pos_list = delete_pos
        if min_word_length >= 2:
            self.min_word_length = min_word_length

        self.get_word_count(ma_docs)
        self.get_word_dict_idx(inverse=inverse)
        print('complete!')

# 밑에는 잠시 보류
    def replace_ma_to_idx(self, doc):
        """형태소 list 를 단어 idx로 바꾼다"""
        for i, ma in enumerate(doc):
            ma_str = '/'.join(ma)
            doc.pop(i)
            doc.insert(i, self.word_dict_idx[ma_str])

        return doc

    def get_ma_docs_idx(self, ma_docs):
        """단어 위치별로 index를 만든다 {app_id: [[word_id1, word_id2, ...], [...]}"""
        ma_docs_idx = defaultdict(list)
        for app_id in self.app_id_list:
            for doc in ma_docs[app_id]:
                doc_idx = self.replace_ma_to_idx(doc[:])  # 리스트를 복사해서 넣을 것 안그러면 없어짐 기존 ma_docs가 대체됨
                ma_docs_idx[app_id].append(doc_idx)

        return ma_docs_idx

    def count_vectorize(self):
        pass