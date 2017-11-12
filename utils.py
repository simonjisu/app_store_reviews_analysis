# -*- coding: utf-8 -*-

from itertools import chain, islice
from collections import deque


class Post_ma(object):
    def __init__(self, auto_load=True, tokenizer_name='twitter'):
        self.filename = './data/post_ma_pairs_' + tokenizer_name + '.txt'
        self.post_ma_pairs = None
        if auto_load:
            self.load()

    def __doc__(self):
        """
        post processing class rules,
        rules are saved at data file
        """

    def update(self, ma_pairs, rewrite=False):
        button = 'a' if not rewrite else 'w'
        with open(self.filename, button, encoding='utf-8') as output_file:
            duplicated_index = []
            for ma in ma_pairs:
                if not self.check_duplicate(ma):
                    text = self.split_before_after(ma)
                    print(text, file=output_file)
                else:
                    duplicated_index.append(ma_pairs.index(ma))

            if len(duplicated_index) != 0:
                print('There are duplicated rules! find you input pairs in index of {}'.format(duplicated_index))

    def delete(self, ma_pairs):
        pass

    def check_duplicate(self, ma):
        if ma in self.post_ma_pairs:
            return True
        else:
            return False

    def split_before_after(self, ma):
        before, after = ma
        before = list(chain.from_iterable(before))
        after = list(chain.from_iterable(after))
        text = '\t'.join(before) + '>>' + '\t'.join(after)
        return text

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