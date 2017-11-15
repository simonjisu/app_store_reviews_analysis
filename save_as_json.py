# -*- coding: utf-8 -*-

import pandas as pd
import ujson
from collections import defaultdict

train_data = pd.read_csv('./data/train_docs.txt', sep='\t')
test_data = pd.read_csv('./data/test_docs.txt', sep='\t')


def save_as_json(data, filename):
    app_id_list = data.iloc[:, 0].unique()
    with open(filename, 'w', encoding='utf-8') as output_file:
        json_dict = defaultdict(dict)
        for app_id in app_id_list:
            reviews_list = data.loc[data.iloc[:, 0] == app_id, 'review'].values.tolist()
            ratings_list = data.loc[data.iloc[:, 0] == app_id, 'rating'].values.tolist()
            json_dict[app_id]['reviews'] = reviews_list
            json_dict[app_id]['ratings'] = ratings_list
            print(app_id)
        txts = ujson.dumps(json_dict, ensure_ascii=False)
        print(txts, file=output_file)
    print('done!')

save_as_json(train_data, './data/train_json.txt')
save_as_json(test_data, './data/test_json.txt')

