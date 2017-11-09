# -*- coding: utf-8 -*-

import sqlalchemy
from data import settings
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String
import pandas as pd
import sys
from collections import defaultdict
from konlpy.tag import Twitter, Kkma, Mecab
import ujson

engine = sqlalchemy.create_engine(settings.DB_TYPE + settings.DB_USER + ":" + settings.DB_PASSWORD + "@" + settings.DB_URL + ":" + settings.DB_PORT + "/" + settings.DB_NAME, echo=settings.QUERY_ECHO)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

class Reviews(Base):
    # table name:
    __tablename__ = 'reviews'
    # table column:
    idx = Column(Integer, primary_key=True)
    app_id = Column(String)
    app_name = Column(String)
    review_id = Column(String)
    title = Column(String)
    author = Column(String)
    author_url = Column(String)
    version = Column(String)
    rating = Column(String)
    review = Column(String)
    category = Column(String)

Base.metadata.create_all(engine)

cols = ['app_id', 'app_name', 'title', 'review', 'rating', 'category', 'author_url', 'author']
data_dict = {i:[] for i in cols}
for row in session.query(Reviews.app_id, Reviews.app_name, Reviews.title, Reviews.review, Reviews.rating,
                         Reviews.category, Reviews.author_url, Reviews.author):
    for i, col_name in enumerate(cols):
        data_dict[col_name].append(row[i])

data = pd.DataFrame(data_dict)
del data_dict


def tokenize_doc(doc, tokenizer):
    return ['/'.join(t) for t in tokenizer.pos(doc)]

def dumps_json_docs(data, tokenizer, filename):
    json_dict = defaultdict(dict)

    for i, doc in enumerate(data.values):
        json_dict[i]['app_id'] = doc[0]
        if len(doc[1]) <= 1000:  # 한글자를 5000자까지 쓴 데이터가 있어서 메모리 부족으로 진행이 안됨
            json_dict[i]['tokens'] = tokenize_doc(doc[1], tokenizer=tokenizer)  
        else:
            json_dict[i]['tokens'] = tokenize_doc(doc[1][:1000], tokenizer=tokenizer)
        json_dict[i]['rating'] = doc[2]

        if i % 1000 == 0: print(i)

    with open(filename, 'w', encoding='utf-8') as f:
        data_json = ujson.dumps(json_dict, ensure_ascii=False)
        print(data_json, file=f)

twitter = Twitter()
kkma = Kkma()
mecab = Mecab()

docs = data.loc[:, ['app_id', 'review', 'rating']]
id_list = docs.loc[:, 'app_id'].unique()
train_idx = int(len(id_list) * 0.8)
train_app_id = id_list[train_idx]
train_data_idx = docs.app_id[docs.app_id.isin([train_app_id])].index[-1]

# Booting
dic_for_booting = defaultdict(list)
dic_for_booting['data'].append(sys.argv[1])
dic_for_booting['tokenizer'].append(sys.argv[2])
if sys.argv[2] == 'twitter':
    dic_for_booting['tokenizer'].append(twitter)
elif sys.argv[2] == 'kkma':
    dic_for_booting['tokenizer'].append(kkma)
elif sys.argv[2] == 'mecab':
    dic_for_booting['tokenizer'].append(mecab)
else:
    print('write 2 sys.argv: data[train or test] and tokenizer[twitter, kkma, mecab]')

print(dic_for_booting)
if dic_for_booting['data'][0] == 'train':
    train_data = docs.iloc[:train_data_idx, :]
    print(len(train_data))
    dumps_json_docs(data=train_data, tokenizer=dic_for_booting['tokenizer'][1],
                    filename='./data/train_data_{}.txt'.format(dic_for_booting['tokenizer'][0]))

elif dic_for_booting['data'][0] == 'test':
    test_data = docs.iloc[train_data_idx:, :]
    print(len(test_data))
    dumps_json_docs(data=test_data, tokenizer=dic_for_booting['tokenizer'][1],
                    filename='./data/test_data_{}.txt'.format(dic_for_booting['tokenizer'][0]))

