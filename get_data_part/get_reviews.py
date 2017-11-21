# -*- coding: utf-8 -*-
import urllib3
import json
import time
import sys
import numpy as np


csvTitles = ['app_id', 'app_name', 'review_id', 'title', 'author', 'author_url', 'version', 'rating', 'review', 'category']
col = len(csvTitles) - 1
http = urllib3.PoolManager()

def getJson(url, http=http):
    response = http.request('GET', url, headers={'User-Agent': 'Mozilla/5.0'})
    time.sleep(1 + np.random.normal(0.5, 0.1))
    datas = response.data.decode('utf-8')
    return json.loads(datas)

def getReviews(app_id_list, filename):
    progress = 0
    with open('./data/' + filename, 'w', encoding='utf-8') as file:
        for i in range(len(csvTitles)):
            file.write(csvTitles[i] + '\t' if i < col else csvTitles[i])
        file.write('\n')

        for app_id in app_id_list:
            progress += 1
            print(round((progress / len(app_id_list)), 4), app_id)
            page = 1
            doit = True
            while doit & (page <= 10):
            # for page in range(1, 11):
                url = 'https://itunes.apple.com/kr/rss/customerreviews/id=%s/page=%d/sortby=mostrecent/json' % (app_id, page)
                data = getJson(url).get('feed')

                if type(data.get('entry')) == list:
                    print(page)
                    try:
                        app_name = data.get('entry')[0]['im:name'].get('label')
                        cate = data.get('entry')[0]['category'].get('attributes').get('label')
                    except:
                        app_name = data.get('entry')['im:name'].get('label')
                        cate = data.get('entry')['category'].get('attributes').get('label')

                    for entry in data.get('entry')[1:]:
                        review_id = entry.get('id').get('label') if entry.get('id') else None
                        title = entry.get('title').get('label') if entry.get('title') else None
                        author = entry.get('author').get('name').get('label') if entry.get('author') else None
                        author_url = entry.get('author').get('uri').get('label') if entry.get('author') else None
                        version = entry.get('im:version').get('label') if entry.get('im:version')else None
                        rating = entry.get('im:rating').get('label') if entry.get('im:rating') else None
                        review = entry.get('content').get('label') if entry.get('content') else None

                        csvData = [app_id, app_name, review_id, title, author, author_url, version, rating,
                                   review, cate]
                        for i in range(len(csvData)):
                            if csvData[i]:
                                csvData[i] = csvData[i].replace('\t', '').replace('\n', ' ').replace('"', '')
                            else:
                                csvData[i] = 'NaN'

                        for i in range(len(csvData)):
                            if csvData[i]:
                                file.write(csvData[i] + '\t' if i < col else csvData[i])
                            else:
                                file.write('NaN' + '\t' if i < col else 'NaN')
                        file.write('\n')

                else:
                    doit = False
                page += 1

button = {'doall': False, 'doparts': False, 'doone': False}
if sys.argv[1] == 'all':
    button['doall'] = True
elif len(sys.argv) == 1:
    button['doone'] = True
elif len(sys.argv) > 2:
    a = int(sys.argv[1])
    b = int(sys.argv[2])
    if a <= b:
        button['doparts'] = True
    else:
        print('first cannot larger than second one, or they are not numbers')
else:
    print('insert argument: -all: do all the list -two numbers: do between numbers, start with 0 ends with 1 to 162000')


urllib3.disable_warnings()
# app link file: total len 4535
with open('new_app_id_list.txt', 'r') as file:
    app_ids = file.readlines()

app_id_list = []
for i in app_ids:
    app_id_list.append(i.strip())

if button['doall']:
    filename = 'reviews.txt'
    getReviews(app_id_list, filename=filename)
elif button['doone']:
    pass
elif button['doparts']:
    numbers = '[' + str(a+1) + '-' + str(b) + ']'
    filename = 'n_reviews' + numbers + '.txt' if b-a != 1 else 'review.txt'
    getReviews(app_id_list[a:b], filename=filename)
else:
    print('error')

# 0 ~ 4535