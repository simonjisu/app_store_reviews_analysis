# coding with utf-8
import urllib3
import json
import pandas as pd
import time
import sys

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

if button['doall']:
    pass
elif button['doone']:
    pass
elif button['doparts']:
    numbers = '[' + str(a) + '-' + str(b) + ']'
    filename = 'reviews' + numbers + '.txt' if b-a == 1 else 'review.txt'
else:
    print('error')

urllib3.disable_warnings()
data = pd.read_csv('App_Store_Links.csv', header=None)
with open('App_Store_Links.csv') as file:
    links = file.readlines()

app_id_list = []
for link in links:
    app_id_list.append(link.split('/id')[1].split('?mt')[0])

csvTitles = ['app_id', 'app_name', 'review_id', 'title', 'author', 'author_url', 'version', 'rating', 'review', 'vote_count']
col = len(csvTitles)
http = urllib3.PoolManager()

def getJson(url, http=http):
    response = http.request('GET', url, headers={'User-Agent': 'Mozilla/5.0'})
    time.sleep(1)
    datas = response.data.decode('utf-8')
    return json.loads(datas)

def getReviews(app_id_list, filename):
    with open('./data/' + filename, 'w', encoding='utf-8') as file:
        for i in range(len(csvTitles)):
            file.write(csvTitles[i] + '\t' if i < col else csvTitles[i])
        file.write('\n')
        for app_id in app_id_list:
            for page in range(1, 11):
                url = 'https://itunes.apple.com/kr/rss/customerreviews/id=%s/page=%d/sortby=mostrecent/json' % (app_id, page)
                data = getJson(url).get('feed')
                print(app_id)
                if data.get('entry') != None:

                    app_name = data.get('entry')[0]['im:name'].get('label')

                    for entry in data.get('entry'):
                        if entry.get('im:name'): continue

                        review_id = entry.get('id').get('label') if entry.get('id') else None
                        title = entry.get('title').get('label') if entry.get('title') else None
                        author = entry.get('author').get('name').get('label') if entry.get('author') else None
                        author_url = entry.get('author').get('uri').get('label') if entry.get('author') else None
                        version = entry.get('im:version').get('label') if entry.get('im:version')else None
                        rating = entry.get('im:rating').get('label') if entry.get('im:rating') else None
                        review = entry.get('content').get('label') if entry.get('content') else None
                        vote_count = entry.get('im:voteCount').get('label') if entry.get('im:voteCount') else None

                        csvData = [app_id, app_name, review_id, title.replace('\n', ' '), author, author_url, version,
                                   rating, review.replace('\n', ' '), vote_count]

                        for i in range(len(csvData)):
                            if csvData[i]:
                                file.write(csvData[i] + '\t' if i < col else csvData[i])
                            else:
                                file.write('NaN' + '\t' if i < col else 'NaN')
                        file.write('\n')

getReviews(app_id_list[a:b], filename=filename)