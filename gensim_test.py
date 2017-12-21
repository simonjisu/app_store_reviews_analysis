import gensim
from gensim.models import doc2vec
from utils import read_jsonl, save_jsonl
from gensim.models.doc2vec import TaggedDocument
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

load_file_path = './data/docs_jsonl_ma_komoran_after_processing.txt'
app_id_list, app_name_list, cate_list, rating_list, ma_list = read_jsonl(load_file_path)

docs = []
for doc in ma_list:
    temp = [ma[0] for ma in doc if ma[1] not in ['SF','SE','SS','SP','SO','SW'] ]
    docs.append(temp)


tagged_docs = [TaggedDocument(d, [r, app_id, cate]) for d, r, app_id, cate in \
               zip(docs, rating_list, app_id_list, cate_list)]

test_docs = tagged_docs[:100000]

model_path = './data/gensim_model/3dmodel'
model = doc2vec.Doc2Vec.load(model_path)
doc_vects = model.docvecs

#
palette = sns.color_palette("RdBu_r", 5)
palette_dict = {str(i):p for i, p in zip(range(1,6), palette)}
fig = plt.figure(num=1, figsize=(8, 8))
ax = fig.gca(projection='3d')
for vec, (_, tags) in zip(doc_vects, test_docs):
    ax.scatter(vec[0], vec[1], vec[2], c=palette_dict[tags[0]])


unique_cate_list = list(set(cate_list))
cate_len = len(unique_cate_list)
palette = sns.color_palette("husl", cate_len)
palette_dict = {str(i):p for i, p in zip(unique_cate_list, palette)}

fig2 = plt.figure(num=2, figsize=(8, 8))
ax = fig2.gca(projection='3d')
for vec, (_, tags) in zip(doc_vects, test_docs):
    ax.scatter(vec[0], vec[1], vec[2], c=palette_dict[tags[2]])
