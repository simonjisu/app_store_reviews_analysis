from analysis import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def NB_data_processing(file_path, save_file_path, major_pos):
    make_dict, app_id_list, app_name_list, cate_list, rating_list, ma_list = \
        make_keyword_dict(file_path, major_pos, print_keywords=False, komoran_option=True, only_ma=False)

    filter_docs_by_dict(make_dict, app_id_list, app_name_list, cate_list, rating_list, ma_list,
                        save_file_path=save_file_path, by_app_id=False)


def NB_build_clf(train_coo_matrix, train_ratings):
    clf = MultinomialNB()
    y = np.asarray(train_ratings)
    clf.fit(train_coo_matrix, y.ravel())

    return clf


def get_conf_matrix(clf, test_coo_matrix, test_ratings):
    pred_ratings = clf.predict(test_coo_matrix)
    conf_mat = confusion_matrix(np.asarray(test_ratings), pred_ratings)
    norm_conf_mat = conf_mat.astype(np.float) / conf_mat.sum(axis=1)[:, np.newaxis]
    print('=' * 50)
    print('Accurary: {0:.4f}%'.format(accuracy_score(np.asarray(test_ratings), pred_ratings)*100))
    print('=' * 50)
    print('Classification Report')
    print('=' * 50)
    print(classification_report(test_ratings, pred_ratings))
    print('='*50)
    return conf_mat, norm_conf_mat


def draw_conf(conf_mat, test_ratings):
    labels = sorted(set(test_ratings))
    np.set_printoptions(precision=2)
    tick_marks = np.arange(len(labels))

    fig = plt.figure()
    plt.imshow(conf_mat, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", fontsize=20)
    plt.colorbar()
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.ylabel("Answer")
    plt.xlabel("Predict")
    plt.show()


def NB_main(load_file_path, vectorizer, ngram_range=(1, 2)):
    app_id_list, app_name_list, cate_list, rating_list, ma_list = read_jsonl(load_file_path)
    # word_idx = load_json('./data/word_idx_json_NB.txt')
    spaced_ma_list = ma_transform_by_spacing(ma_list)
    # 기존 분포
    ax = plt.axes()
    ratings = np.asarray(rating_list, dtype=np.int)
    sns.countplot(ratings, ax = ax)
    ax.set_title('기존 Ratings 문서 분포', fontsize=20)
    plt.show()

    # confusion matrix
    print('**making confusion matrix**')
    train_docs, test_docs, train_ratings, test_ratings = train_test_split(spaced_ma_list, rating_list)
    train_coo_matrix, test_coo_matrix, words, vec = get_matrix(vectorizer, train_docs, ngram_range=ngram_range,
                                                               mode_NB=True, test_spaced_ma_list=test_docs)

    clf = NB_build_clf(train_coo_matrix, train_ratings)
    conf_mat, norm_conf_mat = get_conf_matrix(clf, test_coo_matrix, test_ratings)
    draw_conf(conf_mat, test_ratings)
    draw_conf(norm_conf_mat, test_ratings)


def NB_build_pipeline():
    pipeline = Pipeline([
                ('vect', TfidfVectorizer(analyzer=str.split)),
                ('mb', MultinomialNB()),
            ])

    return pipeline


def NB_CVtest(load_file_path, n_split=10):
    app_id_list, app_name_list, cate_list, rating_list, ma_list = read_jsonl(load_file_path)
    spaced_ma_list = ma_transform_by_spacing(ma_list)

    # GridSearch CV-test Tuning
    pipeline = NB_build_pipeline()

    parameters = {
        "vect__max_features": (5000, None),
        "vect__ngram_range": ((1, 1), (1, 2)),
        "vect__use_idf": (True, False),
        "vect__smooth_idf": (True, False),
        "vect__sublinear_tf": (True, False),
        "vect__norm": ("l1", "l2", None),
        "mb__alpha": (1.0, 2.0),
    }
    cv = KFold(n_splits=n_split, shuffle=True)
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring="accuracy", cv=cv)
    X = spaced_ma_list
    y = rating_list
    grid_search.fit(X, y)
    best_parameters = grid_search.best_estimator_.get_params()


    print("Best score: {}".format(grid_search.best_score_))

    print("Best parameter set:")

    for param_name in parameters:
        print("\t{}: {}".format(param_name, best_parameters[param_name]))

