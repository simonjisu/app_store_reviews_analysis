from analysis import *
from NaiveBayesian import get_conf_matrix, draw_conf
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
import numpy as np

def SVM_build_clf(train_coo_matrix, train_ratings):
    clf = SGDClassifier()
    y = np.asarray(train_ratings)
    clf.fit(train_coo_matrix, y.ravel())

    return clf


def SVM_main(load_file_path, vectorizer, ngram_range=(1, 2)):
    # './data/docs_jsonl_ma_NB.txt' 분류기 때문에 같은 것을 씀

    app_id_list, app_name_list, cate_list, rating_list, ma_list = read_jsonl(load_file_path)
    spaced_ma_list = ma_transform_by_spacing(ma_list)

    print('**making confusion matrix**')
    train_docs, test_docs, train_ratings, test_ratings = train_test_split(spaced_ma_list, rating_list)
    train_coo_matrix, test_coo_matrix, words, vec = get_matrix(vectorizer, train_docs, ngram_range=ngram_range,
                                                               mode_NB=True, test_spaced_ma_list=test_docs)

    clf = SVM_build_clf(train_coo_matrix, train_ratings)
    conf_mat, norm_conf_mat = get_conf_matrix(clf, test_coo_matrix, test_ratings)
    draw_conf(conf_mat, test_ratings)
    draw_conf(norm_conf_mat, test_ratings)


def SVM_build_pipeline():
    pipeline = Pipeline([
                ('vect', TfidfVectorizer(analyzer=str.split)),
                ('mb', SGDClassifier()),
            ])

    return pipeline


def SVM_CVtest(load_file_path, n_split=10):
    app_id_list, app_name_list, cate_list, rating_list, ma_list = read_jsonl(load_file_path)
    spaced_ma_list = ma_transform_by_spacing(ma_list)

    # GridSearch CV-test Tuning
    pipeline = SVM_build_pipeline()

    parameters = {
        "vect__max_features": (5000, None),
        "vect__ngram_range": ((1, 1), (1, 2)),
        "vect__use_idf": (True, False),
        "vect__smooth_idf": (True, False),
        "vect__sublinear_tf": (True, False),
        "vect__norm": ("l1", "l2", None),
        "mb__alpha": (1e-2, 1e-3),
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