from utils import Make_dictionay, Post_ma, Unknown_words


def After_processing(ma_docs, app_id_list):

    UNK = Unknown_words(ma_docs, app_id_list)
    unknown_list = UNK.get_unknown_words()
    make_dict = Make_dictionay()
