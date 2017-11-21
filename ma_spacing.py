from utils import *

def space_jamo(filepath, type_filepath):
    docs, app_id_list = get_data_json(type_filepath)
    space_model = Spacing()
    new_docs = space_model.fit(docs, add_check_list=['!', '~', '.', ';', ':'])
    save_json(filepath, new_docs)

space_jamo(filepath='./data/train_json_space_jamo.txt', type_filepath='./data/train_json.txt')
space_jamo(filepath='./data/test_json_space_jamo.txt', type_filepath='./data/test_json.txt')

