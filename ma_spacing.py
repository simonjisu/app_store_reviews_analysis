from utils import Spacing, read_jsonl, save_jsonl

def space_jamo(output_file_path, input_file_path):
    app_id_list, app_name_list, cate_list, rating_list, review_list = read_jsonl(input_file_path, key_ma=False)
    space_model = Spacing()
    new_review_list = space_model.fit(review_list, add_check_list=['!', '~', '.', ';', ':'])
    save_jsonl(output_file_path, app_id_list, app_name_list, cate_list, rating_list, new_review_list, key_ma=False)

space_jamo(output_file_path='./data/docs_jsonl_space_jamo.txt', input_file_path='./data/docs_jsonl.txt')

