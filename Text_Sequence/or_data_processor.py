import re
import os
import pickle
import shutil
import random

def open_data_txt(filename):
    with open(filename, 'r') as file:
        content = file.read()
        segments = content.split('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')

        segments = list(filter(lambda x: len(x.strip()) >= 1, segments))

        locations = []
        content_codes = []
        labels = []
        for segment in segments:
            line = segment.strip().split("\n")
            line = list(filter(None, line))

            if not line:
                continue

            path_line = line[0].strip()
            words = path_line.split()
            if len(words) < 2:
                continue
            locations.append(words[1])

            label_str = line[-1].strip()
            if label_str not in {'0', '1'}:
                continue
            labels.append(int(label_str))

            code_lines = line[1:-1]
            content_code = ' '.join(code_lines)
            content_codes.append(content_code)

        return content_codes, labels, locations

def write_slices_pkl(save_tokens_path, save_labels_path, save_location_path, all_slices_of_token, label, location):
    with open(save_tokens_path, 'wb') as file:
        pickle.dump(all_slices_of_token, file)
    with open(save_labels_path, 'wb') as file:
        pickle.dump(label, file)
    with open(save_location_path, 'wb') as file:
        pickle.dump(location, file)

def write_slices_txt(save_tokens_path, save_labels_path, save_location_path, all_slices_of_token, label, location):
    for all_slices_of_toke in all_slices_of_token:
        with open(save_tokens_path, 'a') as txt_file:
            txt_file.write(str(all_slices_of_toke)+'\n')
    for l in label:
        with open(save_labels_path, 'a') as txt_file:
            txt_file.write(str(l)+'\n')
    for lo in location:
        with open(save_location_path, 'a') as txt_file:
            txt_file.write(str(lo)+'\n')

def main():
    train_data_path = 'merge_data/ast_data.txt'
    save_tokens_path = "ast_train/ast_data_tokens.pkl"
    save_labels_path = "ast_train/ast_data_labels.pkl"
    save_location_path = "ast_train/ast_data_location.pkl"
    save_tokens_path_txt = "ast_train/ast_data_tokens.txt"
    save_labels_path_txt = "ast_train/ast_data_labels.txt"
    save_location_path_txt = "ast_train/ast_data_location.txt"

    if os.path.exists("ast_train"):
        shutil.rmtree("ast_train")

    all_data_token, labels, locations = open_data_txt(train_data_path)

    write_slices_pkl(save_tokens_path, save_labels_path, save_location_path, all_data_token, labels,
                     locations)
    write_slices_txt(save_tokens_path_txt, save_labels_path_txt, save_location_path_txt, all_data_token,
                     labels, locations)

if __name__ == "__main__":
    main()
