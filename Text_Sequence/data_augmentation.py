import torch
import numpy as np
import pickle
import random
from collections import defaultdict

random.seed(8)
np.random.seed(8)
torch.manual_seed(8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(source_tokens, source_labels, source_location, target_tokens, target_labels, target_location, target_ratio=0.25):
    with open(source_tokens, 'rb') as f1, open(source_labels, 'rb') as f2, open(source_location, 'rb') as f5:
        src_codes = pickle.load(f1)
        src_labels = pickle.load(f2)
        src_location = pickle.load(f5)

    src_codes, src_labels, src_location = balance_samples(
        src_codes, src_labels, src_location, target_ratio, "source"
    )

    with open(target_tokens, 'rb') as f3, open(target_labels, 'rb') as f4, open(target_location, 'rb') as f6:
        tgt_codes = pickle.load(f3)
        tgt_labels = pickle.load(f4)
        tgt_location = pickle.load(f6)

    tgt_codes, tgt_labels, tgt_location = balance_samples(
        tgt_codes, tgt_labels, tgt_location, target_ratio, "target"
    )

    return src_codes, src_labels, src_location, tgt_codes, tgt_labels, tgt_location

def balance_samples(codes, labels, locations, target_ratio, domain_type):
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    n_positive = len(label_to_indices[1])
    n_negative = len(label_to_indices[0])
    total_samples = n_positive + n_negative
    current_ratio = n_positive / total_samples

    target_positive = int(total_samples * target_ratio)
    target_negative = total_samples - target_positive

    if domain_type == "source":
        if n_positive < target_positive:
            additional_positive = target_positive - n_positive
            additional_indices = random.choices(label_to_indices[1], k=additional_positive)
            selected_positive_indices = label_to_indices[1] + additional_indices
        else:
            selected_positive_indices = random.sample(label_to_indices[1], target_positive)

        selected_negative_indices = random.sample(label_to_indices[0], target_negative)

    elif domain_type == "target":
        selected_positive_indices = random.sample(label_to_indices[1], target_positive)

        if n_negative < target_negative:
            additional_negative = target_negative - n_negative
            additional_indices = random.choices(label_to_indices[0], k=additional_negative)
            selected_negative_indices = label_to_indices[0] + additional_indices
        else:
            selected_negative_indices = random.sample(label_to_indices[0], target_negative)

    selected_indices = selected_positive_indices + selected_negative_indices

    adjusted_codes = [codes[i] for i in selected_indices]
    adjusted_labels = [labels[i] for i in selected_indices]
    adjusted_locations = [locations[i] for i in selected_indices]

    combined = list(zip(adjusted_codes, adjusted_labels, adjusted_locations))
    random.shuffle(combined)
    adjusted_codes, adjusted_labels, adjusted_locations = zip(*combined)

    n_final_positive = sum(adjusted_labels)
    n_final_negative = len(adjusted_labels) - n_final_positive
    final_ratio = n_final_positive / len(adjusted_labels)

    return list(adjusted_codes), list(adjusted_labels), list(adjusted_locations)

def write_slices_pkl(save_tokens_path, save_labels_path, save_location_path, all_slices_of_token, label, location):
    with open(save_tokens_path, 'wb') as file:
        pickle.dump(all_slices_of_token, file)
    with open(save_labels_path, 'wb') as file:
        pickle.dump(label, file)
    with open(save_location_path, 'wb') as file:
        pickle.dump(location, file)

def write_slices_txt(save_tokens_path, save_labels_path, save_location_path, all_slices_of_token, label, location):
    for l in label:
        with open(save_labels_path, 'a') as txt_file:
            txt_file.write(str(l)+'\n')
    for lo in location:
        with open(save_location_path, 'a') as txt_file:
            txt_file.write(str(lo)+'\n')

def main():
    source_tokens = "ast_train/ast_data_tokens.pkl"
    source_labels = "ast_train/ast_data_labels.pkl"
    source_location = "ast_train/ast_data_location.pkl"
    target_tokens = "java_ast_train/ast_data_tokens.pkl"
    target_labels = "java_ast_train/ast_data_labels.pkl"
    target_location = "java_ast_train/ast_data_location.pkl"

    ssave_tokens_path = "dann_data_aug/s2/ast_data_tokens.pkl"
    ssave_labels_path = "dann_data_aug/s2/ast_data_labels.pkl"
    ssave_location_path = "dann_data_aug/s2/ast_data_location.pkl"
    ssave_tokens_path_txt = "dann_data_aug/s2/ast_data_tokens.txt"
    ssave_labels_path_txt = "dann_data_aug/s2/ast_data_labels.txt"
    ssave_location_path_txt = "dann_data_aug/s2/ast_data_location.txt"

    tsave_tokens_path = "dann_data_aug/t2/ast_data_tokens.pkl"
    tsave_labels_path = "dann_data_aug/t2/ast_data_labels.pkl"
    tsave_location_path = "dann_data_aug/t2/ast_data_location.pkl"
    tsave_tokens_path_txt = "dann_data_aug/t2/ast_data_tokens.txt"
    tsave_labels_path_txt = "dann_data_aug/t2/ast_data_labels.txt"
    tsave_location_path_txt = "dann_data_aug/t2/ast_data_location.txt"

    src_codes, src_labels, src_location, tgt_codes, tgt_labels, tgt_location = load_data(
        source_tokens,
        source_labels,
        source_location,
        target_tokens,
        target_labels,
        target_location,
        target_ratio=0.25
    )

    write_slices_pkl(ssave_tokens_path, ssave_labels_path, ssave_location_path, src_codes, src_labels, src_location)
    write_slices_txt(ssave_tokens_path_txt, ssave_labels_path_txt, ssave_location_path_txt, src_codes, src_labels, src_location)

    write_slices_pkl(tsave_tokens_path, tsave_labels_path, tsave_location_path, tgt_codes, tgt_labels, tgt_location)
    write_slices_txt(tsave_tokens_path_txt, tsave_labels_path_txt, tsave_location_path_txt, tgt_codes, tgt_labels, tgt_location)

if __name__ == "__main__":
    main()
