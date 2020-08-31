import torch
import numpy as np


# change to a wrong label randomly
def change_label(label, total_label_nums=10):
    wrong_label = label
    while wrong_label == label:
        wrong_label = int(np.random.choice(total_label_nums, 1))

    return wrong_label


def create_random_errors(y_label, ratio):
    n = len(y_label)
    bad_n = int(ratio * n)

    y_label = y_label.clone()  # copy data

    perm = torch.randperm(y_label.size(0))

    # iterate through a permutation for bad_n times in total
    for i in range(bad_n):
        temp_i = perm[i]
        y_label[temp_i] = change_label(y_label[temp_i], 10)
    return y_label, perm


def create_systematic_errors(y_label, label_attack, label_after_attack, ratio):
    n = len(y_label)
    bad_n = int(ratio * n)
    y_label = y_label.clone()  # copy data

    perm = torch.randperm(y_label.size(0))

    # iterate through a permutation for bad_n times in total
    for i in range(bad_n):
        temp_i = perm[i]

        # if the attacked label is the exact label we want to modify
        if y_label[temp_i] == label_attack:
            y_label[temp_i] = label_after_attack
        else:
            y_label[temp_i] = change_label(y_label[temp_i], 10)

    return y_label, perm
