from collections import Counter
from torchvision import models
import torch
import torch.backends.cudnn as cudnn
import joblib
import torch.nn as nn
import numpy as np
import pickle
import math
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score
import os
from torch.autograd import Variable
import argparse
import torch.nn.functional as F
import random

from torchvision.models import ResNet
from sklearn.cluster import KMeans
from torchvision.models import ResNet

def read_data_IE(cross, index):
    S, T, Label = joblib.load('./features/IEMOCAP_ST.pkl')
    if cross == '5':
        print(
            "=======================================Sess{} test====================================".format(index + 1))
        test_data_s = np.vstack((np.array(S[2 * index]), np.array(S[2 * index + 1])))
        test_data_t = np.vstack((np.array(T[2 * index]), np.array(T[2 * index + 1])))
        # test_data_f = np.vstack((np.array(F[2 * index]), np.array(F[2 * index + 1])))
        Test_label = np.hstack((np.array(Label[2 * index]), np.array(Label[2 * index + 1])))

        del Label[2 * index]
        del Label[2 * index]

        S = np.delete(S, 2 * index, axis=0)
        S = np.delete(S, 2 * index, axis=0)

        T = np.delete(T, 2 * index, axis=0)
        T = np.delete(T, 2 * index, axis=0)

        train_data_s = np.array(S[0])
        train_data_t = np.array(T[0])
        train_label = np.array(Label[0])

        for length in range(S.shape[0] - 1):
            train_data_s = np.vstack((train_data_s, S[length + 1]))
            train_data_t = np.vstack((train_data_t, T[length + 1]))
            # train_data_f = np.vstack((train_data_f, F[length + 1]))
            train_label = np.hstack((train_label, np.array(Label[length + 1])))
        index1 = [i for i in range(train_label.shape[0])]
        np.random.shuffle(index1)
        # print(index1)
        train_data_s = train_data_s[index1]
        train_data_t = train_data_t[index1]
        # train_data_f = train_data_f[index1]
        train_label = train_label[index1]
        print('test:', test_data_s.shape, Test_label.shape, test_data_t.shape)
        print('train:', train_data_s.shape, train_label.shape, train_data_t.shape)

    if cross == '10':
        print(
            "=======================================Sess{} test====================================".format(index + 1))

        test_data_s = np.array(S[index])
        test_data_t = np.array(T[index])
        Test_label = np.hstack(np.array(Label[index]))

        del Label[index]

        S = np.delete(S, index, axis=0)

        T = np.delete(T, index, axis=0)

        # F = np.delete(F, index, axis=0)

        train_data_s = np.array(S[0])
        train_data_t = np.array(T[0])
        # train_data_f = np.array(F[0])
        train_label = np.array(Label[0])

        for length in range(S.shape[0] - 1):
            train_data_s = np.vstack((train_data_s, S[length + 1]))
            train_data_t = np.vstack((train_data_t, T[length + 1]))
            # train_data_f = np.vstack((train_data_f, F[length + 1]))
            train_label = np.hstack((train_label, np.array(Label[length + 1])))
        index1 = [i for i in range(train_label.shape[0])]
        np.random.shuffle(index1)
        # print(index1)
        train_data_s = train_data_s[index1]
        train_data_t = train_data_t[index1]
        # train_data_f = train_data_f[index1]
        train_label = train_label[index1]
        print('test:', test_data_s.shape, Test_label.shape, test_data_t.shape, )
        print('train:', train_data_s.shape, train_label.shape, train_data_t.shape)

    if cross == '10r':
        l = list(range(0, 5529))
        random.shuffle(l)
        lll = {}
        print(
            "=======================================Sess{} test====================================".format(index + 1))
        data_s = np.array(S[0])
        data_t = np.array(T[0])
        # data_f = np.array(F[0])
        for length in range(9):
            data_s = np.vstack((data_s, S[length + 1]))
            data_t = np.vstack((data_t, T[length + 1]))
            # data_f = np.vstack((data_f, F[length + 1]))
        label = np.hstack(np.array(Label))
        ll = lll
        test_data_s = data_s[ll[index]]
        test_data_t = data_t[ll[index]]
        # test_data_f = data_f[ll[index]]
        Test_label = label[ll[index]]

        l_train = []
        for length in ll.keys():
            if length != index:
                l_train = l_train + ll[length]
        print(ll.keys())

        train_data_s = data_s[l_train]
        train_data_t = data_t[l_train]
        # train_data_f = data_f[l_train]
        train_label = label[l_train]
        print('test:', test_data_s.shape, Test_label.shape, test_data_t.shape)
        print('train:', train_data_s.shape, train_label.shape, train_data_t.shape)
    return train_data_s, train_data_t, train_label, test_data_s, test_data_t, Test_label
def read_data_MELD(num_classes):
    F3 = open('./features/MELD_ST.pkl', 'rb')
    S_train, T_train, Label1, emt1, S_test, T_test, Label2, emt2, S_valid, T_valid, Label3, emt3 = pickle.load(F3)
    if num_classes == 7:
        S_train = np.concatenate((np.array(S_train[0:9988]), np.array(S_train[9988: 2 * 9988])), axis=-1)
        T_train = np.array(T_train)
        train_label = np.array(Label1)
        S_test = np.concatenate((np.array(S_test[0:2610]), np.array(S_test[2610: 2 * 2610])), axis=-1)
        T_test = np.array(T_test)
        Test_label = np.array(Label2)
        S_valid = np.concatenate((np.array(S_valid[0:1108]), np.array(S_valid[1108: 2 * 1108])), axis=-1)
        T_valid = np.array(T_valid)
        Valid_label = np.array(Label3)
    if num_classes == 5:
        S_train = np.concatenate((np.array(S_train[0:9988]), np.array(S_train[9988: 2 * 9988])), axis=-1)
        T_train = np.array(T_train)
        train_label_a = np.array(Label1)
        S_train = np.delete(S_train, np.where(train_label_a == 5), axis=0)
        T_train = np.delete(T_train, np.where(train_label_a == 5), axis=0)
        train_label_a_ = np.delete(train_label_a, np.where(train_label_a == 5), axis=0)
        S_train = np.delete(S_train, np.where(train_label_a_ == 6), axis=0)
        T_train = np.delete(T_train, np.where(train_label_a_ == 6), axis=0)
        train_label = np.delete(train_label_a_, np.where(train_label_a_ == 6), axis=0)

        S_test = np.concatenate((np.array(S_test[0:2610]), np.array(S_test[2610: 2 * 2610])), axis=-1)
        T_test = np.array(T_test)
        Test_label_a = np.array(Label2)
        S_test = np.delete(S_test, np.where(Test_label_a == 5), axis=0)
        T_test = np.delete(T_test, np.where(Test_label_a == 5), axis=0)
        Test_label_a_ = np.delete(Test_label_a, np.where(Test_label_a == 5), axis=0)
        S_test = np.delete(S_test, np.where(Test_label_a_ == 6), axis=0)
        T_test = np.delete(T_test, np.where(Test_label_a_ == 6), axis=0)
        Test_label = np.delete(Test_label_a_, np.where(Test_label_a_ == 6), axis=0)

        S_valid = np.concatenate((np.array(S_valid[0:1108]), np.array(S_valid[1108: 2 * 1108])), axis=-1)
        T_valid = np.array(T_valid)
        Valid_label_a = np.array(Label3)
        S_valid = np.delete(S_valid, np.where(Valid_label_a == 5), axis=0)
        T_valid = np.delete(T_valid, np.where(Valid_label_a == 5), axis=0)
        Valid_label_a_ = np.delete(Valid_label_a, np.where(Valid_label_a == 5), axis=0)
        S_valid = np.delete(S_valid, np.where(Valid_label_a_ == 6), axis=0)
        T_valid = np.delete(T_valid, np.where(Valid_label_a_ == 6), axis=0)
        Valid_label = np.delete(Valid_label_a_, np.where(Valid_label_a_ == 6), axis=0)
    print('train set:', S_train.shape, T_train.shape, train_label)
    print('valid set:', S_valid.shape, T_valid.shape, Valid_label)
    print('test set:', S_test.shape, T_test.shape, Test_label)
    return S_train, T_train, train_label, S_test, T_test, Test_label, S_valid, T_valid, Valid_label