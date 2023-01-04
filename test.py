import pickle
import math
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, f1_score
import os
import torch.nn.functional as F
import argparse
import torch
from model import SAMS
from read_data import read_data_IE, read_data_MELD
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=128)
parser.add_argument('-d', default='IEMOCAP')
parser.add_argument('-c', '--cross', default='5')
parser.add_argument('-m', default='st')
parser.add_argument('-nc', type=int, default=4)
parser.add_argument('-nn', type=int, default=1)
parser.add_argument('--seed', type = int, default = 2)
opt = parser.parse_args()
# Device configuration
device = torch.device('cuda')
# Hyper parameters
stf = opt.m
cross = opt.cross
batch_size = opt.batch_size
data = opt.d
num_classes = opt.nc
model_path = './model_save/'
nnn = opt.nn
if data == 'IEMOCAP':
    if cross == '5':
        D = 5
        num_classes = 4
        WA = torch.empty(5, 1)
        UA = torch.empty(5, 1)
        WA2 = torch.empty(5, 1)
    if cross == '10':
        D = 10
        num_classes = 4
        WA = torch.empty(10, 1)
        UA = torch.empty(10, 1)
        WA2 = torch.empty(10, 1)
    if cross == '10r':
        D = 10
        num_classes = 4
        WA = torch.empty(10, 1)
        UA = torch.empty(10, 1)
        WA2 = torch.empty(10, 1)
    for index in range(D):
        train_data, train_data1, train_label, test_data, test_data1, Test_label = read_data_IE(
            cross, index)
        best = 0
        if os.path.exists(model_path +  'IE_sess{}_{}.ckpt'.format((index + 1), cross)):
            model_pre = SAMS(num_classes, 78, 768).to(device)
            model_pre.load_state_dict(torch.load(model_path +  'IE_sess{}_{}.ckpt'.format((index + 1), cross)))
            with torch.no_grad():
                correct = 0
                total = 0
                out = []
                n0 = math.ceil((test_data.shape[0] / batch_size))
                outputs = torch.zeros(test_data.shape[0], num_classes)
                for ii in range(n0):
                    start = ii * batch_size
                    end = min(start + batch_size, test_data.shape[0])
                    x = torch.Tensor(test_data[start:end])
                    xf = torch.Tensor(test_data1[start:end])
                    x = x.to(device)
                    xf = xf.to(device)
                    outputs[start:end] = model_pre(x, xf, 0, stf)
                y1 = torch.Tensor(Test_label.ravel())

                _, predicted = torch.max(F.softmax(outputs, 1), 1)
                ave_uar = recall_score(y1, predicted, average='macro')
                ave_acc = accuracy_score(y1, predicted)
                print(confusion_matrix(y1, predicted))
                print('test acc: {:.3f}, test uar: {:.3f}'.format(ave_acc, ave_uar))
                WA[index] = ave_acc
                UA[index] = ave_uar
    print('========================Final_Result===========================')
    print('5-cross average Weighted Accuracy:', WA.sum() / D)
    print('5-cross average UnWeighted Accuracy:', UA.sum() / D)

if data == 'MELD':
    S_train, T_train, train_label, S_test, T_test, Test_label, S_valid, T_valid, Valid_label = read_data_MELD(num_classes)
    best = 0
    if os.path.exists(model_path + 'me_{}way_{}.ckpt'.format(num_classes, nnn)):
        model_pre2 = SAMS(num_classes,78, 768).cuda()
        model_pre2.load_state_dict(torch.load(model_path + 'me_{}way_{}.ckpt'.format(num_classes, nnn)))
        with torch.no_grad():
            correct = 0
            total = 0
            out = []
            n0 = math.ceil((S_test.shape[0] / batch_size))
            outputs = torch.zeros(S_test.shape[0], num_classes)
            for ii in range(n0):
                start = ii * batch_size
                end = min(start + batch_size, S_test.shape[0])
                x = torch.Tensor(S_test[start:end])
                xf = torch.Tensor(T_test[start:end])
                x = x.to(device)
                xf = xf.to(device)
                outputs[start:end] = model_pre2(x, xf, 0, stf)
            y1 = torch.Tensor(Test_label.ravel())
            _, predicted = torch.max(F.softmax(outputs, 1), 1)
            uar = recall_score(y1, predicted, average='macro')
            acc = accuracy_score(y1, predicted)
            f1 = f1_score(y1, predicted, average='weighted')
            print('**************The Final Test*************** %')
            print(confusion_matrix(y1, predicted))
            print('test acc: {:.3f}, test f1: {:.3f}'.format(acc,  f1))


