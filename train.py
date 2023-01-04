import pickle
import math
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, f1_score
import os
from torch.autograd import Variable
import argparse
import torch.nn.functional as F
import random
import torch
import torch.nn as nn
import numpy as np
from torchvision.models import ResNet
from sklearn.cluster import KMeans
from torchvision.models import ResNet
from model import SAMS
from read_data import read_data_IE, read_data_MELD
parser = argparse.ArgumentParser()
parser.add_argument('-num_steps', type=int, default=2500)
parser.add_argument('-b', '--batch_size', type=int, default=128)
parser.add_argument('-lr', type=float, default=0.0001)
parser.add_argument('-d', default='IEMOCAP')
parser.add_argument('-c', '--cross', default='5')
parser.add_argument('-m', default='st')
parser.add_argument('-nc', type=int, default=4)
parser.add_argument('--seed', type =int, default = 2)
parser.add_argument('-nn', type=int, default=1)
opt = parser.parse_args()
# Device configuration
device = torch.device('cuda')

# Hyper parameters
st = opt.m
cross = opt.cross
num_steps = opt.num_steps
batch_size = opt.batch_size
learning_rate = opt.lr
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
        if os.path.exists(model_path +'IE_sess{}_{}.ckpt'.format((index + 1), D)):
            model_pre = SAMS(num_classes, 78, 768).to(device)
            model_pre.load_state_dict(torch.load(model_path + 'IE_sess{}_{}.ckpt'.format((index + 1), D)))
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

                    outputs[start:end] = model_pre(x, xf, 0, st)
                y1 = torch.Tensor(Test_label.ravel())
                _, predicted = torch.max(F.softmax(outputs, 1), 1)
                ave_uar = recall_score(y1, predicted, average='macro')
                ave_acc = accuracy_score(y1, predicted)
                print('**************Last Model Test*************** %')
                print(confusion_matrix(y1, predicted))
                print('Pre test acc: {:.3f}, Pre test uar: {:.3f}'.format(ave_acc, ave_uar))
                best = ave_uar

        model = SAMS(num_classes, 78, 768).to(device)
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        MSE = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        Epoch = 1
        uar_list = []
        acc_list = []
        x0 = []
        x1 = []
        x2 = []
        yy = []
        n = math.ceil((train_data.shape[0] / batch_size))
        for i in range(n):
            start = i * batch_size
            end = min(start + batch_size, train_data.shape[0])
            yy.append(train_label[start:end].ravel())
            x0.append(torch.Tensor(train_data[start:end]))
            x1.append(torch.Tensor(train_data1[start:end]))

        it = iter(x0)
        it1 = iter(x1)
        itt = iter(yy)
        # if os.path.exists('model_path + 'IE_sess{}.ckpt'.format(index+1)):
        #    model.load_state_dict(torch.load(model_path +'IE_sess{}.ckpt'.format(index+1)))
        for step in range(num_steps):
            try:
                x = next(it)
                xf = next(it1)
                labels = next(itt)

            except StopIteration:
                print('=========Epoch {} finished !========='.format(Epoch))
                Epoch += 1
                it = iter(x0)
                it1 = iter(x1)
                itt = iter(yy)
                x = next(it)
                xf = next(it1)
                labels = next(itt)

            labels = torch.LongTensor(labels)
            x = x.to(device)
            xf = xf.to(device)
            labels = labels.to(device)
            # Forward pass
            if st == 'st':
                outputs, outa1, outa2, outa11, outa22, outs, outt = model(x, xf, 1, st)
                loss_st = MSE(outa1, outa22)
                loss_ts = MSE(outa2, outa11)
                loss_s = criterion(outs, labels)
                loss_t = criterion(outt, labels)
                loss_p = criterion(outputs, labels)
                loss = loss_p + loss_st + loss_ts + loss_s + loss_t

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 10 == 0:
                print('step: {}, Loss: {:.4f}'.format(step + 1, loss.item()))
                model.eval()
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

                        outputs[start:end] = model(x, xf, 0, st)
                    y1 = torch.Tensor(Test_label.ravel())

                    _, predicted = torch.max(F.softmax(outputs, 1), 1)

                    ave_uar = recall_score(y1, predicted, average='macro')
                    ave_acc = accuracy_score(y1, predicted)

                    print('**************Test*************** %')
                    print(confusion_matrix(y1, predicted))
                    print('Ave test acc: {:.3f}, Ave test uar: {:.3f}'.format(ave_acc, ave_uar))
                    uar_list.append(float(recall_score(y1, predicted, average='macro')))
                    acc_list.append(float(accuracy_score(y1, predicted)))
                    if best < ave_uar:
                        best = ave_uar
                        torch.save(model.state_dict(), model_path + 'IE_sess{}_{}.ckpt'.format((index + 1), D))
                model.train()
        print('optimal step: {}, optimal uar: {}'.format((np.argmax(uar_list) + 1) * 10, max(uar_list)))
        WA[index] = max(acc_list)
        UA[index] = max(uar_list)
        WA2[index] = acc_list[uar_list.index(max(uar_list))]
    print('========================Final_Result===========================')
    print('Weighted Accuracy:', WA)
    print('UnWeighted Accuracy:', UA)
    print('Weighted Accuracy:', WA2)
    print('5-cross average Weighted Accuracy:', WA.sum() / D)
    print('5-cross average UnWeighted Accuracy:', UA.sum() / D)
    print('5-cross average Weighted Accuracy:', WA2.sum() / D)
if data == 'MELD':
    S_train, T_train, train_label, S_test, T_test, Test_label, S_valid, T_valid, Valid_label = read_data_MELD(num_classes)
    best = 0
    # if os.path.exists(model_path + 'me_{}way.ckpt'.format(num_classes)):
    #     model_pre2 = SAMS(num_classes,78, 768).cuda()
    #     model_pre2.load_state_dict(torch.load(model_path + 'me_{}way.ckpt'.format(num_classes)))
    #     with torch.no_grad():
    #         correct = 0
    #         total = 0
    #         out = []
    #         n0 = math.ceil((S_valid.shape[0] / batch_size))
    #         outputs = torch.zeros(S_valid.shape[0], num_classes)
    #         for ii in range(n0):
    #             start = ii * batch_size
    #             end = min(start + batch_size, S_valid.shape[0])
    #             x = torch.Tensor(S_valid[start:end])
    #             xf = torch.Tensor(T_valid[start:end])
    #             x = x.to(device)
    #             xf = xf.to(device)
    #             outputs[start:end] = model_pre2(x, xf, 0, st)
    #         y1 = torch.Tensor(Valid_label.ravel())
    #         _, predicted = torch.max(F.softmax(outputs, 1), 1)
    #         acc = accuracy_score(y1, predicted)
    #         f1 = f1_score(y1, predicted, average='weighted')
    #         print('*************Last model Valid*************** %')
    #         print(confusion_matrix(y1, predicted))
    #         print('Pre valid acc: {:.3f}, Pre valid f1: {:.3f}'.format(acc, f1))
    #         best = acc
    model = SAMS(num_classes, 78, 768).to(device)
    criterion = nn.CrossEntropyLoss()
    BCE = nn.BCEWithLogitsLoss()
    MSE = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    Epoch = 1
    acc_list = []
    f1_list = []
    uar_list_v = []
    acc_list_v = []
    f1_list_v = []
    loss_v = []
    x0 = []
    x1 = []
    yy = []
    n = math.ceil((S_train.shape[0] / batch_size))
    for i in range(n):
        start = i * batch_size
        end = min(start + batch_size, S_train.shape[0])
        yy.append(train_label[start:end].ravel())
        x0.append(torch.Tensor(S_train[start:end]))
        x1.append(torch.Tensor(T_train[start:end]))
    it = iter(x0)
    it1 = iter(x1)
    itt = iter(yy)
    for step in range(num_steps):

        try:
            x = next(it)
            xf = next(it1)
            labels = next(itt)

        except StopIteration:
            print('=========Epoch {} finished !========='.format(Epoch))
            Epoch += 1
            it = iter(x0)
            it1 = iter(x1)
            itt = iter(yy)
            x = next(it)
            xf = next(it1)
            labels = next(itt)
        labels = torch.LongTensor(labels)
        x = x.to(device)
        xf = xf.to(device)
        labels = labels.to(device)
        if st == 'st':
            outputs, outa1, outa2, outa11, outa22, outs, outt = model(x, xf, 1, st)

            loss_st = F.l1_loss(outa1, outa22)
            loss_ts = F.l1_loss(outa2, outa11)

            loss_s = criterion(outs, labels)
            loss_t = criterion(outt, labels)
            loss_p = criterion(outputs, labels)
            # loss = loss_p + l1_loss + l2_loss + loss_s + loss_t
            loss = loss_p + loss_st + loss_ts + loss_s + loss_t
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 10 == 0:
            # print('step: {}, Loss: {:.4f}'.format(step + 1, loss.item()))
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                out = []
                n0 = math.ceil((S_train.shape[0] / batch_size))
                outputs = torch.zeros(S_train.shape[0], num_classes)
                for ii in range(n0):
                    start = ii * batch_size
                    end = min(start + batch_size, S_train.shape[0])
                    x = torch.Tensor(S_train[start:end])
                    xf = torch.Tensor(T_train[start:end])
                    x = x.to(device)
                    xf = xf.to(device)
                    outputs[start:end] = model(x, xf, 0, st)
                y1 = torch.Tensor(train_label.ravel())
                _, predicted = torch.max(F.softmax(outputs, 1), 1)
                uar = recall_score(y1, predicted, average='macro')
                f1 = f1_score(y1, predicted, average='weighted')
                print('**************Train*************** %')
                print(confusion_matrix(y1, predicted))
                print('train acc: {:.3f}, train f1: {:.3f}'.format(acc, f1))
                print('step: {}, Loss: {:.4f}'.format(step + 1, loss.item()))
            model.train()
        if (step + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                out = []
                n0 = math.ceil((S_valid.shape[0] / batch_size))
                outputs = torch.zeros(S_valid.shape[0], num_classes)
                for ii in range(n0):
                    start = ii * batch_size
                    end = min(start + batch_size, S_valid.shape[0])
                    x = torch.Tensor(S_valid[start:end])
                    xf = torch.Tensor(T_valid[start:end])
                    x = x.to(device)
                    xf = xf.to(device)
                    outputs[start:end] = model(x, xf, 0, st)
                y1 = torch.Tensor(Valid_label.ravel())

                _, predicted = torch.max(F.softmax(outputs, 1), 1)
                acc = accuracy_score(y1, predicted)
                f1 = f1_score(y1, predicted, average='weighted')

                print('**************Valid*************** %')
                print(confusion_matrix(y1, predicted))
                print('valid acc: {:.3f}, valid f1: {:.3f}, valid loss: {:.3f}'.format(acc, f1,
                                                                                                          criterion(
                                                                                                              outputs,
                                                                                                              torch.LongTensor(
                                                                                                                  Valid_label)).item()))
                uar_list_v.append(float(recall_score(y1, predicted, average='macro')))
                acc_list_v.append(float(accuracy_score(y1, predicted)))
                f1_list_v.append(float(f1_score(y1, predicted, average='weighted')))
                loss_v.append(criterion(outputs, torch.LongTensor(Valid_label)).item())
                if best < acc:
                    best = acc
                    torch.save(model.state_dict(), model_path +'me_{}way_{}.ckpt'.format(num_classes, nnn))
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
                    outputs[start:end] = model(x, xf, 0, st)
                y1 = torch.Tensor(Test_label.ravel())
                _, predicted = torch.max(F.softmax(outputs, 1), 1)
                acc = accuracy_score(y1, predicted)
                f1 = f1_score(y1, predicted, average='weighted')
                print('**************Test*************** %')
                print(confusion_matrix(y1, predicted))
                # print('test acc: {:.3f}, test uar: {:.3f}'.format(acc, uar))
                print('test acc: {:.3f}, test uar: {:.3f}, test f1: {:.3f}'.format(acc, uar, f1))
                acc_list.append(float(accuracy_score(y1, predicted)))
                f1_list.append(float(f1_score(y1, predicted, average='weighted')))
                # if os.path.exists('/home/amax/MXH/ASF/result/me'):
                #    model.load_state_dict(torch.load('/home/amax/MXH/ASF/ptorch_3D/result/me'))
                # if best < uar:
                #     best = uar
                #     torch.save(model.state_dict(), '/home/amax/MXH/ASF/result/me_sess{}.ckpt'.format(index+1))
            model.train()
    print("============Valid==================")
    print('optimal step: {}, optimal acc: {}, f1: {}'.format((np.argmax(acc_list_v) + 1) * 10, max(acc_list_v),
                                                             f1_list_v[acc_list_v.index(max(acc_list_v))]))
    print('optimal step: {}, optimal f1: {}'.format((np.argmax(f1_list_v) + 1) * 10, max(f1_list_v)))
    print("============Test==================")
    print('optimal step: {}, optimal acc: {}, f1: {}'.format((np.argmax(acc_list) + 1) * 10, max(acc_list),
                                                             f1_list[acc_list.index(max(acc_list))]))
    print('optimal step: {}, optimal f1: {}'.format((np.argmax(f1_list) + 1) * 10, max(f1_list)))

    if os.path.exists(model_path +'me_{}way_{}.ckpt'.format(num_classes, nnn)):
        model_pre2 = SAMS(num_classes, 78, 768).to(device)
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
                outputs[start:end] = model_pre2(x, xf, 0, st)
            y1 = torch.Tensor(Test_label.ravel())
            _, predicted = torch.max(F.softmax(outputs, 1), 1)
            acc = accuracy_score(y1, predicted)
            f1 = f1_score(y1, predicted, average='weighted')
            print('**************The Final Test*************** %')
            print(confusion_matrix(y1, predicted))

            print('test acc: {:.3f}, test f1: {:.3f}'.format(acc, f1))


