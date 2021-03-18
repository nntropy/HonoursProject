# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File : model_HighWay_BiLSTM_1.py
# @Last Modify Time : 2018/07/19 22:35
# @Contact : bamtercelboo@{gmail.com, 163.com}
# coding=utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import numpy as np
import torch.nn.init as init
import torch.optim.lr_scheduler as lr_scheduler
import sys
import os
import torch.nn.functional as F
import shutil

seed_num = random.randint(10, 920974812423598325)
torch.manual_seed(seed_num)
random.seed(seed_num)


class HBiLSTM(nn.Module):

    def __init__(self, hidden_dim=512, num_layers=4, embed_dim=2048, dropout=0.6, batch_size=16, init_weight_value=2.0,
                 init_weight=True):
        super(HBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        D = embed_dim
        self.bilstm = nn.LSTM(D, self.hidden_dim, num_layers=self.num_layers, bias=True, bidirectional=True
                              , dropout=dropout)
        # if self.args.cuda is True:
        #     self.bilstm.flatten_parameters()
        if init_weight:
            print("Initiating W .......")
            # init.xavier_uniform(self.bilstm.all_weights[0][0], gain=np.sqrt(args.init_weight_value))
            init.xavier_uniform_(self.bilstm.all_weights[0][0], gain=np.sqrt(init_weight_value))
            init.xavier_uniform_(self.bilstm.all_weights[0][1], gain=np.sqrt(init_weight_value))
            init.xavier_uniform_(self.bilstm.all_weights[1][0], gain=np.sqrt(init_weight_value))
            init.xavier_uniform_(self.bilstm.all_weights[1][1], gain=np.sqrt(init_weight_value))
        if self.bilstm.bias is True:
            print("Initiating bias......")
            a = np.sqrt(2 / (1 + 600)) * np.sqrt(3)
            init.uniform_(self.bilstm.all_weights[0][2], -a, a)
            init.uniform_(self.bilstm.all_weights[0][3], -a, a)
            init.uniform_(self.bilstm.all_weights[1][2], -a, a)
            init.uniform_(self.bilstm.all_weights[1][3], -a, a)
        print(self.bilstm.all_weights)

        in_feas = self.hidden_dim
        self.fc1 = self.init_Linear(in_fea=in_feas, out_fea=in_feas, bias=True)
        # Highway gate layer  T in the Highway formula
        self.gate_layer = self.init_Linear(in_fea=in_feas, out_fea=in_feas, bias=True)

        # in_fea = self.args.embed_dim
        # out_fea = self.args.lstm_hidden_dim * 2
        # self.fc1 = self.init_Linear(in_fea=in_fea, out_fea=out_fea, bias=True)
        # self.gate_layer = self.init_Linear(in_fea=in_fea, out_fea=out_fea, bias=True)

        # in_fea = self.args.embed_dim
        # out_fea = self.args.lstm_hidden_dim * 2
        # self.fc1 = self.init_Linear(in_fea=in_fea, out_fea=out_fea, bias=True)
        # self.gate_layer = self.init_Linear(in_fea=in_fea, out_fea=out_fea, bias=True)
        # self.fc1 = None
        # self.gate_layer = None

        # if bidirection convert dim
        self.convert_layer = self.init_Linear(in_fea=self.hidden_dim * 2,
                                              out_fea=embed_dim, bias=True)
        self.hidden = self.init_hidden(self.num_layers, batch_size)

    def init_hidden(self, num_layers, batch_size):
        # the first is the hidden h
        # the second is the cell  c
        # for cuda
        return (Variable(torch.zeros(2 * num_layers, batch_size, self.hidden_dim)).cuda(),
                Variable(torch.zeros(2 * num_layers, batch_size, self.hidden_dim)).cuda())

    def init_Linear(self, in_fea, out_fea, bias):
        linear = nn.Linear(in_features=in_fea, out_features=out_fea, bias=bias)
        # for cuda
        return linear.cuda()

    def forward(self, x, hidden):
        # handle the source input x
        source_x = x
        x, hidden = self.bilstm(x)
        normal_fc = torch.transpose(x, 0, 1)
        # normal_fc = self.gate_layer(normal_fc)
        x = torch.transpose(x, 0, 1)
        x = torch.transpose(x, 1, 2)
        # normal layer in the formula is H
        source_x = torch.transpose(source_x, 0, 1)

        in_fea = self.embed_dim
        out_fea = self.hidden_dim * 2
        self.fc1 = self.init_Linear(in_fea=in_fea, out_fea=out_fea, bias=True)
        self.gate_layer = self.init_Linear(in_fea=in_fea, out_fea=out_fea, bias=True)
        # self.gate_layer = self.init_Linear(in_fea=out_fea, out_fea=in_fea, bias=True)

        # the first way to convert 3D tensor to the Linear
        source_x = source_x.contiguous()
        information_source = source_x.view(source_x.size(0) * source_x.size(1), source_x.size(2))
        information_source = self.gate_layer(information_source)
        information_source = information_source.view(source_x.size(0), source_x.size(1), information_source.size(1))

        '''
        # the another way to convert 3D tensor to the Linear
        list = []
        for i in range(source_x.size(0)):
            information_source = self.gate_layer(source_x[i])
            information_source = information_source.unsqueeze(0)
            list.append(information_source)
        information_source = torch.cat(list, 0)
        '''

        # transformation gate layer in the formula is T
        transformation_layer = torch.sigmoid(information_source)
        # carry gate layer in the formula is C
        carry_layer = 1 - transformation_layer
        # formula Y = H * T + x * C
        allow_transformation = torch.mul(normal_fc, transformation_layer)

        '''
        # you also can choose the strategy that zero-padding
        zeros = torch.zeros(source_x.size(0), source_x.size(1), carry_layer.size(2) - source_x.size(2))
        if self.args.cuda is True:
            source_x = Variable(torch.cat((zeros, source_x.data), 2)).cuda()
        else:
            source_x = Variable(torch.cat((zeros, source_x.data), 2))
        allow_carry = torch.mul(source_x, carry_layer)
        '''
        # the information_source compare to the source_x is for the same size of x,y,H,T
        allow_carry = torch.mul(information_source, carry_layer)
        # allow_carry = torch.mul(source_x, carry_layer)
        information_flow = torch.add(allow_transformation, allow_carry)

        information_flow = information_flow.contiguous()
        information_convert = information_flow.view(information_flow.size(0) * information_flow.size(1),
                                                    information_flow.size(2))
        information_convert = self.convert_layer(information_convert)
        information_convert = information_convert.view(information_flow.size(0), information_flow.size(1),
                                                       information_convert.size(1))

        '''
        # the another way 
        convert = []
            for j in range(information_flow.size(0)):
                # print(information_flow[i].size())
                information_convert = self.convert_layer(information_flow[j])
                # print(information_convert.size())
                information_convert = information_convert.unsqueeze(0)
                convert.append(information_convert)
            information_convert = torch.cat(convert, 0)
        '''
        information_convert = torch.transpose(information_convert, 0, 1)
        return information_convert, hidden


def eval(data_iter, model, args, scheduler):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch[:, :2048], batch[:, -1:]
        # feature.t_()
        feature.reshape(1, len(batch), 2048)
        target.reshape(1, len(batch))
        target.data.sub_(1)
        # feature, target = batch.text, batch.label.data.sub_(1)
        feature, target = feature.cuda(), target.cuda()
        # feature.data.t_(), target.data.sub_(1)  # batch first, index align
        # feature.data.t_(),\
        # target.data.sub_(1)  # batch first, index align
        # target = autograd.Variable(target)

        model.hidden = model.init_hidden(model.num_layers, model.batch_size)
        if feature.size(1) != args.batch_size:
            # print("aaa")
            # continue
            model.hidden = model.init_hidden(model.num_layers, feature.size(1))
        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)
        # scheduler.step(loss.data[0])

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = loss.data[0] / size
    accuracy = float(corrects) / size * 100.0
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))


def test_eval(data_iter, model, save_path, args, model_count):
    # print(save_path)
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch[:, :2048], batch[:, -1:]
        # feature.t_()
        feature.reshape(1, len(batch), 2048)
        target.reshape(1, len(batch))
        feature, target = feature.cuda(), target.cuda()
        # feature.data.t_()
        # target.data.sub_(1)  # batch first, index align
        # target = autograd.Variable(target)
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        model.hidden = model.init_hidden(model.num_layers, model.batch_size)
        if feature.size(1) != args.batch_size:
            # continue
            model.hidden = model.init_hidden(model.num_layers, feature.size(1))
        logit = model(feature)
        loss = F.binary_cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = loss.data[0] / size
    accuracy = float(corrects) / size * 100.0
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    print("model_count {}".format(model_count))
    # test result
    if os.path.exists("./Test_Result.txt"):
        file = open("./Test_Result.txt", "a")
    else:
        file = open("./Test_Result.txt", "w")
    file.write("model " + save_path + "\n")
    file.write("Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n".format(avg_loss, accuracy, corrects, size))
    file.write("model_count {} \n".format(model_count))
    file.write("\n")
    file.close()
    # calculate the best score in current file
    resultlist = []
    if os.path.exists("./Test_Result.txt"):
        file = open("./Test_Result.txt")
        for line in file.readlines():
            if line[:10] == "Evaluation":
                resultlist.append(float(line[34:41]))
        result = sorted(resultlist)
        file.close()
        file = open("./Test_Result.txt", "a")
        file.write("\nThe Current Best Result is : " + str(result[len(result) - 1]))
        file.write("\n\n")
        file.close()
    shutil.copy("./Test_Result.txt", "./snapshot/" + args.mulu + "/Test_Result.txt")
    # whether to delete the model after test acc so that to save space
    if os.path.isfile(save_path) and args.rm_model is True:
        os.remove(save_path)


# everything from now on is done by me, Aidan O'Connor
def train(train_iter, dev_iter, test_iter, no_epochs, model, decay=1e-8, lr=0.001):
    model.cuda()
    # get optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    # get scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    # initialize
    steps = 0
    epoch_step = 0
    model_count = 0
    # begin training
    model.train()
    # epochs
    for epoch in range(1, no_epochs + 1):
        # batch in epochs
        for batch in train_iter:
            feature, target = batch[:, :2048], batch[:, -1:]
            # feature.t_()
            feature = feature.reshape(len(batch), 1, 2048)
            target = target.reshape(len(batch), 1)
            print(feature.shape)
            print(target.shape)
            target.sub_(1)  # batch first, index align
            # cuda enabled
            feature, target = feature.cuda(), target.cuda()
            # do step
            optimizer.zero_grad()
            model.zero_grad()

            model.hidden = model.init_hidden(model.num_layers, model.batch_size)
            log, aux = model(feature.float(), model.hidden)
            print(log.shape)
            loss = F.binary_cross_entropy(log, target.long())

            loss.backward()
            optimizer.step()
            steps += 1
            # log every 50 steps
            if steps % 50 == 0:
                train_size = len(train_iter.dataset)
                corrects = (torch.max(log, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = float(corrects) / 25 * 100.0
                sys.stdout.write(
                    '\rBatch[{}/{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                                train_size,
                                                                                loss.data[0],
                                                                                accuracy,
                                                                                corrects,
                                                                                25))
            if steps % 100 == 0:
                eval(dev_iter, model)
            if steps % 400 == 0:
                if not os.path.isdir("D:\\5th\Honours\Code\models\weights"):
                    os.makedirs("D:\\5th\Honours\Code\models\weights")
                save_prefix = os.path.join("D:\\5th\Honours\Code\models\weights", 'snapshot')
                save_path = '{}_steps{}.pt'.format(save_prefix, steps)
                torch.save(model, save_path)
                print("\n", save_path, end=" ")
                test_model = torch.load(save_path)
                model_count += 1
                test_eval(test_iter, test_model, save_path, model_count)
        return model_count


