from cmath import sqrt
import math
from os import replace
from turtle import forward, shape
from matplotlib.pyplot import axis, cla
import torch
import torch.nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules import padding, transformer
from torch.nn.modules.activation import ReLU
import copy

class VanillaGCN(torch.nn.Module):
    def __init__(self, region_size, features_in, features_out):
        super(VanillaGCN, self).__init__()
        self.W = torch.nn.Linear(features_in, features_out)
        self.batch_normalization = torch.nn.BatchNorm1d(region_size)

    def forward(self,  features, Adjancy_Matrix):
        output = F.leaky_relu(
            self.batch_normalization(
                self.W(torch.matmul(Adjancy_Matrix, features))))
        return output


class GraphAttentionLayer(torch.nn.Module):
    def __init__(self, region_size, in_features_num, out_features_num, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features_num = in_features_num   # 节点表示向量的输入特征维度
        self.out_features_num = out_features_num   # 节点表示向量的输出特征维度
        self.dropout = dropout    # dropout参数
        self.alpha = alpha     # leakyrelu激活的参数
        self.concat = concat   # 如果为true, 再进行elu激活
        # 定义可训练参数，即论文中的W和a
        self.W = torch.nn.Parameter(torch.zeros(
            size=(in_features_num, out_features_num)))
        torch.nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        self.a = torch.nn.Parameter(torch.zeros(size=(2*out_features_num, 1)))
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)   # xavier初始化

        self.sparse = torch.nn.Parameter(torch.zeros(size=(region_size, region_size)))
        torch.nn.init.xavier_uniform_(self.sparse.data, gain=1.414)   # xavier初始化

        # 定义leakyrelu激活函数
        self.leakyrelu = torch.nn.LeakyReLU(self.alpha)
        self.k = 10   # only consider the top-k similar nodes when calculate the attention

    def forward(self, inp, adj):
        hidden_features = torch.matmul(inp, self.W)
        N = hidden_features.size()[1]
        adj_input = torch.cat([hidden_features.repeat(1, 1, N).view(inp.size()[0], N*N, -1), hidden_features.repeat(1, N, 1)], dim=2).view(inp.size()[0], N, -1, 2*self.out_features_num)
        e = self.leakyrelu(torch.matmul(adj_input, self.a).squeeze(3))

        print("e: ", e.size())
        print("e tensor is on device:", e.device)
        adj = adj.cuda()
        zero_vec = -1e13 * torch.ones_like(e).cuda()
        attention = torch.where(adj > 0.00001, e, zero_vec)
        # attention = torch.mul(e, adj)
        # sparse = F.sign(self.sparse - 0.5)
        sparse = torch.relu(torch.sign(self.sparse - 0.3))

        attention = torch.matmul(attention, self.sparse)
        # attention = attention * 100
        print(sparse)
        print("self.sparse zero count: ", torch.sum(torch.eq(self.sparse, 0.0)))
        attention = F.softmax(attention, dim=2)
        # attention = F.normalize(attention, p=2, dim=2)
        attention = F.dropout(attention, self.dropout,
                              training=self.training)

        transposed_tensor = attention.transpose(1, 2)
        attention = (attention + transposed_tensor)/2


        h_prime = torch.matmul(attention, hidden_features)
        if self.concat:
            h_prime = F.elu(h_prime)

        return h_prime, attention

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features_num) + ' -> ' + str(self.out_features_num) + ')'


class whole_network(torch.nn.Module):
    def __init__(self, region_size, in_features_num, hidden_num1, hidden_num2, out_features_num, correlation_rate, dropout):
        super(whole_network, self).__init__()

        self.correlation = None
        self.in_features_num = in_features_num
        self.hidden_num1 = hidden_num1
        self.hidden_num2 = hidden_num2
        self.out_features_num = out_features_num
        self.W1 = torch.nn.Parameter(torch.zeros(
            size=(self.in_features_num, self.hidden_num1)))
        self.W2 = torch.nn.Parameter(torch.zeros(
            size=(self.hidden_num1, self.hidden_num2)))
        self.W3 = torch.nn.Parameter(torch.zeros(
            size=(self.hidden_num2, self.out_features_num)))    
        # self.alpha_1=(torch.nn.Parameter(torch.ones(1))/2).cuda()
        # self.alpha_2=(torch.nn.Parameter(torch.ones(1))/2).cuda()
        self.region_size = region_size
        self.correlation_rate = correlation_rate
        self.dropout = dropout

        self.gat_layer_1 = GraphAttentionLayer(region_size=self.region_size,
            in_features_num=self.in_features_num, out_features_num=self.hidden_num1, dropout=self.dropout, alpha=0.01, concat=False)

        self.gat_layer_2 = GraphAttentionLayer(region_size=self.region_size,
            in_features_num=self.hidden_num1, out_features_num=self.hidden_num2, dropout=self.dropout, alpha=0.01, concat=False)
        
        self.gat_layer_3 = GraphAttentionLayer(region_size=self.region_size,
            in_features_num=self.hidden_num2, out_features_num=8, dropout=0, alpha=0.01, concat=False)

        # self.gcn_layer1_1 = VanillaGCN(region_size, 32, 8)
        # # self.gcn_layer1_2 = VanillaGCN(16, 8)
        # self.gcn_layer1_2 = VanillaGCN(region_size, 8, 4)
        # self.gcn_layer1_3 = VanillaGCN(region_size, 4, 1)

        # self.gcn_layer1_1 = VanillaGCN(region_size, 32, 8)
        self.gcn_layer1_2 = VanillaGCN(region_size, 8, 1)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                torch.nn.init.constant_(m.bias, 0.0)

    def forward(self, x, edge_weight): 
        # self.correlation = torch.ones((BOLD_signals.shape[0], 90, 90)).cuda()
        # for i in range(BOLD_signals.shape[0]):
        #     self.correlation[i] = torch.corrcoef(BOLD_signals[i])

        # correlation = self.correlation
        # print("edge_weight size : ", edge_weight.size())
        new_edge_weight = torch.sub(1, edge_weight)
        # 保存对应的
        # edge_weight[edge_weight < self.correlation_rate] = 1 # 0.15
        # edge_weight[edge_weight != 1] = 0
        
        # if not self.training:
        #     print("======================= test", self.correlation_rate)
        #     attention = edge_weight[0, :, :]
        #     for row in attention:
        #         for element in row:
        #             print(element.item(), end=' ')
        #         print()
        # else:

        #     print("======================= train", self.correlation_rate)

        # edge_weight_size = edge_weight.size()
        # edge_weight_flat = edge_weight.view(edge_weight_size[0], -1)
        # num_ones = int(0.5 * edge_weight_flat.size(1))
        # for i in range(edge_weight_flat.size(0)):
        #     edge_weight_flat[i, torch.argsort(edge_weight_flat[i])[:num_ones]] = 1
        #     edge_weight_flat[i, edge_weight_flat[i] != 1] = 0
        # edge_weight_new = edge_weight_flat.view(edge_weight_size[0], 99, 99)

        gat_output_1, fc1 = self.gat_layer_1(x, new_edge_weight)
        # print("gat_output_1: ", gat_output_1.size())
        # print("fc1: ", fc1.size())
        gat_output_2, fc2 = self.gat_layer_2(gat_output_1, fc1)

        gat_output_3, fc3 = self.gat_layer_3(gat_output_2, fc2)
        # print("gat_output_2: ", gat_output_2.size())
        # print("fc2: ", fc2.size())
        # gcn_output_1_1 = self.gcn_layer1_1(gat_output_2, fc2)
        # print("gcn_output_1_1: ", gcn_output_1_1.size())
        # print("fc2: ", fc2.size())        
        gcn_output_1_2 = self.gcn_layer1_2(gat_output_3, fc3)
        # print("gcn_output_1_2: ", gcn_output_1_2.size())
        # print("fc2: ", fc2.size())           
        # gcn_output_1_3 = self.gcn_layer1_3(gcn_output_1_2, fc2)

        # gcn_output_1_4 = self.gcn_layer1_4(gcn_output_1_3, fc3)
        # print("gcn_output_1_3: ", gcn_output_1_3.size())
        # print("fc2: ", fc2.size())


        return gcn_output_1_2, fc3


    # def forward(self, inp, adj):
    #
    #     hidden_features = torch.matmul(inp, self.W)
    #     # 打印特征
    #     # if not self.training:
    #     #     print("================== hidden_features_print")
    #     #     hidden_features_print = hidden_features[0, :, :]
    #     #     for row in hidden_features_print:
    #     #         for element in row:
    #     #             print(element.item(), end=' ')
    #     #         print()
    #
    #     N = hidden_features.size()[1]
    #     # hidden_features_index_0 = hidden_features[0, :, :]
    #     adj_input = torch.cat([hidden_features.repeat(1, 1, N).view(inp.size()[0], N*N, -1), hidden_features.repeat(1, N, 1)], dim=2).view(inp.size()[0], N, -1, 2*self.out_features_num)
    #     # adj_input = torch.cat((hidden_features.repeat(1, N).view(N * N, -1), hidden_features.repeat(N, 1)), dim=1).view(N, -1, 2 * self.out_features_num)
    #
    #     # # print((torch.matmul(adj_input, self.a)).size())
    #     # if not self.training:
    #     #     print("================== adj_input_print")
    #     #     adj_input_print = adj_input[0, :, :]
    #     #     print("adj_input_print : ", adj_input.size())
    #     #     print("a : ", self.a.size())
    #
    #     #     print("hidden_features_index_0 : ", hidden_features_index_0.size())
    #     #     adj_input_index_0 = torch.cat([hidden_features_index_0.repeat(1, N).view(N * N, -1), hidden_features_index_0.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features_num)
    #     #     print("adj_input_index_0 : ", adj_input_index_0.size())
    #     # else:
    #     #     print("================== adj_input_print training")
    #     #     print("adj_input_print : ", adj_input.size())
    #
    #
    #     e = self.leakyrelu(torch.matmul(adj_input, self.a).squeeze(3))
    #
    #     # if not self.training:
    #     #     print("================== e")
    #     #     e_print = e[0, :, :]
    #     #     for row in e_print:
    #     #         for element in row:
    #     #             print(element.item(), end=' ')
    #     #         print()
    #     # else:
    #     #     print("================== e training")
    #     #     print(e.size())
    #
    #     # print(e.sum(axis=1))
    #     zero_vec = -1e13 * torch.ones_like(e)
    #     attention = torch.where(adj > 0.00001, e, zero_vec)
    #     # if not self.training:
    #     #     print("================== attention")
    #     #     attention_print = attention[0, :, :]
    #     #     for row in attention_print:
    #     #         for element in row:
    #     #             print(element.item(), end=' ')
    #     #         print()
    #     # print(attention.sum(axis=2))
    #     attention = F.softmax(attention, dim=2)
    #     # print(attention)
    #     # print(attention.sum(axis=2))
    #     attention = F.dropout(attention, self.dropout,
    #                           training=self.training)
    #
    #     # 将两个 [3, 3] 的小矩阵进行转置
    #     transposed_tensor = attention.transpose(1, 2)
    #     # 对转置前后的张量进行相加
    #     attention = (attention + transposed_tensor)/2
    #
    #     # if not self.training:
    #     #     print("================== softmax")
    #     #     attention_print = attention[0, :, :]
    #     #     for row in attention_print:
    #     #         for element in row:
    #     #             print(element.item(), end=' ')
    #     #         print()
    #     h_prime = torch.matmul(attention, hidden_features)
    #     if self.concat:
    #         h_prime = F.elu(h_prime)
    #     # if not self.training:
    #     #     print(attention.size())
    #     #     print(hidden_features.size())
    #     #     print("================== h_prime")
    #     #     h_prime_print = h_prime[0, :, :]
    #     #     for row in h_prime_print:
    #     #         for element in row:
    #     #             print(element.item(), end=' ')
    #     #         print()
    #     return h_prime, attention