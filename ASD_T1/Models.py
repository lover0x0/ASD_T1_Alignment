import torch
import torch.nn as nn
# from modules import *
from modules import GCN, InputTransition, DownTransition, MAHGCN
import torch.nn.functional as F
import GATs

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCNEncoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.gcn1 = GCN(in_dim,in_dim,0.3)
        self.gcn2 = GCN(in_dim,out_dim,0.3)

    def forward(self,g,h):
        out1 = self.gcn1(g, h)
        out1 = self.gcn2(g, out1)
        # out1 = self.gcn2(g, h)

        return out1

class GCNNET(nn.Module):
    def __init__(self,num_class=2):
        super(GCNNET, self).__init__()
        self.GCNEncoder = GCNEncoder(500,1)
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(500,256)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = torch.nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, num_class)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self,g):
        batch_size = g.shape[0]
        h = torch.zeros(g.size())
        for s in range(g.shape[0]):
            h[s,:,:] = torch.eye(500)
        h = h.cuda()
        g = torch.tensor(g, dtype=torch.float32).cuda()
        out = torch.zeros(batch_size, 500)
        for s in range(batch_size):
            out1 = self.GCNEncoder.forward(g[s, :, :],h[s, :, :])
            out1 = out1.cuda()
            out[s, :] = torch.squeeze(out1)
        out = out.cuda()
        out = self.drop(out)
        out = self.fc1(out)
        out = self.bn1.cuda()(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2.cuda()(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.bn3.cuda()(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.softmax(out)

        return out

class CNNEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_tr = InputTransition(in_channels, 16)
        self.down_32 = DownTransition(16, 1)
        self.down_64 = DownTransition(32, 1)
        self.down_128 = DownTransition(64, 2)
        self.down_256 = DownTransition(128, 2)
        self.down_512 = DownTransition(256, 2)

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.max_pool = nn.AdaptiveMaxPool3d((1, 1, 1))

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_32(out16)
        out64 = self.down_64(out32)
        out128 = self.down_128(out64)
        out256 = self.down_256(out128)
        out512 = self.down_512(out256)

        out_avg = self.avg_pool(out512)
        out_max = self.max_pool(out512)
        out = out_avg + out_max

        return out


class ClsNet(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.CNNEncoder = CNNEncoder(1)
        self.fc = nn.Sequential(
            nn.Conv3d(512, 128, 1, bias=True),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.1),
            nn.Conv3d(128, 32, 1, bias=True),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.1),
            nn.Conv3d(32, out_channels, 1, bias=True),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.CNNEncoder(x)
        out = self.fc(out)
        # out = self.softmax(out)

        return out


class FusionNet(nn.Module):
    def __init__(self,out_channels):
        super().__init__()
        self.GCNEncoder = GCNEncoder(500, 1)
        self.drop = nn.Dropout(0.3)
        self.CNNEncoder = CNNEncoder(1)
        self.fc1 = nn.Sequential(
            nn.Linear(500, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.fc_align = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.fc_fusion = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, out_channels)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, g, x):
        batch_size = g.shape[0]
        h = torch.zeros(g.size())
        for s in range(g.shape[0]):
            h[s, :, :] = torch.eye(500)
        h = h.cuda()
        g = torch.tensor(g, dtype=torch.float32).cuda()
        out_fmri = torch.zeros(batch_size, 500)
        for s in range(batch_size):
            out1 = self.GCNEncoder.forward(g[s, :, :], h[s, :, :])
            out1 = out1.cuda()
            out_fmri[s, :] = torch.squeeze(out1)
        out_fmri = self.drop(out_fmri).cuda()
        out_T1 = self.CNNEncoder(x)
        out_T1 = torch.squeeze(out_T1)
        out_T1 = self.drop(out_T1).cuda()
        out_fmri = self.fc1(out_fmri)
        out_T1 = self.fc2(out_T1)
        fusion_input = torch.cat((out_fmri, out_T1), dim=1)
        fusion_input_align = self.fc_align(fusion_input)
        out = self.fc_fusion(fusion_input_align)
        out = self.softmax(out)

        return out


class sNet(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, dim // 8, kernel_size=(3, 3, 3)),
            nn.GroupNorm(8, dim // 8),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(dim // 8, dim // 4, kernel_size=(3, 3, 3)),
            nn.GroupNorm(8, dim // 4),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(dim // 4, dim // 2, kernel_size=(3, 3, 3)),
            nn.GroupNorm(8, dim // 2),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(dim // 2, dim, kernel_size=(3, 3, 3)),
            nn.GroupNorm(8, dim),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(dim, dim // 4, kernel_size=(1, 1, 1)),
            nn.GroupNorm(8, dim // 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 64))
        )
        self.linear_classification = nn.Sequential(
            nn.Linear(8192, 2),  # 减少全连接层的输入节点数
        )

    def forward(self, mri):
        conv1_out = self.conv1(mri)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        conv5_out = conv5_out.reshape(conv5_out.shape[0], conv5_out.shape[1], conv5_out.shape[-1])
        conv_5_out_squeeze = conv5_out.reshape(conv5_out.shape[0], -1)
        output = self.linear_classification(conv_5_out_squeeze)
        return output



class MAHGCNNET(nn.Module):
    def __init__(self,ROInum,layer, num_class=2):
        super(MAHGCNNET, self).__init__()
        self.ROInum=ROInum
        self.layer = layer
        self.paranum=0
        for i in range(100,ROInum+100,100):
            self.paranum=self.paranum+i
        #self.mMAHGCNs = nn.ModuleList()
        #for t in range(tsize):
            #self.mMAHGCNs.append(MAHGCN.MultiresolutionMAHGCN(nn.ReLU(),0.3))
        self.MAHGCN = MAHGCN(nn.ReLU(),0.3,self.layer)
        #self.gcrn = GLSTM_multi.ConvLSTM(ROInum, 1)

        self.bn1 = torch.nn.BatchNorm1d(self.paranum)
        self.fl1 = nn.Linear(self.paranum,512)
        self.bn2 = torch.nn.BatchNorm1d(512)
        self.fl2 = nn.Linear(512,64)
        self.bn3 = torch.nn.BatchNorm1d(64)
        self.fl3 = nn.Linear(64,num_class)


        #self.dropout = nn.Dropout(0.6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, g1, g2, g3, g4, g5):
        batch_size = g1.shape[0]
        ROInum = self.ROInum

        fea = torch.zeros(batch_size, ROInum, ROInum)
        for s in range(batch_size):
            fea[s,:,:] = torch.eye(ROInum)
        fea = fea.cuda()
        g1 = g1.cuda()
        g2 = g2.cuda()
        g3 = g3.cuda()
        g4 = g4.cuda()
        g5 = g5.cuda()
        out = torch.zeros(batch_size, self.paranum)

        for s in range(batch_size):
            temp = self.MAHGCN(g1[s, :, :], g2[s, :, :], g3[s, :, :],g4[s, :, :], g5[s, :, :], fea[s, :, :])
            temp.cuda()
            out[s, :] = torch.squeeze(temp)
        out = out.cuda()

        out = self.bn1.cuda()(out)
        out = F.relu(out)

        out = self.fl1(out)
        out = self.bn2.cuda()(out)
        out_feature = F.relu(out)
        out = self.fl2(out_feature)
        out = self.bn3.cuda()(out)
        out = self.fl3(out)
        # out = self.softmax(out)

        return out_feature, out


class GATNET(nn.Module):
    def __init__(self,  region_size, in_features_num, hidden_num1, hidden_num2, out_features_num):
        super(GATNET, self).__init__()
        self.region_size = region_size
        self.in_features_num = in_features_num
        self.hidden_num1 = hidden_num1
        self.hidden_num2 = hidden_num2
        self.out_features_num = out_features_num
        self.GAT = GATs.whole_network(self.region_size, self.in_features_num, self.hidden_num1, self.hidden_num2, self.out_features_num, 0, 0.3)
        self.bn1 = nn.BatchNorm1d(500)
        self.fc1 = nn.Linear(500,512)
        self.bn2 = torch.nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 64)
        self.bn3 = torch.nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = x.cuda()
        out, a = self.GAT(x,0)
        out = torch.squeeze(out)
        out = out.cuda()
        out = self.bn1.cuda()(out)
        out = F.relu(out)
        out = self.fc1(out)
        out = self.bn2.cuda()(out)
        out_feature = F.relu(out)
        out = self.fc2(out_feature)
        out = self.bn3.cuda()(out)
        out = F.relu(out)
        out = self.fc3(out)

        return out_feature, out















