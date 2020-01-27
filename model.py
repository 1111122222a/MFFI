import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from resnet import Backbone


class CNN(nn.Module):
    def __init__(self,num_layers, drop_ratio, mode='ir',resnet_path=None):
        super(CNN,self).__init__()

        if resnet_path:
            self.resnet_layer = Backbone(num_layers=num_layers, drop_ratio=drop_ratio, mode=mode)
            state_dict = torch.load(resnet_path)
            self.resnet_layer.load_state_dict(state_dict)
        else:
            self.resnet_layer = Backbone(num_layers, drop_ratio, mode=mode)
        self.norm_layer       = nn.BatchNorm1d(512)

    def forward(self,data):
        _,fea_maps  = self.resnet_layer(data)
        fea_vectors = fea_maps.view(fea_maps.size(0), fea_maps.size(1), -1)
        feas        = torch.mean(fea_vectors, dim=2)
        feas        = self.norm_layer(feas)
        return fea_maps,feas

class AttributeNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size,output_size):
        super(AttributeNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_size,output_size)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.norm_layer_3 = nn.BatchNorm1d(output_size)
    def forward(self,x):
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        x2 = self.norm_layer_3(x2)
        return x2


class DCM_Logit():
    r"""Implement of DCM_Logit :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, d=30.0, m=0.50, easy_margin=False):
        self.d = d
        self.m = m
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input,attribute_output, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(attribute_output))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.d

        return output


class SAM(nn.Module):
    def __init__(self,input_size,output_size,n_att):
        super(SAM, self).__init__()
        self.fc1        = nn.Linear(input_size, output_size)
        self.bn4        = nn.BatchNorm1d(512)
        self.n_att      = n_att

    def forward(self, x):
        '''

        :param x: input feature maps Batch * C * H * W
               weight: attribute classfier weights  N*C
        :return:
               out: attention value + input value
        '''
        Batch, C, H, W = x.size()
        x_project      = x.view(x.size(0),x.size(1),-1)  #Batch*C*N  N=H*W
        x_project      = torch.mean(x_project,dim=2)     #Batch*C
        predict_att    = F.sigmoid(self.fc1(x_project))  #Batch*40

        _,argsort_pred = torch.sort(-predict_att,dim=1)
        n_att          = self.n_att
        argsort_pred   = argsort_pred[:,:n_att]
        argsort_pred   = argsort_pred.reshape(Batch*n_att)
        weight         = self.fc1.weight  # 40*512
        select_weight = weight[argsort_pred,:]
        select_weight = select_weight.view(Batch,n_att,-1)   #Batch * 5 * C

        # x Batch * C * N
        x_re          = x.view(x.size(0),x.size(1),-1) # x Batch * C * N
        cam           = torch.bmm(select_weight,x_re)  #Batch * 5 * N
        cam_min, _ = torch.min(cam, dim=2)
        cam_min = cam_min.unsqueeze(2).repeat(1, 1, H)
        cam_min = cam_min.unsqueeze(3).repeat(1, 1, 1, W)
        cam_max, _ = torch.max(cam, dim=2)

        cam_max = cam_max.unsqueeze(2).repeat(1, 1, H)
        cam_max = cam_max.unsqueeze(3).repeat(1, 1, 1, W)
        cam = cam.view(cam.size(0), cam.size(1), H, W)  # Batch * 5 * H * W
        cam = torch.div(cam - cam_min, cam_max)

        Spatial_Map,_ = torch.max(cam,dim=1) #Batch * H * W
        Spatial_Map   = Spatial_Map.unsqueeze(1) # Batch*1*H*W

        Spatial_Map_ex = Spatial_Map.repeat(1, C, 1, 1)  # Batch * C * H * W

        add_maps     = torch.mul(x, Spatial_Map_ex)  # BATCH*C*H*W
        fea_maps     = x + add_maps  # BATCH * C * H *W
        out_feas     = fea_maps.view(fea_maps.size(0), fea_maps.size(1), -1 )# BATCH * C * [H *W]
        out_feas     = torch.mean(out_feas, dim=2)
        out_feas     = self.bn4(out_feas)
        return predict_att,out_feas,fea_maps