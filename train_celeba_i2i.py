import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data_reader import CelebA
from torch.autograd import Variable
import torch.nn as nn
from tqdm import tqdm
from model import CNN,I2I_DCM_Logit,SAM,AttributeNetwork
import numpy as np
import argparse
from datetime import datetime
import torch.nn.functional as F
import scipy.io as scio
from sklearn.preprocessing import normalize
import math

def compute_topx_acc(cnn_model,sam,probe_loader,gallery_loader):

    test_binary_atts_list  = []
    test_predict_atts_list = []

    probeset_feas          = []
    galleryset_feas        = []
    probeset_labels        = []
    galleryset_labels      = []

    cnn_model.eval()
    sam.eval()
    # get model output
    with torch.no_grad():

        for batch_imgs,batch_atts,batch_binary_atts,batch_labels in gallery_loader:
            batch_labels = batch_labels.numpy()
            batch_imgs           = Variable(batch_imgs).cuda().float()
            fea_maps, fea_vecs   = cnn_model(batch_imgs)
            pred_atts,cnn_output,spatial_fea_maps  = sam(fea_maps)  # Batch*C*H*W
            pred_atts            = pred_atts.cpu().detach().numpy()   # Batch*40*2
            test_predict_atts_list.append(pred_atts)
            batch_binary_atts   = batch_binary_atts.cpu().detach().numpy()
            test_binary_atts_list.append(batch_binary_atts)
            cnn_output          = F.normalize(cnn_output, p=2, dim=1)
            cnn_output_T        = torch.transpose(cnn_output, 1, 0)  # 512 * N_samples
            cnn_output_T        = cnn_output_T.cpu().detach().numpy()
            galleryset_feas.append(cnn_output_T)
            galleryset_labels.append(batch_labels)

        for batch_imgs, batch_atts, batch_binary_atts, batch_labels in probe_loader:
            batch_labels       = batch_labels.numpy()
            batch_imgs         = Variable(batch_imgs).cuda().float()
            fea_maps, fea_vecs = cnn_model(batch_imgs)
            pred_atts, cnn_output, spatial_fea_maps = sam(fea_maps)  # Batch*C*H*W
            pred_atts = pred_atts.cpu().detach().numpy()  # Batch*40*2
            test_predict_atts_list.append(pred_atts)
            batch_binary_atts = batch_binary_atts.cpu().detach().numpy()
            test_binary_atts_list.append(batch_binary_atts)
            cnn_output   = F.normalize(cnn_output, p=2, dim=1)
            cnn_output   = cnn_output.cpu().detach().numpy()
            probeset_feas.append(cnn_output)
            probeset_labels.append(batch_labels)

        print('finish feature')

    galleryset_labels  = np.hstack(galleryset_labels)  # N_samples
    probeset_labels    = np.hstack(probeset_labels)  # N_samples
    galleryset_feas    = np.hstack(galleryset_feas)  # N_samples
    probeset_feas      = np.vstack(probeset_feas)  # N_samples

    i2i_scores      = np.matmul(probeset_feas,galleryset_feas)
    label_idx       = np.argsort(-i2i_scores,axis=1)[:,:10]
    unique_labels   = galleryset_labels
    pred_labels     = np.take(unique_labels, label_idx)
    i2i_average_acc = get_i2i_topx_acc(y_true=probeset_labels, y_pred=pred_labels)

    return i2i_average_acc


def get_i2i_topx_acc(y_true,y_pred):
    '''
    We use this function to compute top-x acc.
    :param y_true:
    :param y_pred:
    :return: average acc, per class acc and the label index of per class acc
    '''
    top_1_per_acc       = []
    top_5_per_acc = []
    top_10_per_acc = []
    y_true_ex     = np.expand_dims(y_true,1)
    y_true_ex     = np.tile(y_true_ex,[1,10])
    same_or_not   = y_true_ex == y_pred
    top_1_same_or_not = np.sum(same_or_not[:,:1],axis=1)
    top_5_same_or_not = np.sum(same_or_not[:,:5],axis=1)
    top_10_same_or_not = np.sum(same_or_not[:,:10],axis=1)
    unique_labels = np.unique(y_true)

    for label in unique_labels:
        true_samples_idx = np.where(y_true == label)[0]
        top_1_pred_same  = np.where(top_1_same_or_not[true_samples_idx] >= 1)[0]
        top_5_pred_same  = np.where(top_5_same_or_not[true_samples_idx] >= 1)[0]
        top_10_pred_same = np.where(top_10_same_or_not[true_samples_idx] >= 1)[0]
        n_samples        = true_samples_idx.shape[0]

        top_1_n_true  = top_1_pred_same.shape[0]
        top_5_n_true  = top_5_pred_same.shape[0]
        top_10_n_true = top_10_pred_same.shape[0]

        top_1_acc  = top_1_n_true / n_samples
        top_5_acc  = top_5_n_true / n_samples
        top_10_acc = top_10_n_true / n_samples

        top_1_per_acc.append(top_1_acc)
        top_5_per_acc.append(top_5_acc)
        top_10_per_acc.append(top_10_acc)

    top_1_per_acc     = np.array(top_1_per_acc)
    top_1_average_acc = np.mean(top_1_per_acc)
    top_5_per_acc     = np.array(top_5_per_acc)

    top_5_average_acc  = np.mean(top_5_per_acc)
    top_10_per_acc     = np.array(top_10_per_acc)
    top_10_average_acc = np.mean(top_10_per_acc)

    average_acc = np.array([top_1_average_acc, top_5_average_acc, top_10_average_acc])

    return average_acc

def main(parser):

    # prepare parameters
    opt           = parser.parse_args()
    BATCHSIZE     = opt.batch_size
    EPOCHS        = opt.epochs
    resnet_path   = opt.resnet_path
    N_ATT         = opt.n_att
    ##load data
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(opt.SIZE_TRAIN),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    transform_test = transforms.Compose([
        transforms.Resize(opt.SIZE_TEST),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    trainset         = CelebA(opt, transform=transform_train)
    train_loader     = DataLoader(dataset=trainset, batch_size=BATCHSIZE, num_workers=8, pin_memory=True, )

    probeset         = CelebA(opt, transform=transform_test, setting='test', tag='i2i',pg_tag='probe')
    probe_loader     = DataLoader(dataset=probeset, batch_size=BATCHSIZE, num_workers=8, pin_memory=True, )
    galleryset       = CelebA(opt, transform=transform_test, setting='test', tag='i2i',pg_tag='gallery')
    gallery_loader   = DataLoader(dataset=galleryset, batch_size=BATCHSIZE, num_workers=8, pin_memory=True, )

    n_train = len(trainset)
    n_person = trainset.train_unique_labels.shape[0]
    # build model
    print('init networks')
    cnn_model         = CNN(num_layers=opt.net_depth, drop_ratio=opt.drop_ratio, mode=opt.net_mode, resnet_path=resnet_path)
    sam               = SAM(input_size=512, output_size=40,n_att=N_ATT)
    i2i_dcm_logit     = I2I_DCM_Logit(in_features=512,out_features=n_person,d=opt.d,m=opt.margin)
    cnn_model.cuda()
    i2i_dcm_logit.cuda()
    sam.cuda()

    # multi-gpu setting
    if torch.cuda.device_count() > 1:
        print("Let's use",torch.cuda.device_count(),"GPUs to train our model !!!")
        cnn_model               = nn.DataParallel(cnn_model)
        i2i_dcm_logit           = nn.DataParallel(i2i_dcm_logit)
        sam                     = nn.DataParallel(sam)
    # model optimizers
    cnn_model_optim               = torch.optim.Adam(cnn_model.parameters(), lr=opt.lr_cnn,weight_decay=opt.weight_decay)
    i2i_dcm_logit_optim           = torch.optim.Adam(i2i_dcm_logit.parameters(), lr=opt.lr_dcm,weight_decay=opt.weight_decay)
    sam_optim                     = torch.optim.Adam(sam.parameters(), lr=opt.lr_sam,weight_decay=opt.weight_decay )

    print("training...")
    m = 0
    loss_list                           = []
    dcm_loss_list                       = []
    cea_loss_list                       = []
    test_i2i_average_acc_list           = []
    iter_list                           = []
    for epoch in tqdm(range(EPOCHS)):
        cnn_model.train()
        sam.train()
        i2i_dcm_logit.train()
        i = 0
        epoch_start_time = datetime.now()
        for batch_imgs,batch_atts,batch_binary_atts,batch_labels in train_loader:
            # prepare data
            batch_imgs = Variable(batch_imgs).cuda().float()  # 32*1024
            batch_binary_atts = Variable(batch_binary_atts).cuda().float()  # 32*1024
            unique_train_labels = trainset.train_unique_labels
            re_batch_labels = []
            for label in batch_labels.numpy():
                idx = np.where(unique_train_labels == label)[0][0]
                re_batch_labels.append(idx)
            re_batch_labels = torch.LongTensor(re_batch_labels).cuda()

            # get model output
            fea_maps,feas_vec                     = cnn_model(batch_imgs)
            pred_atts,batch_feas,spatial_fea_maps = sam(fea_maps)
            dcm_logit                             = i2i_dcm_logit(batch_feas,label=re_batch_labels)

            # compute loss

            target      = Variable(re_batch_labels).cuda()
            CEA         = nn.BCELoss().cuda()
            DCM         = nn.CrossEntropyLoss().cuda()
            cea_loss    = CEA(input=pred_atts,target=batch_binary_atts) * 15
            dcm_loss    = DCM(input=dcm_logit, target=target)
            loss        = cea_loss+dcm_loss
            # update
            cnn_model.zero_grad()
            i2i_dcm_logit.zero_grad()
            sam.zero_grad()

            loss.backward()
            cnn_model_optim.step()
            i2i_dcm_logit_optim.step()
            sam_optim.step()
            if m == 0:
                iter_start_time = datetime.now()
                loss_list.append(loss.cpu().detach().numpy())
                dcm_loss_list.append(dcm_loss.cpu().detach().numpy())
                cea_loss_list.append(cea_loss.cpu().detach().numpy())
                iter_list.append(m)
                print('loss:{:.5},dcm_loss:{:.5},cea_loss:{:.5}'.format(loss.cpu().detach().numpy(),dcm_loss.cpu().detach().numpy(),
                                                                                                            cea_loss.cpu().detach().numpy()))
            if (m!=0) and (m % 500 == 0):
                iter_end_time = datetime.now()
                iter_spent_time = (iter_end_time - iter_start_time).seconds / 60
                iter_start_time = iter_end_time
                loss_list.append(loss.cpu().detach().numpy())
                dcm_loss_list.append(dcm_loss.cpu().detach().numpy())
                cea_loss_list.append(cea_loss.cpu().detach().numpy())
                iter_list.append(m)
                print('Have finished {} samples and remain {} samples, spend {:.2f} minutes'.format(BATCHSIZE*500,n_train-BATCHSIZE*i,iter_spent_time))

                print(
                    'loss:{:.5},dcm_loss:{:.5},cea_loss:{:.5}'.format(loss.cpu().detach().numpy(),
                                                                                            dcm_loss.cpu().detach().numpy(),
                                                                                            cea_loss.cpu().detach().numpy(),
                                                                                            ))
            m+=1
            i+=1

        epoch_end_time = datetime.now()
        epoch_spend_time = (epoch_end_time-epoch_start_time).seconds/60
        print('Spend {:.2f} minutes to complete one epoch '.format(epoch_spend_time))
        print('#############')
        print('Testing ...')
        test_i2i_average_acc = compute_topx_acc(cnn_model=cnn_model,
                                                sam=sam, probe_loader=probe_loader,
                                                gallery_loader=gallery_loader)

        test_i2i_average_acc_list.append(test_i2i_average_acc)


        for q, top_number in enumerate(opt.top_x):

            print('Testset Imgae to Image top-{} average class accuracy:{:.2f}'.format(top_number,
                                                                                 test_i2i_average_acc[q]))



        if not os.path.exists(opt.basemodel_dir):
            os.makedirs(opt.basemodel_dir)
        basemodel_dir = opt.basemodel_dir + '/'+'Top_'+str(opt.n_att)+'_spatial_model'
        if not os.path.exists(basemodel_dir):
            os.makedirs(basemodel_dir)
        if not os.path.exists(basemodel_dir +'/'+opt.cnn_model_dir):
            os.makedirs(basemodel_dir+'/'+opt.cnn_model_dir)
        if not os.path.exists(basemodel_dir + '/' + opt.i2i_dcm_logit_dir):
            os.makedirs(basemodel_dir + '/' + opt.i2i_dcm_logit_dir)
        if not os.path.exists(basemodel_dir + '/' + opt.sam_model_dir):
            os.makedirs(basemodel_dir + '/' + opt.sam_model_dir)

        print('#############')
        print('save networks')
        cnn_model_name = opt.cnn_model_name + '_epoch_' + str(epoch)+'.pkl'
        i2i_dcm_logit_name = opt.i2i_dcm_logit_name + '_epoch_' + str(epoch)+'.pkl'
        sam_model_name = opt.sam_model_name + '_epoch_' + str(epoch)+'.pkl'
        if torch.cuda.device_count() > 1:
            torch.save(cnn_model.module.state_dict(),basemodel_dir + '/' + opt.cnn_model_dir + '/' + cnn_model_name)
            torch.save(i2i_dcm_logit.module.state_dict(),basemodel_dir + '/' + opt.i2i_dcm_logit_dir + '/' + i2i_dcm_logit_name)
            torch.save(sam.module.state_dict(), basemodel_dir+ '/' + opt.sam_model_dir + '/' + sam_model_name)
        else:
            torch.save(cnn_model.state_dict(), basemodel_dir + '/' + opt.cnn_model_dir + '/' + cnn_model_name)
            torch.save(i2i_dcm_logit.state_dict(),basemodel_dir + '/' + opt.i2i_dcm_logit_dir + '/' + i2i_dcm_logit_name)
            torch.save(sam.state_dict(), basemodel_dir+ '/' + opt.sam_model_dir + '/' + sam_model_name)

        loss_arr     = np.array(loss_list)
        dcm_loss_arr  = np.array(dcm_loss_list)
        cea_loss_arr = np.array(cea_loss_list)
        iter_arr     = np.array(iter_list)
        test_i2i_average_acc_arr      = np.array(test_i2i_average_acc_list)

        results_name = basemodel_dir + '/'+'Top_' + str(N_ATT)+'_spatial_results.mat'
        scio.savemat(results_name, {'top-number': np.array(opt.top_x),
                                                                  'loss': loss_arr,'dcm_loss': dcm_loss_arr,'cea_loss': cea_loss_arr,
                                                                  'iter': iter_arr,
                                                                  'test_i2i_average_acc': test_i2i_average_acc_arr,
                                                                  'drop_ratio': opt.drop_ratio,
                                                                  'weight_decay': opt.weight_decay,
                                                                  'margin': opt.margin,
                                                                  'd': opt.d,
                                                                  'lr_cnn': opt.lr_cnn,
                                                                  'lr_attribute': opt.lr_attribute,
                                                                  'lr_sam': opt.lr_sam,
                                                                  'batch_size': opt.batch_size,
                                                                  'attribute_hidden_size': np.array([256]),
                                                                  })
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='./data/CelebA', help='path to dataset')
    parser.add_argument('--img_dir', default='img_align_celeba', help='path to image')
    parser.add_argument('--Anno_dir', default='Anno', help='path to Anno')
    parser.add_argument('--split_dir', default='Eval', help='path to spilt')
    parser.add_argument('--pg_idx_dir', default='pg_idx', help='path to spilt')
    parser.add_argument('--att_filename', default='list_attr_celeba.txt', help='name of attribute file')
    parser.add_argument('--label_filename', default='identity_CelebA.txt', help='name of identity name')
    parser.add_argument('--split_filename', default='list_eval_partition.txt', help='name of split file')
    parser.add_argument('--tag', default=None, help='whether build probe or gallery set or not')
    parser.add_argument('--pg_tag', default='probe', help='probe or gallery set')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--lr_sam', type=int, default=0.00005, help='learning rate to trian sam')
    parser.add_argument('--lr_cnn', type=float, default=0.00005, help='learning rate to train cnn')
    parser.add_argument('--lr_dcm', type=float, default=0.00005, help='learning rate to train dcm')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='learning rate to train model')
    parser.add_argument('--SIZE_TRAIN', type=tuple, default=(112, 112), help='input train size')
    parser.add_argument('--SIZE_TEST', type=tuple, default=(112, 112), help='input test size')
    parser.add_argument('--net_depth', type=int, default=50, help='net depth')
    parser.add_argument('--drop_ratio', type=float, default=0.6, help='drop ratio')
    parser.add_argument('--net_mode', type=str, default='ir_se', help='net mode')
    parser.add_argument('--epochs', type=int, default=20, help='epoch of training model')
    parser.add_argument('--margin', type=float, default=0.1, help='margin of training model')
    parser.add_argument('--d', type=float, default=32, help='scale of training model')
    parser.add_argument('--n_att', type=float, default=10, help='top-D att')
    parser.add_argument('--resnet_path', type=str, default='./results/resnet/model_ir_se50.pth',help='pretrain resnet model')
    parser.add_argument('--basemodel_dir', type=str, default='./results/basemodel/spatial_model_2',help='root dir of saving model')
    parser.add_argument('--cnn_model_dir',    type=str, default='cnn_model', help='root dir of cnn model')
    parser.add_argument('--i2i_dcm_logit_model_dir', type=str, default='i2i_dcm_model', help='root dir of i2i dcm logit model')
    parser.add_argument('--sam_model_dir', type=str, default='sam_model', help='root dir of spatial attention model')
    parser.add_argument('--cnn_model_name',               type=str,   default='cnn_model',               help='cnn model name')
    parser.add_argument('--attribute_model_name',         type=str,   default='attribute_model',         help='attribute model name')
    parser.add_argument('--sam_model_name',               type=str,    default='sam', help='spatial attention model name')
    parser.add_argument('--results_name',         type=str,   default='results_1.mat',help='results matfile name ')
    parser.add_argument('--top_x', type=list, default=[1, 5, 10], help='top_x to compute accuracy')

    main(parser)