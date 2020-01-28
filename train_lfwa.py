import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data_reader import LFWA
from torch.autograd import Variable
import torch.nn as nn
from tqdm import tqdm
from model import CNN,DCM_Logit,SAM,AttributeNetwork
import numpy as np
import argparse
from datetime import datetime
import torch.nn.functional as F
import scipy.io as scio
from sklearn.preprocessing import normalize
import math

def compute_topx_acc(cnn_model,attribute_model,sam,testset,test_loader):

    test_labels_list       = []
    i2a_scores_list   = []
    test_binary_atts_list  = []
    test_predict_atts_list = []
    a2i_scores_list  = []
    cnn_model.eval()
    attribute_model.eval()
    sam.eval()
    # get model output
    with torch.no_grad():
        # First we get prototypes
        test_class_embeddings      = torch.from_numpy(testset.test_class_embeddings)
        test_class_embeddings      = Variable(test_class_embeddings).cuda().float()
        attribute_network_output   = attribute_model(test_class_embeddings)
        attribute_network_output   = F.normalize(attribute_network_output, p=2, dim=1)  # N_identity * 512
        attribute_network_output_T = torch.transpose(attribute_network_output, 1, 0)  # 512 * N_identity

        for batch_imgs,batch_atts,batch_binary_atts,batch_labels in test_loader:
            # sceond  we get a batch feature vectors
            batch_labels         = batch_labels.numpy()
            test_labels_list.append(batch_labels)
            batch_imgs           = Variable(batch_imgs).cuda().float()
            fea_maps, fea_vecs   = cnn_model(batch_imgs)
            pred_atts,cnn_output,spatial_fea_maps  = sam(fea_maps)  # Batch*C*H*W
            pred_atts            = pred_atts.cpu().detach().numpy()   # Batch*40*2
            test_predict_atts_list.append(pred_atts)
            batch_binary_atts    = batch_binary_atts.cpu().detach().numpy()
            test_binary_atts_list.append(batch_binary_atts)
            # third, we compute the similarity
            cnn_output           = F.normalize(cnn_output, p=2, dim=1)
            cnn_output_T         = torch.transpose(cnn_output, 1, 0)  # 512 * N_samples
            i2a_scores      = torch.matmul(cnn_output, attribute_network_output_T)  # N_samples * N_identiy
            a2i_scores     = torch.matmul(attribute_network_output, cnn_output_T)  # N_identity * N_samples
            i2a_scores      = i2a_scores.cpu().detach().numpy()
            a2i_scores     = a2i_scores.cpu().detach().numpy()
            i2a_scores_list.append(i2a_scores)
            a2i_scores_list.append(a2i_scores)
        print('finish feature')
    test_labels      = np.hstack(test_labels_list)  # N_samples
    i2a_scores  = np.vstack(i2a_scores_list)  # N_samples * N_identiy
    a2i_scores = np.hstack(a2i_scores_list)

    # we compute classification acc、a2i acc and attribute i2a acc respectively
    # First we compute classification acc
    test_binary_atts = np.vstack(test_binary_atts_list)
    test_predict_atts = np.vstack(test_predict_atts_list)
    att_classify_average_acc = get_att_classify_acc(pred_atts=test_predict_atts,
                                                                          true_atts=test_binary_atts)
    label_idx     = np.argsort(-i2a_scores, axis=1)[:, :10]
    unique_labels = testset.test_unique_labels
    pred_labels   = np.take(unique_labels, label_idx)
    i2a_average_acc = get_i2a_topx_acc(y_true=test_labels, y_pred=pred_labels)
    # Second we compute a2i acc

    label_idx     = np.argsort(-a2i_scores, axis=1)[:, :10]
    unique_labels = test_labels
    pred_labels   = np.take(unique_labels, label_idx)
    a2i_average_acc, = get_a2i_topx_acc(y_true=testset.test_unique_labels,y_pred=pred_labels)

    return i2a_average_acc, a2i_average_acc,att_classify_average_acc


def get_a2i_topx_acc(y_true,y_pred):
    '''
    We compute a2i acc
    :param y_true:
    :param y_pred:
    :return:
    '''

    n_person    = y_true.shape[0]
    y_true      = np.expand_dims(y_true, 1)
    y_true_ex   = np.tile(y_true, [1, 10])
    same_or_not = y_true_ex == y_pred

    top_1_same_or_not  = np.sum(same_or_not[:, :1], axis=1)
    top_5_same_or_not  = np.sum(same_or_not[:, :5], axis=1)
    top_10_same_or_not = np.sum(same_or_not[:, :10], axis=1)

    top_1_pred_same  = np.where(top_1_same_or_not >= 1)[0]
    top_5_pred_same  = np.where(top_5_same_or_not >= 1)[0]
    top_10_pred_same = np.where(top_10_same_or_not >= 1)[0]

    top_1_per_acc  = np.zeros(n_person, dtype=int)
    top_5_per_acc  = np.zeros(n_person, dtype=int)
    top_10_per_acc = np.zeros(n_person, dtype=int)

    top_1_per_acc[top_1_pred_same]   = 1
    top_5_per_acc[top_5_pred_same]   = 1
    top_10_per_acc[top_10_pred_same] = 1

    top_1_average_acc  = np.mean(top_1_per_acc)
    top_5_average_acc  = np.mean(top_5_per_acc)
    top_10_average_acc = np.mean(top_10_per_acc)

    average_acc       = np.array([top_1_average_acc,top_5_average_acc,top_10_average_acc])

    return average_acc

def get_i2a_topx_acc(y_true,y_pred):
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

def get_att_classify_acc(pred_atts,true_atts):

    idx = np.where(pred_atts >= 0.5)
    pred_atts = np.zeros(pred_atts.shape)
    pred_atts[idx[0], idx[1]] = 1

    same_or_not = (true_atts == pred_atts).astype(int)
    n_samples = same_or_not.shape[0]
    n_equal = np.sum(same_or_not, axis=0)
    per_acc = n_equal / n_samples
    average = np.average(per_acc)


    return average

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
    trainset         = LFWA(opt, transform=transform_train)
    train_loader     = DataLoader(dataset=trainset, batch_size=BATCHSIZE, num_workers=8, pin_memory=True, )
    testset          = LFWA(opt, transform=transform_test)
    test_loader      = DataLoader(dataset=testset, batch_size=BATCHSIZE, num_workers=8, pin_memory=True, )
    n_train = len(trainset)

    # build model
    print('init networks')
    cnn_model         = CNN(num_layers=opt.net_depth, drop_ratio=opt.drop_ratio, mode=opt.net_mode, resnet_path=resnet_path)
    attribute_model   = AttributeNetwork(40, 256, 512)
    sam               = SAM(input_size=512, output_size=40,n_att=N_ATT)
    arcmargin_compute = DCM_Logit(d=opt.d,m=opt.margin)
    cnn_model.cuda()
    attribute_model.cuda()
    sam.cuda()

    # multi-gpu setting
    if torch.cuda.device_count() > 1:
        print("Let's use",torch.cuda.device_count(),"GPUs to train our model !!!")
        cnn_model               = nn.DataParallel(cnn_model)
        attribute_model         = nn.DataParallel(attribute_model)
        sam                     = nn.DataParallel(sam)
    # model optimizers
    cnn_model_optim               = torch.optim.Adam(cnn_model.parameters(), lr=opt.lr_cnn,weight_decay=opt.weight_decay)
    attribute_model_optim         = torch.optim.Adam(attribute_model.parameters(), lr=opt.lr_attribute,weight_decay=opt.weight_decay)
    sam_optim                     = torch.optim.Adam(sam.parameters(), lr=opt.lr_sam,weight_decay=opt.weight_decay )

    print("training...")
    m = 0
    loss_list                           = []
    dcm_loss_list                        = []
    cea_loss_list                       = []
    test_i2a_average_acc_list      = []
    test_a2i_average_acc_list     = []
    test_att_classify_average_acc_list  = []
    iter_list    = []
    for epoch in tqdm(range(EPOCHS)):
        cnn_model.train()
        sam.train()
        attribute_model.train()
        i = 0
        epoch_start_time = datetime.now()
        for batch_imgs,batch_atts,batch_binary_atts,batch_labels in train_loader:
            # prepare data
            unique_labels     = torch.unique(batch_labels).numpy()
            unique_batch_atts = torch.from_numpy(trainset.class_embeddings[unique_labels,:])
            batch_imgs        = Variable(batch_imgs).cuda().float()  # 32*1024
            batch_binary_atts = Variable(batch_binary_atts).cuda().float()  # 32*1024
            unique_batch_atts = Variable(unique_batch_atts).cuda().float()
            # get model output
            fea_maps,feas_vec                     = cnn_model(batch_imgs)
            unique_attribute_network_out          = attribute_model(unique_batch_atts)
            pred_atts,batch_feas,spatial_fea_maps = sam(fea_maps)
            # compute loss
            re_batch_labels = []
            for label in batch_labels.numpy():
                idx = np.where(unique_labels == label)[0][0]
                re_batch_labels.append(idx)
            re_batch_labels = torch.LongTensor(re_batch_labels).cuda()

            output      = arcmargin_compute.forward(input=batch_feas, attribute_output=unique_attribute_network_out,label=re_batch_labels)
            target      = Variable(re_batch_labels).cuda()
            CEA         = nn.BCELoss().cuda()
            DCM         = nn.CrossEntropyLoss().cuda()
            cea_loss    = CEA(input=pred_atts,target=batch_binary_atts) * 15
            dcm_loss    = DCM.forward(input=output, target=target)

            loss       = cea_loss+dcm_loss
            # update
            cnn_model.zero_grad()
            attribute_model.zero_grad()
            sam.zero_grad()

            loss.backward()
            cnn_model_optim.step()
            attribute_model_optim.step()
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
        test_i2a_average_acc, test_a2i_average_acc,\
        test_att_classify_average_acc = compute_topx_acc(cnn_model=cnn_model,
                                                                          attribute_model=attribute_model,
                                                                          sam=sam, test_loader=test_loader,
                                                                          testset=testset, )

        test_i2a_average_acc_list.append(test_i2a_average_acc)
        test_a2i_average_acc_list.append(test_a2i_average_acc)
        test_att_classify_average_acc_list.append(test_att_classify_average_acc)


        for q, top_number in enumerate(opt.top_x):

            print('Testset Imgae to Attribute top-{} average class accuracy:{:.2f}'.format(top_number,
                                                                                 test_i2a_average_acc[q]))
        print('\n')
        for q, top_number in enumerate(opt.top_x):
            print('Testset Attribute to Image top-{} average class accuracy:{:.2f}'.format(top_number,
                                                                                  test_a2i_average_acc[q]))
        print('Testset att classify acc is {:.2f}'.format(test_att_classify_average_acc))


        if not os.path.exists(opt.basemodel_dir):
            os.makedirs(opt.basemodel_dir)
        basemodel_dir = opt.basemodel_dir + '/'+'Top_'+str(opt.n_att)+'_spatial_model'
        if not os.path.exists(basemodel_dir):
            os.makedirs(basemodel_dir)
        if not os.path.exists(basemodel_dir+'/'+opt.cnn_model_dir):
            os.makedirs(basemodel_dir+'/'+opt.cnn_model_dir)
        if not os.path.exists(basemodel_dir + '/' + opt.att_model_dir):
            os.makedirs(basemodel_dir + '/' + opt.att_model_dir)
        if not os.path.exists(basemodel_dir + '/' + opt.sam_model_dir):
            os.makedirs(basemodel_dir + '/' + opt.sam_model_dir)

        print('#############')
        print('save networks')
        cnn_model_name = opt.cnn_model_name + '_epoch_' + str(epoch)+'.pkl'
        attribute_model_name = opt.attribute_model_name + '_epoch_' + str(epoch)+'.pkl'
        sam_model_name = opt.sam_model_name + '_epoch_' + str(epoch)+'.pkl'
        if torch.cuda.device_count() > 1:
            torch.save(cnn_model.module.state_dict(),basemodel_dir + '/' + opt.cnn_model_dir + '/' + cnn_model_name)
            torch.save(attribute_model.module.state_dict(),basemodel_dir + '/' + opt.att_model_dir + '/' + attribute_model_name)
            torch.save(sam.module.state_dict(), basemodel_dir+ '/' + opt.sam_model_dir + '/' + sam_model_name)
        else:
            torch.save(cnn_model.state_dict(), basemodel_dir + '/' + opt.cnn_model_dir + '/' + cnn_model_name)
            torch.save(attribute_model.state_dict(),basemodel_dir + '/' + opt.att_model_dir + '/' + attribute_model_name)
            torch.save(sam.state_dict(), basemodel_dir+ '/' + opt.sam_model_dir + '/' + sam_model_name)

        loss_arr     = np.array(loss_list)
        dcm_loss_arr  = np.array(dcm_loss_list)
        cea_loss_arr = np.array(cea_loss_list)
        iter_arr     = np.array(iter_list)
        test_i2a_average_acc_arr      = np.array(test_i2a_average_acc_list)
        test_a2i_average_acc_arr     = np.array(test_a2i_average_acc_list)
        test_att_classify_average_acc_arr  = np.array(test_att_classify_average_acc_list)

        results_name = basemodel_dir + '/'+'Top_' + str(N_ATT)+'_spatial_results.mat'
        scio.savemat(results_name, {'top-number': np.array(opt.top_x),
                                                                  'loss': loss_arr,'dcm_loss': dcm_loss_arr,'cea_loss': cea_loss_arr,
                                                                  'iter': iter_arr,
                                                                  'test_i2a_average_acc': test_i2a_average_acc_arr,
                                                                  'test_a2i_average_acc': test_a2i_average_acc_arr,
                                                                  'test_att_classify_average_acc': test_att_classify_average_acc_arr,
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
    parser.add_argument('--root_dir', default='./data/LFWA', help='path to dataset')
    parser.add_argument('--split_file_dir', default='split', help='path to split file dir')
    parser.add_argument('--img_dir', default='lfw-deepfunneled', help='path to image dir')
    parser.add_argument('--att_file_name', default='lfw_att_40.mat', help='filename of att file')
    parser.add_argument('--split_file_name', default='indices_train_test.mat', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--lr_sam', type=int, default=0.00005, help='the number of choice samples')
    parser.add_argument('--lr_cnn', type=float, default=0.00005, help='learning rate to train model')
    parser.add_argument('--lr_attribute', type=float, default=0.00005, help='learning rate to train model')
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
    parser.add_argument('--resnet_path', type=str, default='./results/resnet/model_ir_se50.pth',
                        help='pretrain resnet model')
    parser.add_argument('--basemodel_dir', type=str, default='./results/basemodel/spatial_model_lfwa',
                        help='root dir of saving model')
    parser.add_argument('--cnn_model_dir', type=str, default='cnn_model', help='root dir of cnn model')
    parser.add_argument('--att_model_dir', type=str, default='att_model', help='root dir of attribute model')
    parser.add_argument('--sam_model_dir', type=str, default='sam_model', help='root dir of spatial attention model')
    parser.add_argument('--cnn_model_name', type=str, default='cnn_model', help='cnn model name')
    parser.add_argument('--attribute_model_name', type=str, default='attribute_model', help='attribute model name')
    parser.add_argument('--sam_model_name', type=str, default='sam', help='spatial attention model name')
    parser.add_argument('--results_name', type=str, default='results_1.mat', help='results matfile name ')
    parser.add_argument('--top_x', type=list, default=[1, 5, 10], help='top_x to compute accuracy')
    main(parser)
