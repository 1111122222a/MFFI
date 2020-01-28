from torch.utils.data import Dataset
import numpy as np
import os
import PIL
from PIL import Image
import scipy.io as scio
from sklearn.preprocessing import normalize


class LFWA(Dataset):
    def __init__(self, opt, transform=None, setting='trainval'):
        '''

        :param filename: txt file contains img names and labels
        :param img_dir: root dir of img
        :param resize_height: if none, not resize
        :param resize_width:
        :param repeat: the repeat number of all sample data, default is 1, when repeat==None, infinite loop

        '''
        self.opt                       = opt
        self.probe_gallery_idx_dir     = opt.root_dir + '/' + opt.pg_idx_dir
        self.split_file_dir            = opt.root_dir + '/' + opt.split_file_dir
        self.img_att_label_list        = self.read_file(setting)
        self.index                     = np.arange(len(self.img_att_label_list))
        self.setting                   = setting
        self.transform                 = transform

    def __getitem__(self, index):
        #index = i % self.len
        idx = self.index[index]
        img_name,att,binary_att,label = self.img_att_label_list[idx]
        img_path           = os.path.join(self.opt.root_dir+'/'+self.opt.img_dir+'/'+img_name)
        img                = self.load_data(img_path)
        if self.transform is not None:
            img = self.transform(img)
        label             = np.array(label,dtype=int)
        if ((index == len(self.img_att_label_list)-1)) and (self.setting != 'test'):
            np.random.shuffle(self.index)
        return img, att,binary_att, label

    def __len__(self):

        data_len = len(self.img_att_label_list)
        return data_len

    def load_data(self, filename):

        '''
        read img and return PIL format
        :param filename:
        :return:img data
        '''

        if not os.path.exists(filename):
            print('Warning:no file:{}', filename)
            return None
        else:
            img = Image.open(filename)
            return img


    def read_file(self,setting):


        person_names = os.listdir(self.opt.root_dir+'/'+self.opt.img_dir)
        att_mat     = scio.loadmat(self.opt.root_dir+'/'+self.opt.att_file_name)

        image_names = att_mat['name'].squeeze()
        att_vecs    = att_mat['label'].squeeze()

        n_person        = len(person_names)
        person_identity = range(n_person)
        person_names = np.array(person_names, dtype=str)

        all_img_att_label_list = []
        all_labels             = []

        for i in range(att_vecs.shape[0]):
            image_path = image_names[i][0]
            image_path = image_path.split('\\')
            person_name = image_path[0]
            image_name  = image_path[1]
            image_path = person_name+ '/' +image_name
            label_idx = np.where(person_names == person_name)[0][0]
            label     = person_identity[label_idx]
            all_labels.append(label)
            att       = att_vecs[i,:]
            all_img_att_label_list.append((image_path,att,label))

        ## compute class embeddings
        all_labels    = np.array(all_labels)
        unique_labels = np.unique(all_labels)

        # split train and test

        if not os.path.exists(self.split_file_dir):
            os.makedirs(self.split_file_dir)
            n_unique_label = unique_labels.shape[0]
            all_index = np.arange(n_unique_label)
            np.random.shuffle(all_index)
            n_train_identity = int(0.8*n_unique_label)
            train_identity   = unique_labels[all_index[:n_train_identity]]
            test_identiy     = unique_labels[all_index[n_train_identity:]]

            train_idx        = []
            test_idx         = []
            for l in train_identity:
                idx = np.where(all_labels==l)[0]
                train_idx.append(idx)
            train_idx = np.hstack(train_idx)

            for l in test_identiy:
                idx = np.where(all_labels==l)[0]
                test_idx.append(idx)
            test_idx = np.hstack(test_idx)

            scio.savemat(self.split_file_dir+'/'+'split.mat',{'train_idx':train_idx,'test_idx':test_idx})
        else:
            split_mat = scio.loadmat(self.split_file_dir+'/'+'split.mat')
            train_idx = split_mat['train_idx'].squeeze()
            test_idx  = split_mat['test_idx'].squeeze()

        class_embeddings = []
        for l in unique_labels:
            idx = np.where(all_labels==l)[0]
            class_att_vecs = att_vecs[idx,:]
            class_embedding = np.mean(class_att_vecs,axis=0)
            class_embeddings.append(class_embedding)
        self.class_embeddings = normalize(np.array(class_embeddings).astype(np.float),norm='l2',axis=1)

        if setting == 'test':
            idx = test_idx
        elif setting == 'trainval':
            idx = train_idx

        np.random.shuffle(idx)

        img_att_label_list = []
        labels             = []
        for i in idx:
            img_name        = all_img_att_label_list[i][0]
            binary_att      = all_img_att_label_list[i][1]
            label           = all_img_att_label_list[i][2]
            re_label        = np.where(unique_labels==label)[0][0]
            labels.append(re_label)
            class_embedding = self.class_embeddings[re_label,:]
            img_att_label_list.append((img_name,class_embedding,binary_att,re_label))


        if setting == 'test':
            test_labels = np.array(labels)
            self.test_unique_labels = np.unique(test_labels)
            self.test_class_embeddings = self.class_embeddings[self.test_unique_labels]
        elif (setting == 'trainval') or (setting == 'train'):
            train_labels = np.array(labels)
            self.train_unique_labels    = np.unique(train_labels)
            self.train_class_embeddings = self.class_embeddings[self.train_unique_labels]

        return img_att_label_list

class CelebA(Dataset):
    def __init__(self, opt, transform=None, setting='trainval',tag=None,pg_tag='probe'):
        '''

        :param filename: txt file contains img names and labels
        :param img_dir: root dir of img
        :param resize_height: if none, not resize
        :param resize_width:
        :param repeat: the repeat number of all sample data, default is 1, when repeat==None, infinite loop

        '''
        self.opt                       = opt
        self.probe_gallery_idx_dir = opt.root_dir + '/' + opt.pg_idx_dir
        self.img_att_label_list        = self.read_file(setting,tag=tag,pg_tag=pg_tag)
        self.index                     = np.arange(len(self.img_att_label_list))
        self.setting                   = setting
        self.transform                 = transform

    def __getitem__(self, index):
        #index = i % self.len
        idx = self.index[index]
        img_name,att,binary_att,label = self.img_att_label_list[idx]
        img_path           = os.path.join(self.opt.root_dir+'/'+self.opt.img_dir+'/'+img_name)
        img                = self.load_data(img_path)
        if self.transform is not None:
            img = self.transform(img)
        label             = np.array(label,dtype=int)
        att               = np.array(att,dtype=float)
        binary_att        = np.array(binary_att,dtype=int)
        if ((index == len(self.img_att_label_list)-1)) and (self.setting != 'test'):
            np.random.shuffle(self.index)
        return img, att,binary_att, label

    def __len__(self):

        data_len = len(self.img_att_label_list)
        return data_len

    def load_data(self, filename):

        '''
        read img and return PIL format
        :param filename:
        :return:img data
        '''

        if not os.path.exists(filename):
            print('Warning:no file:{}', filename)
            return None
        else:
            img = Image.open(filename)
            return img


    def read_file(self,setting,tag,pg_tag):

        att_file    = open(self.opt.root_dir+'/'+self.opt.Anno_dir+'/'+self.opt.att_filename,'r')
        label_file  = open(self.opt.root_dir+'/'+self.opt.Anno_dir+'/'+self.opt.label_filename,'r')
        split_file  = open(self.opt.root_dir+'/'+self.opt.split_dir+'/'+self.opt.split_filename,'r')
        att_lines   = att_file.readlines()[2:]
        label_lines = label_file.readlines()
        split_lines = split_file.readlines()
        train_idx   = []
        valid_idx   = []
        test_idx    = []
        i = 0
        # first, we get the idx to split dataset
        for line in split_lines:
            content_list = line.strip('\n').split(' ')
            if int(content_list[-1]) == 0:
                train_idx.append(i)
            elif int(content_list[-1]) == 1:
                valid_idx.append(i)
            else:
                test_idx.append(i)
            i+=1
        #second, we get each identity class embedding
        all_labels    = []
        all_img_names = []
        all_att_vecs  = []
        all_img_att_label_list = []
        for i in range(len(att_lines)):
            att_line     = att_lines[i]
            label_line   = label_lines[i]
            content_list = att_line.strip('\n').split(' ')
            content_list = [x for x in content_list if x != '']
            content_list = content_list[1:]
            att = []
            for x in content_list:
                if int(x) == 1:
                    att.append(int(x))
                else:
                    att.append(0)
            content = label_line.split(' ')
            name    = content[0]
            label   = int(content[1])-1
            all_labels.append(label)
            all_img_names.append(name)
            all_att_vecs.append(att)
            all_img_att_label_list.append((name, att, label))

        all_att_vecs  = np.array(all_att_vecs)
        all_labels    = np.array(all_labels)
        unique_labels = np.unique(all_labels)
        all_class_embedding_list = []

        for label in unique_labels:
            idx = np.where(all_labels == label)[0]
            one_identity_vecs = all_att_vecs[idx,:]
            class_embedding   = np.reshape(np.mean(one_identity_vecs,axis=0),[1,-1])
            all_class_embedding_list.append(class_embedding)
        self.class_embeddings = normalize(np.vstack(all_class_embedding_list).astype(np.float),norm='l2',axis=1)

        # finally, we build the list of (img att label) for each dataset
        if setting   == 'trainval':
            idx = train_idx + valid_idx
        elif setting == 'train':
            idx = train_idx
        elif setting == 'valid':
            idx = valid_idx
        elif setting == 'test':
            idx = test_idx

        img_att_label_list = []
        labels             = []
        att_vecs           = []
        np.random.shuffle(idx)
        for i in idx:
            img_name   = all_img_att_label_list[i][0]
            binary_att = all_img_att_label_list[i][1]
            label      = all_img_att_label_list[i][2]
            att        = list(self.class_embeddings[label,:])
            labels.append(label)
            att_vecs.append(att)
            img_att_label_list.append((img_name,att,binary_att,label))

        if setting == 'test':
            test_labels = np.array(labels)
            self.test_unique_labels = np.unique(test_labels)
            self.test_class_embeddings = self.class_embeddings[self.test_unique_labels]
            if tag != None:
                if not os.path.exists(self.probe_gallery_idx_dir):
                    os.makedirs(self.probe_gallery_idx_dir)
                    probe_idx_list = []
                    gallery_idx_list = []
                    for l in self.test_unique_labels:
                        idx        =  np.where(test_labels==l)[0]
                        n_idx      = idx.shape[0]
                        n_probe    = int(n_idx*0.8)
                        n_gallery  = n_idx - n_probe
                        np.random.shuffle(idx)
                        probe_idx_list.append(idx[:n_probe])
                        gallery_idx_list.append(idx[n_probe:])

                    probe_idx   = np.hstack(probe_idx_list)
                    gallery_idx = np.hstack(gallery_idx_list)
                    scio.savemat(self.probe_gallery_idx_dir+'/'+'pg_idx.mat',{'probe_idx':probe_idx,'gallery_idx':gallery_idx})
                else:
                    mat = scio.loadmat(self.probe_gallery_idx_dir+'/'+'pg_idx.mat')
                    probe_idx   = mat['probe_idx'].squeeze()
                    gallery_idx = mat['gallery_idx'].squeeze()
                if pg_tag == 'probe':
                    probe_image_att_label_list = []
                    for idx in probe_idx:
                        probe_image_att_label_list.append(img_att_label_list[idx])
                    img_att_label_list = probe_image_att_label_list
                else:
                    gallery_image_att_label_list = []
                    for idx in gallery_idx:
                        gallery_image_att_label_list.append(img_att_label_list[idx])
                    img_att_label_list = gallery_image_att_label_list

        elif (setting == 'trainval') or (setting == 'train'):
            train_labels                = np.array(labels)
            self.train_unique_labels    = np.unique(train_labels)
            self.train_class_embeddings = self.class_embeddings[self.train_unique_labels]

        return img_att_label_list
