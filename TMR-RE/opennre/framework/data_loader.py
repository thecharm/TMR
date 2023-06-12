import torch
import torch.utils.data as data
import os, random, json, logging
import numpy as np
import sklearn.metrics
import timm
import cv2
from tqdm import tqdm, trange
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer

class SentenceREDataset(data.Dataset):
    """
    Sentence-level relation extraction dataset
    """

    def __init__(self, text_path, pic_path, rel2id, tokenizer, sample_ratio, kwargs):
        """
        Args:
            text_path: path of the input text
            pic_path: path of the input image
            rel2id: dictionary of relation->id mapping
            tokenizer: function of tokenizing
            sample_ratio: training data sampling ratio
        """

        super().__init__()
        self.text_path = text_path
        if 'train' in text_path:
            mode = 'train'
        elif 'val' in text_path:
            mode = 'val'
        else:
            mode = 'test'

        # get generated images path

        # self.pic_path_FineGrained_dif = '/home/thecharm/re/RE/diffusion_pic/{}/'.format(mode)
        self.pic_path_FineGrained_dif = pic_path.replace('img_org', 'diffusion_pic')
        # get original images path
        self.pic_path_FineGrained_ori = pic_path
        self.pic_path_CoarseGrained_ori = pic_path.replace('org', 'vg')

        # Load the text file
        self.data = []
        f = open(text_path, encoding='UTF-8')
        f_lines = f.readlines()
        for i1 in tqdm(range(len(f_lines))):
            line = f_lines[i1].rstrip()
            if len(line) > 0:
                dic1 = eval(line)
                self.data.append(dic1)
        f.close()
        self.rel2id = rel2id
        logging.info(
            "Loaded sentence RE dataset {} with {} lines and {} relations.".format(text_path, len(self.data),
                                                                                   len(self.rel2id)))

        # get the path of reflecting dict
        self.img_aux_path_dif = text_path.replace('ours_{}.txt'.format(mode), 'mre_dif_{}_dict.pth'.format(mode))
        self.img_aux_path_ori = text_path.replace('ours_{}.txt'.format(mode), 'mre_{}_dict.pth'.format(mode))
        # load the reflecting dict
        self.state_dict_dif = torch.load(self.img_aux_path_dif)
        self.state_dict_ori = torch.load(self.img_aux_path_ori)

        # get the path of correlation scores
        self.weak_ori = text_path.replace('ours_{}.txt'.format(mode), '{}_weight_weak.txt'.format(mode))
        self.strong_ori = text_path.replace('ours_{}.txt'.format(mode), '{}_weight_strong.txt'.format(mode))
        self.weak_dif = text_path.replace('ours_{}.txt'.format(mode), 'dif_{}_weight_weak.txt'.format(mode))
        self.strong_dif = text_path.replace('ours_{}.txt'.format(mode), 'dif_{}_weight_strong.txt'.format(mode))
        # load the correlation scores
        with open(self.weak_ori, 'r', encoding='utf-8') as f_rel:
            lines = f_rel.readlines()
            self.weak_ori = {}
            for line in lines:
                img_id_key, score = line.split('\t')[0], float(line.split('\t')[1].replace('\n', ''))
                self.weak_ori[img_id_key] = score
        with open(self.strong_ori, 'r', encoding='utf-8') as f_rel:
            lines = f_rel.readlines()
            self.strong_ori = {}
            for line in lines:
                img_id_key, score = line.split('\t')[0], float(line.split('\t')[1].replace('\n', ''))
                self.strong_ori[img_id_key] = score
        with open(self.weak_dif, 'r', encoding='utf-8') as f_rel:
            lines = f_rel.readlines()
            self.weak_dif = {}
            for line in lines:
                img_id_key, score = line.split('\t')[0], float(line.split('\t')[1].replace('\n', ''))
                self.weak_dif[img_id_key] = score
        with open(self.strong_dif, 'r', encoding='utf-8') as f_rel:
            lines = f_rel.readlines()
            self.strong_dif = {}
            for line in lines:
                img_id_key, score = line.split('\t')[0], float(line.split('\t')[1].replace('\n', ''))
                self.strong_dif[img_id_key] = score

        self.tokenizer = tokenizer
        self.kwargs = kwargs
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        self.tokenizer_ = BertTokenizer.from_pretrained('bert-base-uncased')

        # get the path of phrases
        self.phrase_path = text_path.replace('ours_{}.txt'.format(mode), 'phrase_text_{}.json'.format(mode))
        f_grounding = open(self.phrase_path, 'r')
        self.phrase_data = json.load(f_grounding)

        self.sample_ratio = sample_ratio
        sample_indexes = random.choices(list(range(len(f_lines))), k=int(len(f_lines) * sample_ratio))
        if 'train' in pic_path:
            new_state_dict = {}
            new_state_dict2 = {}
            new_grounding_text = {}
            num = 0
            for idx in sample_indexes:
                new_state_dict[num] = self.state_dict_dif[idx]
                new_state_dict2[num] = self.state_dict_ori[idx]
                new_grounding_text[str(num)] = self.phrase_data[str(idx)]
                num += 1
            self.state_dict_dif = new_state_dict
            self.state_dict_ori = new_state_dict2
            self.phrase_data = new_grounding_text
            tmp_data = [self.data[idx] for idx in sample_indexes]
            self.data = tmp_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        aux_imgs_dif = []
        aux_imgs_ori = []
        image_dif = Image.open(os.path.join(self.pic_path_FineGrained_dif, self.data[index]['img_id'])).convert('RGB')

        for i in range(3):
            # get visual objects from generated images
            if len(self.state_dict_dif[index])>i:
                xy = self.state_dict_dif[index][i].tolist()
                aux_img_dif = image_dif.crop((xy[0], xy[1], xy[2], xy[3]))
                try:
                    img_features = self.transform(aux_img_dif).tolist()
                    aux_img_dif = img_features
                    aux_imgs_dif.append(aux_img_dif)
                except:
                    a = 1 # do nothing and skip

            # get visual objects from original images
            if len(self.state_dict_ori[index])>i:
                image_ori_object = Image.open(os.path.join(self.pic_path_CoarseGrained_ori, 'crops/' + self.state_dict_ori[index][i])).convert('RGB')
                img_features_ori = self.transform(image_ori_object).tolist()
                aux_img_ori = img_features_ori
                aux_imgs_ori.append(aux_img_ori)

        # padding zero tensor if less than 3
        for i in range(3 - len(aux_imgs_dif)):
            aux_imgs_dif.append(torch.zeros((3, 224, 224)).tolist())

        for i in range(3 - len(aux_imgs_ori)):
            aux_imgs_ori.append(torch.zeros((3, 224, 224)).tolist())

        assert len(aux_imgs_dif) == len(aux_imgs_ori) == 3

        item = self.data[index]
        item['grounding'] = self.phrase_data[str(index)] # phrase
        seq = list(self.tokenizer(item, **self.kwargs)) # input ids
        img_id = item['img_id'] # img_id
        h = item['h'] # head entity
        t = item['t'] # tail entity

        # load the generated image
        image_dif = cv2.imread((os.path.join(self.pic_path_FineGrained_dif, self.data[index]['img_id'])))
        size = (224, 224)
        img_features_dif = cv2.resize(image_dif, size, interpolation=cv2.INTER_AREA)
        img_features_dif = torch.tensor(img_features_dif)
        img_features_dif = img_features_dif.transpose(1, 2).transpose(0, 1)
        img_dif = torch.reshape(img_features_dif, (3, 224, 224)).to(torch.float32).tolist()

        # load the original image
        image_ori = cv2.imread((os.path.join(self.pic_path_FineGrained_ori, self.data[index]['img_id'])))
        size = (224, 224)
        img_features_ori = cv2.resize(image_ori, size, interpolation=cv2.INTER_AREA)
        img_features_ori = torch.tensor(img_features_ori)
        img_features_ori = img_features_ori.transpose(1, 2).transpose(0, 1)
        img_ori = torch.reshape(img_features_ori, (3, 224, 224)).to(torch.float32).tolist()

        pic_dif = [img_dif]
        pic_ori = [img_ori]
        pic_dif_objects = aux_imgs_dif
        pic_ori_objects = aux_imgs_ori

        np_pic1 = np.array(pic_dif).astype(np.float32)
        np_pic2 = np.array(pic_ori).astype(np.float32)
        np_pic3 = np.array(pic_dif_objects).astype(np.float32)
        np_pic4 = np.array(pic_ori_objects).astype(np.float32)
        weight = [self.weak_ori[img_id], self.strong_ori[img_id], self.weak_dif[img_id], self.strong_dif[img_id]]

        list_p1 = list(torch.tensor(np_pic1).unsqueeze(0))
        list_p2 = list(torch.tensor(np_pic2).unsqueeze(0))
        list_p3 = list(torch.tensor(np_pic3).unsqueeze(0))
        list_p4 = list(torch.tensor(np_pic4).unsqueeze(0))

        res = [self.rel2id[item['relation']]] + [img_id] + seq + list_p1 + list_p2 + list_p3 + list_p4 + [torch.tensor(weight)]

        return res  # label, seq1, seq2, ...,pic

    def collate_fn(data):
        data = list(zip(*data))
        labels = data[0]
        img_id = data[1]
        seqs = data[2:]
        batch_labels = torch.tensor(labels).long()  # (B)
        batch_seqs = []
        for seq in seqs:
            # print(seq)
            batch_seqs.append(torch.cat(seq, 0))  # (B, L)
        return [batch_labels] + [img_id] + batch_seqs

    def eval(self, pred_result, use_name=False):
        """
        Args:
            pred_result: a list of predicted label (id)
                Make sure that the `shuffle` param is set to `False` when getting the loader.
            use_name: if True, `pred_result` contains predicted relation names instead of ids
        Return:
            {'acc': xx}
        """
        correct = 0
        total = len(self.data)
        correct_positive = 0
        pred_positive = 0
        gold_positive = 0
        correct_category = np.zeros([31, 1])
        org_category = np.zeros([31, 1])
        n_category = np.zeros([31, 1])
        data_with_pred_T = []
        data_with_pred_F = []
        neg = -1
        for name in ['NA', 'na', 'no_relation', 'Other', 'Others', 'none', 'None']:
            if name in self.rel2id:
                if use_name:
                    neg = name
                else:
                    neg = self.rel2id[name]
                break
        y_pred = []
        y_gt = []
        for i in range(total):
            y_pred.append(pred_result[i])
            y_gt.append(self.rel2id[self.data[i]['relation']])
            if use_name:
                golden = self.data[i]['relation']
            else:
                golden = self.rel2id[self.data[i]['relation']]  # Ground Truth Label
                n_category[golden] += 1
            data_with_pred = (str(self.data[i]) + str(pred_result[i]))
            if golden == pred_result[i]:
                correct += 1
                data_with_pred_T.append(data_with_pred)
                if golden != neg:
                    correct_positive += 1
                    correct_category[golden] += 1
                else:
                    correct_category[0] += 1
            else:
                data_with_pred_F.append(data_with_pred)
            if golden != neg:
                gold_positive += 1
                org_category[golden] += 1
            else:
                org_category[0] += 1
            if pred_result[i] != neg:
                pred_positive += 1
        acc = float(correct) / float(total)
        try:
            micro_p = float(correct_positive) / float(pred_positive)
        except:
            micro_p = 0
        try:
            micro_r = float(correct_positive) / float(gold_positive)
        except:
            micro_r = 0
        try:
            micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
        except:
            micro_f1 = 0

        result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1}
        logging.info('Evaluation result: {}.'.format(result))
        return result, correct_category, org_category, n_category, data_with_pred_T, data_with_pred_F


def SentenceRELoader(text_path,  pic_path, rel2id, tokenizer,
                     batch_size, shuffle, num_workers=8, sample_ratio=1.0, collate_fn=SentenceREDataset.collate_fn, **kwargs):
    dataset = SentenceREDataset(text_path=text_path, pic_path=pic_path,
                                rel2id=rel2id,
                                tokenizer=tokenizer,
                                sample_ratio=sample_ratio,
                                kwargs=kwargs)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return data_loader




