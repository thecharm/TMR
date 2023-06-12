import random
import os
import torch
import json
import ast
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torchvision import transforms
import logging
import json
logger = logging.getLogger(__name__)

class NERProcessor(object):
    def __init__(self, data_path, bert_name) -> None:
        self.data_path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=True)

    def load_from_file(self, mode="train", sample_ratio=1.0):
        """
        mode: dataset mode. Defaults to "train"
        sample_ratio: sample ratio in low resouce. Defaults to 1.0
        """
        load_file = self.data_path[mode]
        logger.info("Loading data from {}".format(load_file))

        # split text and img id
        with open(load_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            raw_words, raw_targets = [], []
            raw_word, raw_target = [], []
            imgs = []
            imgs_dif = []
            for line in lines:
                if line.startswith("IMGID:"):
                    img_id = line.strip().split('IMGID:')[1] + '.jpg'
                    img_id_dif = line + '.png'
                    imgs.append(img_id)
                    imgs_dif.append(img_id_dif)
                    continue
                if line != "\n":
                    raw_word.append(line.split('\t')[0])
                    label = line.split('\t')[1][:-1]
                    if 'OTHER' in label:
                        label = label[:2] + 'MISC'
                    raw_target.append(label)
                else:
                    raw_words.append(raw_word)
                    raw_targets.append(raw_target)
                    raw_word, raw_target = [], []
        assert len(raw_words) == len(raw_targets) == len(imgs) == len(imgs_dif), "{}, {}, {}, {}".format(len(raw_words), len(raw_targets), len(imgs), len(imgs_dif))

        # load visual objects from original and generated images
        aux_path = self.data_path[mode+"_auximgs"]
        aux_imgs = torch.load(aux_path)
        aux_path_dif = self.data_path[mode+"_auximgs_dif"]
        aux_imgs_dif = torch.load(aux_path_dif)

        # load weak correlation between text and original image
        with open(self.data_path['%s_weak_ori'%mode], 'r', encoding='utf-8') as f_rel:
            lines = f_rel.readlines()
            weak_ori = {}
            for line in lines:
                img_id_key, score = line.split('	')[0], float(line.split('	')[1].replace('\n', ''))
                weak_ori[img_id_key] = score

        # load strong correlation between text and original image
        with open(self.data_path['%s_strong_ori'%mode], 'r', encoding='utf-8') as f_rel:
            lines = f_rel.readlines()
            strong_ori = {}
            for line in lines:
                img_id_key, score = line.split('	')[0], float(line.split('	')[1].replace('\n', ''))
                strong_ori[img_id_key] = score

        # load weak correlation between text and generated image
        with open(self.data_path['%s_weak_dif'%mode], 'r', encoding='utf-8') as f_rel:
            lines = f_rel.readlines()
            weak_dif = {}
            for line in lines:
                img_id_key, score = line.split('	')[0], float(line.split('	')[1].replace('\n', ''))
                weak_dif[img_id_key] = score

        # load strong correlation between text and generated image
        with open(self.data_path['%s_strong_dif'%mode], 'r', encoding='utf-8') as f_rel:
            lines = f_rel.readlines()
            strong_dif = {}
            for line in lines:
                img_id_key, score = line.split('	')[0], float(line.split('	')[1].replace('\n', ''))
                strong_dif[img_id_key] = score

        # load phrases for visual objects detection
        with open(self.data_path['%s_grounding_text'%mode], 'r', encoding='utf-8') as f_ner_phrase_text:
            data_phrase_text = json.load(f_ner_phrase_text)

        # sample data, only for low-resource
        if sample_ratio != 1.0:
            sample_indexes = random.choices(list(range(len(raw_words))), k=int(len(raw_words)*sample_ratio))
            sample_raw_words = [raw_words[idx] for idx in sample_indexes]
            sample_raw_targets = [raw_targets[idx] for idx in sample_indexes]
            sample_imgs = [imgs[idx] for idx in sample_indexes]
            imgs_dif = [imgs_dif[idx] for idx in sample_indexes]

            assert len(sample_raw_words) == len(sample_raw_targets) == len(sample_imgs), "{}, {}, {}".format(len(sample_raw_words), len(sample_raw_targets), len(sample_imgs))
            return {"words": sample_raw_words, "targets": sample_raw_targets, "imgs": sample_imgs, "aux_imgs":aux_imgs, 'imgs_dif': imgs_dif, 'aux_imgs_dif': aux_imgs_dif, 'weak_ori':weak_ori, 'strong_ori':strong_ori,
                'weak_dif':weak_dif, 'strong_dif':strong_dif, 'phrase_text':data_phrase_text}

        return {"words": raw_words, "targets": raw_targets, "imgs": imgs, "aux_imgs":aux_imgs, 'imgs_dif': imgs_dif, 'aux_imgs_dif': aux_imgs_dif, 'weak_ori':weak_ori, 'strong_ori':strong_ori,
                'weak_dif':weak_dif, 'strong_dif':strong_dif, 'phrase_text':data_phrase_text}

    # transform labels to numbers
    def get_label_mapping(self):
        LABEL_LIST = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
        label_mapping = {label:idx for idx, label in enumerate(LABEL_LIST, 1)}
        label_mapping["PAD"] = 0
        return label_mapping


class NERDataset(Dataset):
    def __init__(self, processor, transform, img_path=None, aux_img_path=None, img_path_dif=None,
                 max_seq=40, sample_ratio=1, mode='train', ignore_idx=0) -> None:
        """
        processor: NERProcessor
        transform: image transformation operation
        img_path: the loading path for original images
        aux_img_path: the loading path for visual objects extracted from original images
        img_path_dif: the loading path for generated images
        max_seq: the max length for input text
        sample_ratio: sample ratio in low resouce. Defaults to 1.0
        mode: dataset mode. Defaults to "train"
        ignore_idx: the index for padding tokens
        """
        self.processor = processor
        self.transform = transform
        self.data_dict = processor.load_from_file(mode, sample_ratio)
        self.tokenizer = processor.tokenizer
        self.label_mapping = processor.get_label_mapping()
        self.max_seq = max_seq
        self.ignore_idx = ignore_idx
        self.img_path = img_path
        self.aux_img_path = aux_img_path[mode] if aux_img_path is not None else None
        self.img_path_dif = img_path_dif
        self.mode = mode
        self.sample_ratio = sample_ratio

    def __len__(self):
        return len(self.data_dict['words'])

    def __getitem__(self, idx):
        # get input text, labels and two kinds of images
        word_list, label_list, img, img_dif = self.data_dict['words'][idx], self.data_dict['targets'][idx], self.data_dict['imgs'][idx], self.data_dict['imgs_dif'][idx]
        # get correlation coefficient
        weak_ori, weak_dif, strong_ori, strong_dif = self.data_dict['weak_ori'][img], self.data_dict['weak_dif'][img], self.data_dict['strong_ori'][img], self.data_dict['strong_dif'][img]
        # get phrases for visual objects detection
        phrase_text = self.data_dict['phrase_text'][img]

        # text processing by BERT
        tokens, labels = [], []
        for i, word in enumerate(word_list):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            label = label_list[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(self.label_mapping[label])
                else:
                    labels.append(self.label_mapping["X"])
        if len(tokens) >= self.max_seq - 1:
            tokens = tokens[0:(self.max_seq - 2)]
            labels = labels[0:(self.max_seq - 2)]
        encode_dict = self.tokenizer.encode_plus(tokens, max_length=self.max_seq, truncation=True, padding='max_length')
        input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], encode_dict['attention_mask']
        labels = [self.label_mapping["[CLS]"]] + labels + [self.label_mapping["[SEP]"]] + [self.ignore_idx]*(self.max_seq-len(labels)-2)
        encode_dict_g = self.tokenizer.encode_plus(phrase_text, max_length=5, truncation=True, padding='max_length')
        input_ids_g, token_type_ids_g, attention_mask_g = encode_dict_g['input_ids'], encode_dict_g['token_type_ids'], encode_dict_g['attention_mask']

        # image process
        if self.img_path is not None and self.img_path_dif is not None:
            # fine-grained image feature processing
            img_path = os.path.join(self.img_path, img)
            try:
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
            except:
                # if the image doesn't exist, use all zero tensors for substitution
                image = torch.zeros((3, 224, 224))
            img_path_dif = os.path.join(self.img_path_dif, img_dif)
            try:
                image_dif = Image.open(img_path_dif).convert('RGB')
            except:
                image_dif = Image.open(img_path_dif.replace('\n', '')).convert('RGB')
            image_dif = self.transform(image_dif)

            # coarse-grained image feature processing
            if self.aux_img_path is not None:
                aux_imgs = []
                aux_img_paths = []
                if img in self.data_dict['aux_imgs']:
                    aux_img_paths = self.data_dict['aux_imgs'][img]
                    aux_img_paths = [os.path.join(self.aux_img_path, path) for path in aux_img_paths]
                for i in range(min(3, len(aux_img_paths))):
                    aux_img = Image.open(aux_img_paths[i]).convert('RGB')
                    aux_img = self.transform(aux_img)
                    aux_imgs.append(aux_img)

                # visual objects padding
                for i in range(3-len(aux_img_paths)):
                    aux_imgs.append(torch.zeros((3, 224, 224)))
                aux_imgs = torch.stack(aux_imgs, dim=0)
                assert len(aux_imgs) == 3

            if self.img_path_dif is not None:
                aux_imgs_dif = []
                for i in range(3):
                    # get the visual objects for generated images by coordinates
                    if len(self.data_dict['aux_imgs_dif'][img_dif.replace('\n', '')]) > i:
                        try:
                            xy = self.data_dict['aux_imgs_dif'][img_dif.replace('\n', '')][i].tolist()
                            aux_img_dif = image_dif.crop((xy[0], xy[1], xy[2], xy[3]))
                            img_features_dif = self.transform(aux_img_dif).tolist()
                            aux_imgs_dif.append(img_features_dif)
                        except:
                            aux_imgs_dif.append(torch.zeros((3, 224, 224)))

                # visual objects padding
                for i in range(3-len(aux_imgs_dif)):
                    aux_imgs_dif.append(torch.zeros((3, 224, 224)))
                aux_imgs_dif = torch.stack(aux_imgs_dif, dim=0)
                assert len(aux_imgs_dif) == 3

            weight = [weak_ori, strong_ori, weak_dif, strong_dif]

            if self.aux_img_path and self.img_path_dif:
                return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(
                    attention_mask), torch.tensor(
                    labels), image, aux_imgs, image_dif, aux_imgs_dif, torch.tensor(weight), torch.tensor(
                    input_ids_g), torch.tensor(token_type_ids_g), torch.tensor(attention_mask_g)
            elif self.aux_img_path:
                return torch.tensor(input_ids), torch.tensor(
                    attention_mask), torch.tensor(labels), image, aux_imgs
            elif self.img_path_dif:
                return torch.tensor(input_ids), torch.tensor(
                    attention_mask), torch.tensor(labels), image_dif, aux_imgs_dif

        assert len(input_ids) == len(token_type_ids) == len(attention_mask) == len(labels)
        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(labels)
