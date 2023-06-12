import torch
import os
from torch import nn
import torch.nn.functional as F
from torchcrf import CRF
from .modeling_bert import BertModel
from transformers.modeling_outputs import TokenClassifierOutput
from torchvision.models import resnet50
import timm
import math
import transformers
import numpy as np


class TMR_NERModel(nn.Module):
    def __init__(self, label_list, args):
        """
        label_list: the list of target labels
        args: argparse
        """
        super(TMR_NERModel, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.bert_name)  # get the pre-trained BERT model for the text
        self.bert_config = self.bert.config
        self.image2token_emb = 1024

        self.model_resnet50 = timm.create_model('resnet101', pretrained=True) # get the pre-trained ResNet model for the image

        self.num_labels = len(label_list)  # the number of target labels
        self.crf = CRF(self.num_labels, batch_first=True)
        self.fc = nn.Linear(1000, self.num_labels)
        self.dropout = nn.Dropout(0.4)

        self.linear_extend_pic = nn.Linear(self.bert.config.hidden_size, self.args.max_seq*self.image2token_emb)
        self.linear_pic = nn.Linear(2048, self.bert.config.hidden_size)

        # the attention mechanism for fine-grained features
        self.linear_q_fine = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.linear_k_fine = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.linear_v_fine = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)

        # the attention mechanism for coarse-grained features
        self.linear_q_coarse = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.linear_k_coarse = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.linear_v_coarse = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)

        self.combine_linear = nn.Linear(self.bert.config.hidden_size + self.image2token_emb*2, 1000)

    def forward(self, input_ids=None, attention_mask=None, labels=None, images=None, aux_imgs=None,
                images_dif=None, aux_images_dif=None, weight=None, input_ids_phrase=None, attention_mask_phrase=None):
        """
        input_ids: the ids of tokens from input text
        attention_mask: the attention mask for the token list
        labels: the ground truth
        images: the original images from the dataset
        aux_images: the visual objects from the original images
        images_dif: the generated images from stable-diffusion
        aux_images_dif: the visual objects from the generated images
        weight: the correlation coefficient between text and two types of images
        input_ids_phrase: the ids of tokens from phrases used for visual objects detection
        attention_mask_phrase: the attention mask for input_ids_phrase
        """

        feature_OriImg_FineGrained = self.model_resnet50.forward_features(images)
        feature_OriImg_CoarseGrained = self.model_resnet50.forward_features(aux_imgs.reshape(-1, 3, 224, 224))
        feature_DifImg_FineGrained = self.model_resnet50.forward_features(images_dif)
        feature_DifImg_CoarseGrained = self.model_resnet50.forward_features(aux_images_dif.reshape(-1, 3, 224, 224))

        pic_diff = torch.reshape(feature_DifImg_FineGrained, (-1, 2048, 49))
        pic_diff = torch.transpose(pic_diff, 1, 2)
        pic_diff = torch.reshape(pic_diff, (-1, 49, 2048))
        pic_diff = self.linear_pic(pic_diff)
        pic_diff_ = torch.sum(pic_diff, dim=1)

        pic_ori = torch.reshape(feature_OriImg_FineGrained, (-1, 2048, 49))
        pic_ori = torch.transpose(pic_ori, 1, 2)
        pic_ori = torch.reshape(pic_ori, (-1, 49, 2048))
        pic_ori = self.linear_pic(pic_ori)
        pic_ori_ = torch.sum(pic_ori,dim=1)

        pic_diff_objects = torch.reshape(feature_DifImg_CoarseGrained, (-1, 2048, 49))
        pic_diff_objects = torch.transpose(pic_diff_objects, 1, 2)
        pic_diff_objects = torch.reshape(pic_diff_objects, (-1, 3, 49, 2048))
        pic_diff_objects = torch.sum(pic_diff_objects, dim=2)
        pic_diff_objects = self.linear_pic(pic_diff_objects)
        pic_diff_objects_ = torch.sum(pic_diff_objects,dim=1)

        pic_ori_objects = torch.reshape(feature_OriImg_CoarseGrained, (-1, 2048, 49))
        pic_ori_objects = torch.transpose(pic_ori_objects, 1, 2)
        pic_ori_objects = torch.reshape(pic_ori_objects, (-1, 3, 49, 2048))
        pic_ori_objects = torch.sum(pic_ori_objects, dim=2)
        pic_ori_objects = self.linear_pic(pic_ori_objects)#*weight_objects[:,:,0].reshape(-1,3,1)
        pic_ori_objects_ = torch.sum(pic_ori_objects,dim=1)#.view(bsz, 16, 64)

        output_text = self.bert(input_ids, attention_mask)
        hidden_text = output_text['last_hidden_state']

        output_phrases = self.bert(input_ids_phrase, attention_mask_phrase)
        hidden_phrases = output_phrases['last_hidden_state']

        hidden_k_text = self.linear_k_fine(hidden_text)
        hidden_v_text = self.linear_v_fine(hidden_text)
        pic_q_diff = self.linear_q_fine(pic_diff)
        pic_diffusion = torch.sum(torch.tanh(self.att(pic_q_diff, hidden_k_text, hidden_v_text)), dim=1)

        hidden_k_text = self.linear_k_fine(hidden_text)
        hidden_v_text = self.linear_v_fine(hidden_text)
        pic_q_origin = self.linear_q_fine(pic_ori)
        pic_original = torch.sum(torch.tanh(self.att(pic_q_origin, hidden_k_text, hidden_v_text)), dim=1)

        hidden_k_phrases = self.linear_k_coarse(hidden_phrases)
        hidden_v_phrases = self.linear_v_coarse(hidden_phrases)
        pic_q_diff_objects = self.linear_q_coarse(pic_diff_objects)
        pic_diffusion_objects = torch.sum(torch.tanh(self.att(pic_q_diff_objects, hidden_k_phrases, hidden_v_phrases)), dim=1)

        hidden_k_phrases = self.linear_k_coarse(hidden_phrases)
        hidden_v_phrases = self.linear_v_coarse(hidden_phrases)
        pic_q_ori_objects = self.linear_q_coarse(pic_ori_objects)
        pic_original_objects = torch.sum(torch.tanh(self.att(pic_q_ori_objects, hidden_k_phrases, hidden_v_phrases)), dim=1)

        # correlation allocation
        pic_ori_final = (pic_original+pic_ori_) * weight[:, 1].reshape(-1, 1) + (pic_original_objects+pic_ori_objects_) * weight[:, 0].reshape(-1,1)
        pic_diff_final = (pic_diffusion+pic_diff_) * weight[:, 3].reshape(-1, 1) + (pic_diffusion_objects+pic_diff_objects_) * weight[:, 2].reshape(-1, 1)

        # assign image features to each token
        pic_ori = torch.tanh(self.linear_extend_pic(pic_ori_final).reshape(-1, self.args.max_seq, self.image2token_emb))
        pic_diff = torch.tanh(self.linear_extend_pic(pic_diff_final).reshape(-1, self.args.max_seq, self.image2token_emb))
        emissions = self.fc(torch.relu(self.combine_linear(torch.cat([hidden_text, pic_ori, pic_diff], dim=-1))))

        # classification
        logits = self.crf.decode(emissions, attention_mask.byte())
        loss = None
        if labels is not None:
            loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='sum')
        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )

    # the attention mechanism
    def att(self, query, key, value):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)  # (5,50)
        att_map = F.softmax(scores, dim=-1)

        return torch.matmul(att_map, value)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(sum=0.0, std=0.05)



