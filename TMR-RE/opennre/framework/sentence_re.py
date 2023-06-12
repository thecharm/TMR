import os, logging, json
from tqdm import tqdm
import torch
from torch import nn, optim
from .data_loader import SentenceRELoader
from .utils import AverageMeter
import numpy as np
from sklearn import metrics
from seqeval.metrics import f1_score

class SentenceRE(nn.Module):
    def __init__(self,
                 model,
                 train_path,
                 train_pic_path,
                 val_path,
                 val_pic_path,
                 test_path,
                 test_pic_path,
                 ckpt,
                 batch_size=64,
                 max_epoch=100,
                 lr=0.1,
                 weight_decay=1e-3,
                 warmup_step=300,
                 opt='adamw',
                 sample_ratio=1.0):

        super().__init__()
        self.max_epoch = max_epoch
        # Load data
        if train_path != None:
            self.train_loader = SentenceRELoader(
                train_path,
                train_pic_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                True,
                sample_ratio=sample_ratio)

        if val_path != None:
            self.val_loader = SentenceRELoader(
                val_path,
                val_pic_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False,
                sample_ratio=sample_ratio)

        if test_path != None:
            self.test_loader = SentenceRELoader(
                test_path,
                test_pic_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False,
                sample_ratio=sample_ratio)
        # Model
        self.model = model
        self.parallel_model = nn.DataParallel(self.model)
        # Criterion
        self.criterion = nn.CrossEntropyLoss()
        # Params and optimizer
        params = self.parameters()
        self.lr = lr
        if opt == 'sgd':
            self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
        elif opt == 'adamw':  # Optimizer for BERT
            from transformers import AdamW
            params = list(self.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            grouped_params = [
                {
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
                    'weight_decay': 0.01,
                    'lr': lr,
                    'ori_lr': lr
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': lr,
                    'ori_lr': lr
                }
            ]
            self.optimizer = AdamW(grouped_params, correct_bias=False)
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'adamw'.")
        # Warmup
        if warmup_step > 0:
            from transformers import get_linear_schedule_with_warmup
            training_steps = self.train_loader.dataset.__len__() // batch_size * self.max_epoch
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_step,
                                                             num_training_steps=training_steps)
        else:
            self.scheduler = None
        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.ckpt = ckpt

    def train_model(self, metric='acc'):
        best_metric = 0
        global_step = 0
        loader = [self.train_loader]
        for epoch in range(self.max_epoch):
            self.train()
            logging.info("=== Epoch %d train ===" % epoch)
            avg_loss = AverageMeter()
            avg_acc = AverageMeter()
            avg_f1 = AverageMeter()
            for l1 in loader:
                t = tqdm(l1, ncols=110)
                for iter, data in enumerate(t):
                    if torch.cuda.is_available():
                        for i in range(len(data)):
                            try:
                                data[i] = data[i].cuda()
                            except:
                                pass

                    label = data[0]
                    img_id = data[1]
                    args = data[2:]
                    logits, rep = self.parallel_model(*args)
                    loss = self.criterion(logits, label)
                    score, pred = logits.max(-1)  # (B)
                    acc = float((pred == label).long().sum()) / label.size(0)
                    f1 = metrics.f1_score(pred.cpu(), label.cpu(), average='macro')
                    # Log
                    avg_loss.update(loss.item(), 1)
                    avg_acc.update(acc, 1)
                    avg_f1.update(f1, 1)
                    t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg, f1=avg_f1.avg)
                    # Optimize
                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
            # Val
            logging.info("=== Epoch %d val ===" % epoch)
            result, correct_category, org_category, n_category, data_pred_t, data_pred_f, id_list, feature_list = self.eval_model(
                self.val_loader)
            logging.info('Metric {} current / best: {} / {}'.format(metric, result[metric], best_metric))
            if result[metric] > best_metric:
                best_metric = result[metric]

            folder_path = '/'.join(self.ckpt.split('/')[:-1])
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            torch.save({'state_dict': self.model.state_dict()}, self.ckpt)
            logging.info("Best ckpt and saved.")
        logging.info("Best %s on val set: %f" % (metric, best_metric))

    def eval_model(self, eval_loader):
        self.eval()
        avg_acc = AverageMeter()
        avg_loss = AverageMeter()
        pred_result = []
        id_list = []
        feature_list = []
        with torch.no_grad():
            t = tqdm(eval_loader, ncols=110)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                img_id = data[1]
                args = data[2:]
                logits, rep = self.parallel_model(*args)
                loss = self.criterion(logits, label)
                score, pred = logits.max(-1)  # (B)
                id_list.append(img_id)
                feature_list.append(rep)
                # Save result
                for i in range(pred.size(0)):
                    pred_result.append(pred[i].item())

                # Log
                acc = float((pred == label).long().sum()) / label.size(0)
                avg_acc.update(acc, pred.size(0))
                avg_loss.update(loss.item(), pred.size(0))
                t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg)
        result, correct_category, org_category, n_category, data_pred_t, data_pred_f = eval_loader.dataset.eval(
            pred_result)
        # save prediction into JSON
        with open('./pred_results.josn', 'w') as f2:
            json.dump(pred_result, f2)
        return result, correct_category, org_category, n_category, data_pred_t, data_pred_f, id_list, feature_list

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

