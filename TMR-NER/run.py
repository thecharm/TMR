import os
import argparse
import logging
import sys
sys.path.append("..")

import torch
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from models.bert_model import TMR_NERModel
from processor.dataset import NERProcessor, NERDataset
from modules.train import NERTrainer

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# from tensorboardX import SummaryWriter

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    'twitter15': TMR_NERModel,
    'twitter17': TMR_NERModel
}

TRAINER_CLASSES = {
    'twitter15': NERTrainer,
    'twitter17': NERTrainer
}
DATA_PROCESS = {
    'twitter15': (NERProcessor, NERDataset),
    'twitter17': (NERProcessor, NERDataset)
}

DATA_PATH = {
    'twitter15': {
                # input text data
                'train': './data/twitter2015/train.txt',
                'dev':  './data/twitter2015/val.txt',
                'test':  './data/twitter2015/test.txt',

                # visual objects data
                'train_auximgs':  './data/twitter2015/twitter2015_train_dict.pth',
                'dev_auximgs':  './data/twitter2015/twitter2015_val_dict.pth',
                'test_auximgs':  './data/twitter2015/twitter2015_test_dict.pth',
                'train_auximgs_dif':  './data/twitter2015/train_grounding_cut_dif.pth',
                'dev_auximgs_dif':  './data/twitter2015/val_grounding_cut_dif.pth',
                'test_auximgs_dif':  './data/twitter2015/test_grounding_cut_dif.pth',

                # correlation coefficient data
                'train_weak_ori': './data/twitter2015/ner_train_weight_weak.txt',
                'dev_weak_ori': './data/twitter2015/ner_val_weight_weak.txt',
                'test_weak_ori': './data/twitter2015/ner_test_weight_weak.txt',
                'train_strong_ori': './data/twitter2015/ner_train_weight_strong.txt',
                'dev_strong_ori': './data/twitter2015/ner_val_weight_strong.txt',
                'test_strong_ori': './data/twitter2015/ner_test_weight_strong.txt',
                'train_weak_dif': './data/twitter2015/ner_diff_train_weight_weak.txt',
                'dev_weak_dif': './data/twitter2015/ner_diff_val_weight_weak.txt',
                'test_weak_dif': './data/twitter2015/ner_diff_test_weight_weak.txt',
                'train_strong_dif': './data/twitter2015/ner_diff_train_weight_strong.txt',
                'dev_strong_dif': './data/twitter2015/ner_diff_val_weight_strong.txt',
                'test_strong_dif': './data/twitter2015/ner_diff_test_weight_strong.txt',
                
                # phrase text data
                'train_grounding_text':'./data/twitter2015/ner15_grounding_text_train.json',
                'dev_grounding_text': './data/twitter2015/ner15_grounding_text_val.json',
                'test_grounding_text': './data/twitter2015/ner15_grounding_text_test.json',
    },

    'twitter17': {
                # text data
                'train': './data/twitter2017/train.txt',
                'dev':  './data/twitter2017/valid.txt',
                'test':  './data/twitter2017/test.txt',
        
                # visual objects data
                'train_auximgs': './data/twitter2017/twitter2017_train_dict.pth',
                'dev_auximgs': './data/twitter2017/twitter2017_val_dict.pth',
                'test_auximgs': './data/twitter2017/twitter2017_test_dict.pth',
                'train_auximgs_dif':  './data/twitter2017/train_grounding_cut_dif.pth',
                'dev_auximgs_dif':  './data/twitter2017/val_grounding_cut_dif.pth',
                'test_auximgs_dif':  './data/twitter2017/test_grounding_cut_dif.pth',

                # correlation coefficient data
                'train_weak_ori': './data/twitter2017/ner_train_weight_weak.txt',
                'dev_weak_ori': './data/twitter2017/ner_val_weight_weak.txt',
                'test_weak_ori': './data/twitter2017/ner_test_weight_weak.txt',
                'train_strong_ori': './data/twitter2017/ner_train_weight_strong.txt',
                'dev_strong_ori': './data/twitter2017/ner_val_weight_strong.txt',
                'test_strong_ori': './data/twitter2017/ner_test_weight_strong.txt',
                'train_weak_dif': './data/twitter2017/diff_ner_train_weight_weak.txt',
                'dev_weak_dif': './data/twitter2017/diff_ner_val_weight_weak.txt',
                'test_weak_dif': './data/twitter2017/diff_ner_test_weight_weak.txt',
                'train_strong_dif': './data/twitter2017/diff_ner_train_weight_strong.txt',
                'dev_strong_dif': './data/twitter2017/diff_ner_val_weight_strong.txt',
                'test_strong_dif': './data/twitter2017/diff_ner_test_weight_strong.txt',

                # phrase text data
                'train_grounding_text': './data/twitter2017/ner17_grounding_text_train.json',
                'dev_grounding_text': './data/twitter2017/ner17_grounding_text_val.json',
                'test_grounding_text': './data/twitter2017/ner17_grounding_text_test.json',
            },
        
}

# # image data
# IMG_PATH = {
#     'twitter15': '/home/thecharm/twitter15_data/twitter2015_images',
#     'twitter17': '/home/thecharm/data/twitter2017_images',
# }
#
# IMG_PATH_dif = {
#     'twitter15': {
#         'train': '/home/thecharm/re/RE/ner15_diffusion_pic/train/',
#         'test': '/home/thecharm/re/RE/ner15_diffusion_pic/test/',
#         'dev':  '/home/thecharm/re/RE/ner15_diffusion_pic/val/',
#     },
#     'twitter17': {
#         'train': '/home/thecharm/ner_17_diffusion/twitter17_diff_images',
#         'test': '/home/thecharm/ner_17_diffusion/twitter17_diff_images',
#         'dev': '/home/thecharm/ner_17_diffusion/twitter17_diff_images',
#     }
# }
#
# # auxiliary images
# AUX_PATH = {
#     'twitter15': {
#                 'train': '/home/thecharm/twitter15_data/twitter2015_aux_images/train/crops',
#                 'dev': '/home/thecharm/twitter15_data/twitter2015_aux_images/val/crops',
#                 'test': '/home/thecharm/twitter15_data/twitter2015_aux_images/test/crops',
#             },
#
#     'twitter17': {
#                 'train': '/home/thecharm/data/twitter2017_aux_images/train/crops',
#                 'dev': '/home/thecharm/data/twitter2017_aux_images/val/crops',
#                 'test': '/home/thecharm/data/twitter2017_aux_images/test/crops',
#             }
# }


# original image data
IMG_PATH = {
    'twitter15': './data/twitter2015_images',
    'twitter17': './data/twitter2017_images',
}

# generated image data， you can put your own generated image data in the following path
IMG_PATH_dif = {
    'twitter15': {
        'train': './data/ner15_diffusion_pic/train/',
        'test': './data/ner15_diffusion_pic/test/',
        'dev':  './data/ner15_diffusion_pic/val/',
    },
    'twitter17': {
        'train': './data/ner_17_diffusion/twitter17_diff_images',
        'test': './data/ner_17_diffusion/twitter17_diff_images',
        'dev': './data/ner_17_diffusion/twitter17_diff_images',
    }
}

# auxiliary images for visual objects
AUX_PATH = {
    'twitter15': {
                'train': './data/twitter2015_aux_images/train/crops',
                'dev': './data/twitter2015_aux_images/val/crops',
                'test': './data/twitter2015_aux_images/test/crops',
            },

    'twitter17': {
                'train': './data/twitter2017_aux_images/train/crops',
                'dev': './data/twitter2017_aux_images/val/crops',
                'test': './data/twitter2017_aux_images/test/crops',
            }
}

def set_seed(seed=2021):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='twitter15', type=str, help="The name of dataset.")
    parser.add_argument('--bert_name', default='bert-large-uncased', type=str, help="Pretrained language model path")
    parser.add_argument('--num_epochs', default=30, type=int, help="num training epochs")
    parser.add_argument('--device', default='cuda', type=str, help="cuda or cpu")
    parser.add_argument('--batch_size', default=32, type=int, help="batch size")
    parser.add_argument('--lr', default=1e-5, type=float, help="learning rate")
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--eval_begin_epoch', default=1, type=int, help="epoch to start evluate")
    parser.add_argument('--seed', default=1, type=int, help="random seed, default is 1")
    parser.add_argument('--load_path', default=None, type=str, help="Load model from load_path")
    parser.add_argument('--save_path', default=None, type=str, help="save model at save_path")
    parser.add_argument('--write_path', default=None, type=str, help="do_test=True, predictions will be write in write_path")
    parser.add_argument('--notes', default="", type=str, help="input some remarks for making save path dir.")
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--max_seq', default=128, type=int)
    parser.add_argument('--ignore_idx', default=-100, type=int)
    parser.add_argument('--sample_ratio', default=1.0, type=float, help="only for low resource.")

    args = parser.parse_args()

    data_path, img_path, img_path_dif, aux_path = DATA_PATH[args.dataset_name], IMG_PATH[args.dataset_name], IMG_PATH_dif[args.dataset_name], AUX_PATH[args.dataset_name]
    model_class, Trainer = MODEL_CLASSES[args.dataset_name], TRAINER_CLASSES[args.dataset_name]
    data_process, dataset_class = DATA_PROCESS[args.dataset_name]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    set_seed(args.seed) # set seed, default is 1
    if args.save_path is not None:  # make save_path dir
        if not os.path.exists('./ckpt/' + args.save_path):
            os.makedirs('./ckpt/' + args.save_path, exist_ok=True)
    print(args)
    logdir = "logs/" + args.dataset_name+ "_"+str(args.batch_size) + "_" + str(args.lr) + args.notes
    # writer = SummaryWriter(logdir=logdir)
    writer=None

    processor = data_process(data_path, args.bert_name)
    train_dataset = dataset_class(processor, transform, img_path, aux_path, img_path_dif['train'],args.max_seq, sample_ratio=args.sample_ratio, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    dev_dataset = dataset_class(processor, transform, img_path, aux_path, img_path_dif['dev'],args.max_seq, mode='dev')
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    test_dataset = dataset_class(processor, transform, img_path, aux_path, img_path_dif['test'],args.max_seq, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    label_mapping = processor.get_label_mapping()
    label_list = list(label_mapping.keys())
    model = TMR_NERModel(label_list, args)

    trainer = Trainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader, model=model,
                      label_map=label_mapping, args=args, logger=logger, writer=writer)

    if args.do_train:
        # train
        trainer.train()
        # test best model
        args.load_path = os.path.join('./ckpt'+args.save_path, 'best_model.pth')
        trainer.test()

    if args.only_test:
        # only do test
        trainer.test()

    torch.cuda.empty_cache()
    # writer.close()

if __name__ == "__main__":
    main()