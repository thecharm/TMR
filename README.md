# Source Code of TMR for Multimodal Entity and Relation Extraction
Official Implementation of our Paper ["Rethinking Multimodal Entity and Relation Extraction from A Translation Point of View"](https://aclanthology.org/2023.acl-long.376/) in ACL 2023.

## Motivation
![intro](Figure/Introd.png)

We borrow the idea of “back-translation” for this purpose. The text description is first "translated" to an image reference using diffusion models. The reference is “back-translated” to text words/phrases. The alignment/misalignment can then be evaluated and used to help the extraction.

## Model Architecture
![Model](Figure/Model.png) 

The framework of the proposed Translation motivated Multimodal Representation learning (TMR), generates divergence-aware cross-modal representations by introducing two additional streams of Generative Back-translation and High-Resource Divergence Estimation.

## Required Environment
To run the codes, you need to install the requirements for [NER](TMR-NER/) or [RE](TMR-RE).
```
pip install -r requirements.txt
```

## Data Preparation
+ Twitter2015 & Twitter2017

You need to download three kinds of data to run the code.

1. The raw images of [Twitter2015]((https://drive.google.com/file/d/1qAWrV9IaiBadICFb7mAreXy3llao_teZ/view?usp=sharing)) and [Twitter2017](https://drive.google.com/file/d/1ogfbn-XEYtk9GpUECq1-IwzINnhKGJqy/view?usp=sharing). 
2. The visual objects from the raw images from [HVPNeT](https://github.com/zjunlp/HVPNeT), many thanks.
3. Our generated images of [Twitter2015](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21040753r_connect_polyu_hk/EfjInJb_StpNr1tJa-kmnwIBqMAAuPkz7He2yrPFUYEbQw?e=2rKYnB) and [Twitter2017](https://drive.google.com/file/d/1m9uhILG_5uhGi53kpYkmSsUXhepWfoLK/view?usp=sharing).

Then you should put folders `twitter2015_images`, `twitter2017_images`, `ner15_diffusion_pic`, `ner17_diffusion_pic`, `twitter2015_aux_images`, and `twitter2017_aux_images` under the "./data" directory.

+ MNRE

You need to download three kinds of data to run the code.

1. The raw images of [MNRE](https://github.com/thecharm/MNRE).
2. The visual objects from the raw images from [HVPNeT](https://github.com/zjunlp/HVPNeT), many thanks.
3. Our generated images of [MNRE](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21040753r_connect_polyu_hk/EU6TTUViKZJFudQEkGg_5aABTh68dox-ITUebV65PTvuGQ?e=0fys6m).

Then you should put folders `img_org`, `img_vg`, `diffusion_pic` under the "./data" path.

## Data Preprocessing
To extract visual objects, we first use the NLTK parser to extract noun phrases from the text and apply the [visual grouding toolkit](https://github.com/zyang-ur/onestage_grounding) to detect objects. Detailed steps are as follows:

1. Using the NLTK parser (or Spacy, textblob) to extract noun phrases from the text.
2. Applying the [visual grounding toolkit](https://github.com/zyang-ur/onestage_grounding) to detect objects. For the original images, the extracted objects are stored in `img_vg`. The images of the object obey the following naming format: `imgname_pred_yolo_crop_num.png`, where `imgname` is the name of the raw image corresponding to the object, `num` is the number of the object predicted by the toolkit. For the generated images, the extracted objects are stored in the form of coordinates in the file `./data/txt/mre_dif_train_dif.pth`.
3. For the original images, we construct a dictionary to record the correspondence between the raw images and the objects. Taking `mre_train_dif.pth` as an example, the format of the dictionary can be seen as follows: `{imgname:['imgname_pred_yolo_crop_num0.png', 'imgname_pred_yolo_crop_num1.png', ...] }`, where key is the name of raw images, value is a List of the objects.
4. For the generated images, when processing images, we crop them using the coordinates in the file to obtain visual objects.
5. For the original images, we suggest that you can use our visual objects. As for the generated images, if you want to use your own images instead of ours, we suggest that you can put your images in the path `./data/diffusion_pic` and detect visual objects through [visual grounding toolkit](https://github.com/zyang-ur/onestage_grounding) based on the giving phrases from `./data/txt/phrase_text_train.json`.

## Path Structure
The expected structures of Paths are:

### Multimodal Named Entity Recognition
```
TMR-NER
 |-- ckpt # save the checkpoint
 |-- data
 |    |-- twitter2015  # text data
 |    |    |-- train.txt
 |    |    |-- valid.txt
 |    |    |-- test.txt
 |    |    |-- twitter2015_train_dict.pth  # {imgname: [object-image]}
 |    |    |-- ...
 |    |    |-- ner_diff_train_weight_strong.txt  # strong correlation score for generated images
 |    |    |-- ner_train_weight_strong.txt  # strong correlation score for original images
 |    |    |-- ner_diff_train_weight_weak.txt  # weak correlation score for generated images
 |    |    |-- ner_train_weight_weak.txt  # weak correlation score for original images
 |    |    |-- ...
 |    |    |-- ner15_grounding_text_train.json # {imgname: phrase for object detection}
 |    |    |-- ...
 |    |    |-- train_grounding_cut_dif.pth # {imgname: [coordinates]}
 |    |    |-- ...
 |    |-- twitter2015_images       # original image data
 |    |-- twitter2015_aux_images   # visual object image data for original image
 |    |-- ner15_diffusion_pic   # generated image data
 |    |-- twitter2017
 |    |-- twitter2017_images
 |    |-- twitter2017_aux_images
 |    |-- ner17_diffusion_pic
 |-- models	# models
 |    |-- bert_model.py # our model
 |    |-- modeling_bert.py
 |-- modules
 |    |-- metrics.py    # metric
 |    |-- train.py  # trainer
 |-- processor
 |    |-- dataset.py    # processor, dataset
 |-- logs     # code logs
 |-- run.py   # main 
```

`bert_model.py` is the file for our TMR-NER model.

`metrics.py` is the file that sets evaluation indicators such as F1 score.

`dataset.py` is the file for processing raw data.

`train.py` is the file that sets up training, testing, and other processes.

`run.py` is used for running the whole program.


### Multimodal Relation Extraction

```
TMR-RE
 |-- ckpt # save the check point
 |-- data
 |    |-- txt  # text data
 |    |    |-- ours_train.txt # input data
 |    |    |-- ours_val.txt
 |    |    |-- ours_test.txt
 |    |    |-- mre_train_dict.pth  # {imgname: [object-image]}
 |    |    |-- ...
 |    |    |-- dif_train_weight_strong.txt  # strong correlation score for generated image
 |    |    |-- train_weight_strong.txt  # strong correlation score for original image
 |    |    |-- dif_train_weight_weak.txt  # weak correlation score for generated image
 |    |    |-- train_weight_weak.txt  # weak correlation score for original image
 |    |    |-- ...
 |    |    |-- phrase_text_train.json # {imgname: phrase for object detection}
 |    |    |-- ...
 |    |    |-- mre_dif_train_dif.pth # {imgname: [coordinates]}
 |    |    |-- ...
 |    |-- img_org       # original image data
 |    |-- img_vg   # visual object image data for original image
 |    |-- diffusion_pic   # our generated image data
 |    |-- ours_rel2id.json # target relations
 |-- opennre	# main framework 
 |    |-- encoder # main model
 |    |    |-- bert_encoder.py # TMR-RE
 |    |    |-- modeling_bert.py
 |    |-- framework # processing files
 |    |    |-- data_loader.py # data processor
 |    |    |-- sentence_re.py # trainer
 |    |    |-- utils.py
 |    |-- model # classifier
 |    |    |-- softmax_nn.py # main classifier
 |    |    |-- modeling_bert.py 
 |    |    |-- base_model.py # supporting the classifier, no modification required
 |    |-- tokenization # tokenizers, no modification required
 |    |-- pretrain.py # basic file
 |    |-- utils.py # basic file
 |-- opennre.egg-info
 |-- results # saving the results
 |    |-- test # results for test set
 |    |-- val # results for validation set
 |-- run.py   # main 
```

A combination of `bert_encoder.py` and `softmax_nn.py` becomes our TMR-NER model.

`data_loader.py` is the file for processing raw data.

`sentence_re.py` is the file that sets up training, testing, and other processes.

`run.py` is used for running the whole program.

## Model Training
The data path and GPU-related configuration are in the `run.py`. To train the NER model, run this script:

```shell
python -u run.py \
	--dataset_name='twitter15/17'\
	--bert_name="bert-large-uncased"\
	--num_epochs=30\
	--eval_begin_epoch=12\
	--batch_size=12\
	--lr=3e-5\
	--warmup_ratio=0.01\
	--seed = 1234\
	--do_train\
	--ignore_idx=0\
	--max_seq=70\
	--sample_ratio=1.0\
	--save_path="your_ner_ckpt_path"
```

To train the RE model, run this script:

```shell
python run.py \
      --ckpt='your_re_ckpt_path' \
      --max_epoch=20 \
      --batch_size=16 \
      --lr=1e-5 \
      --sample_ratio=1.0
```

## Model Testing
To test the NER model, you can set `load_path` to the trained model path, then run the following script:

```shell
python -u run.py \
      --dataset_name="twitter15/twitter17" \
      --bert_name="bert-large-uncased" \
      --only_test \
      --max_seq=70 \
      --sample_ratio=1.0 \
      --load_path='your_ner_ckpt_path'
```

The checkpoints of our pre-trained models can be found here: [ner_twitter_17](https://drive.google.com/drive/folders/1Dk7x5o0Z31TTx0l8dcFI_ctbWENeSL-V?usp=sharing) and [ner_twitter_15](https://drive.google.com/drive/folders/18tNBGBcxg7HI3BFTR9Rkp8EA1hkQNgFb?usp=sharing).

The settings are similar for RE testing:


```shell
python run.py \
      --ckpt='your_re_ckpt_path' \
      --max_epoch=20 \
      --batch_size=16 \
      --lr=1e-5 \
      --only_test \
      --sample_ratio=1.0 \
```

## Divergence Estimation

Below is the Architecture of our Multimodal Divergence Estimator (MDE), which is trained on high-resource vision-language datasets, and Supervised Contrastive Learning (SCL) is applied to enhance the generalization.

![model2](Figure/Model2.png)   ![case](Figure/case1.png)

You can find the codes and [the model weights](https://drive.google.com/file/d/1YOIdZyy70tEgnGM7bvuGdEBfPX7PLwT7/view?usp=sharing) under the directory`Divergence Estimator`. If you want to apply this model in your custom dataset, please refer to the inference code of [ViLT](https://github.com/dandelin/ViLT).

## Back Translation

We generate the back-translated images from the text prompt with the [stable-diffusion model](https://github.com/CompVis/stable-diffusion). Please follow their instructions to install the corresponding environment.

Also, we provide the scripts to extract and save the generated images under the directory `Back Translation`. Just modify the paths and run the scripts:
```
python diffusion.py
```

Some generated examples are shown as:

![sup_case](Figure/sup_case.png)

## Citation

If you find this repo helpful, please cite the following:

``` latex
@inproceedings{zheng2023rethinking,
  title={Rethinking Multimodal Entity and Relation Extraction from a Translation Point of View},
  author={Zheng, Changmeng and Feng, Junhao and Cai, Yi and Wei, Xiaoyong and Li, Qing},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={6810--6824},
  year={2023}
}
```


