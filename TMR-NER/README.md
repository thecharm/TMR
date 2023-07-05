# TMR-NER

Code for the ACL2023 paper "[Rethinking Multimodal Entity and Relation Extraction from a Translation Point of View](xxxxxx)" (Named Entity Recognition Section).


Requirements
==========
To run the codes, you need to install the requirements:
```
pip install -r requirements.txt
```

Data Preprocess
==========
To extract visual objects, following xxxxx, we first use the NLTK parser to extract noun phrases from the text and apply the [visual grouding toolkit](https://github.com/zyang-ur/onestage_grounding) to detect objects. Detailed steps are as follows:

1. Using the NLTK parser (or Spacy, textblob) to extract noun phrases from the text.
2. Applying the [visual grouding toolkit](https://github.com/zyang-ur/onestage_grounding) to detect objects. For the original images, the extracted objects are stored in `twitter2015_aux_images`. The images of the object obey the following naming format: `imgname_pred_yolo_crop_num.png`, where `imgname` is the name of the raw image corresponding to the object, `num` is the number of the object predicted by the toolkit. For the generated images, the extracted objects are stored in the form of coordinates in file `twitter2015/train_grounding_cut_dif.pth`.
3. For the orginal images, we construct a dictionary to record the correspondence between the raw images and the objects. Taking `twitter2015/twitter2015_train_dict.pth` as an example, the format of the dictionary can be seen as follows: `{imgname:['imgname_pred_yolo_crop_num0.png', 'imgname_pred_yolo_crop_num1.png', ...] }`, where key is the name of raw images, value is a List of the objects.
4. For the generated images, when processing images, we crop them using the coordinates in the file to obtain visual objects.
5. For the original images, we suggest that you can use our visual objects. As for the generated images, if you want to use your own images instead of ours, we suggest that you can put your images in the path `ner15_diffusion_pic` and detect visual objects through [visual grouding toolkit](https://github.com/zyang-ur/onestage_grounding) based on the giving phrases. 


Data Download
==========

+ Twitter2015 & Twitter2017

You need to download three kinds of data to run the code.

1. The raw images of Twitter2015 and Twitter2017.
2. The visual objects from the raw images.
3. Our generated images of Titter2015 and Twitter2017.

You can download these data by XXXXXXX, and then place folders `twitter2015_images`, `twitter2017_images`, `ner15_diffusion_pic`, `ner17_diffusion_pic`, `twitter2015_aux_images`, and `twitter2017_aux_images` in the "./data" path.
	
Files' Structure
==========

The expected structure of files is:

```
TMR-NER
 |-- ckpt # save the check point
 |-- data
 |    |-- twitter2015  # text data
 |    |    |-- train.txt
 |    |    |-- valid.txt
 |    |    |-- test.txt
 |    |    |-- twitter2015_train_dict.pth  # {imgname: [object-image]}
 |    |    |-- ...
 |    |    |-- ner_diff_train_weight_strong.txt  # strong correlation score for generated image
 |    |    |-- ner_train_weight_strong.txt  # strong correlation score for original image
 |    |    |-- ner_diff_train_weight_weak.txt  # weak correlation score for generated image
 |    |    |-- ner_train_weight_weak.txt  # weak correlation score for original image
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

`train.py` is thr file that sets up training, testing, and other processes.

`run.py` is used for running the whole program.


Train
==========

## NER Task

The data path and GPU related configuration are in the `run.py`. To train ner model, run this script:

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

Test
==========
## NER Task

To test ner model, you can use the tained model and set `load_path` to the model path, then run following script:

```shell
python -u run.py \
      --dataset_name="twitter15/twitter17" \
      --bert_name="bert-large-uncased" \
      --only_test \
      --max_seq=70 \
      --sample_ratio=1.0 \
      --load_path='your_ner_ckpt_path'
```

