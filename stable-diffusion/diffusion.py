from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from tqdm import tqdm
import os

model_id = "stabilityai/stable-diffusion-2"
# print(os.path.exists('ner15_diffusion_pic/train/50447.png'))

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler,
                                               torch_dtype=torch.float16)
pipe = pipe.to("cuda")

types = ['val', 'test']
for type_ in types:
    pic_ = set()
    # file = open('/home/thecharm/re/RE/benchmark/ours/txt/ours_%s.txt'%type_, 'r')
    if type_=='val':
        file = open('/home/ubuntu/twitter2017/%s.txt'%'valid', 'r')
    else:
        file = open('/home/ubuntu/twitter2017/%s.txt'%type_, 'r')
    # file = open('%s.txt'%type_)
    lines = file.readlines()
    tokens = []
    for line in tqdm(lines):
        if line=='\n' or line=='\r':
            sentence = ' '.join(tokens[1:])
            img_id = tokens[0]
            # print('ner15_diffusion_pic/%s/%s'%(type_, img_id + '.png'))
            # if os.path.exists('ner15_diffusion_pic/%s/%s'%(type_, img_id + '.png')):
            #     continue
            if img_id in pic_:
                tokens = []
                continue
            image = pipe(sentence).images[0]
            pic_.add(img_id)
            image.save('ner17_diffusion_pic/%s/%s'%(type_, img_id + '.png'))
            tokens = []
        else:
            words = line.split('\t')
            tokens.append(words[0])

    # for line in tqdm(lines):
    #     line = eval(line)
    #     prompt = ' '.join(line['token'])
    #     img_id = line['img_id']
    #     if img_id in pic_:
    #         continue
    #     image = pipe(prompt).images[0]
    #     pic_.add(img_id)
    #     image.save('diffusion_pic_png/%s/%s'%(type_, img_id.replace('jpg','png')))

