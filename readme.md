# How to run the model

 (1) download the pretrained weights from https://pan.baidu.com/s/1gDryB10_WA1Mz6ZMcKVNkQ, passwd: y4eo. I give up Google Drive because it is super slow and I am really disappointed with it!!!

 (2) [CUDA_VISIBLE_DEVICES=xxx] python3 main.py <input_folder> <output_folder> [mode]

​      mode == 'all': generate 1x, 2x, 4x sharp images simultaneously

​      mode == 'middle': generate 2x sharp images

​      mode == 'deblur': generate 1x sharp images

​      mode == 'final': generate 4x sharp images

​      default mode is final

requirements:

python3

Tensorflow>=1.5.0

CUDA9.0