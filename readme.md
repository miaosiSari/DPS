# How to run the model

 (1) download the pretrained weights from https://pan.baidu.com/s/1uFIFc53NG4daRsJM3YjGjw, password: 51em. I give up Google Drive because it is super slow and I am really disappointed with it!!! Please unzip the weights into the "GOPRO_GAN" folder.

 (2) [CUDA_VISIBLE_DEVICES=xxx] python3 main.py <input_folder> <output_folder> [mode]

​      mode == 'all': generate 1x, 2x, 4x sharp images simultaneously

​      mode == 'middle': generate 2x sharp images

​      mode == 'deblur': generate 1x sharp images

​      mode == 'final': generate 4x sharp images

​      default mode is final

For example, "CUDA_VISIBLE_DEVICES=0,1 python3 main.py in out middle" means you are goint to use GPU 0, 1 to do super-resolution(2x) task. The input images are in the folder "in" and the output images will be in the folder "out". If the output folder does not exist, the program would use os.mkdir to create an empty one.

<img src="image/demo.gif" width="400px"/> 
<img src="image/demo2.gif" width="400px"/>
requirements:

python3

Tensorflow>=1.5.0

CUDA9.0
