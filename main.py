#coding=utf-8
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
import math
import random
import json
import traceback
from PIL import Image
try:
    from matplotlib import pyplot as plt
except:
    pass
import time
import cv2
import os
import sys
try:
    import cPickle as pk
except:
    import pickle as pk
import tensorflow.contrib.slim as slim

size_low_res =  (180, 320, 3)
patchsize = 64
sizeCropH = patchsize
sizeCropW = patchsize
colors = 3
scale = 4
decay_step = 5
decay_rate1 = 0.90
decay_rate2 = 0.85
learning_rate1=2e-5
learning_rate2=5e-5
batchsize = 12
name = 'GOPRO'
path = name+'_log.txt'
mode = 'test'
l1_or_l2 = 'l2'
max_epochs = 15
now_epoch= 0
step = 0
pyfile = True


def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha*x)

def Conv(name, x, filter_size, in_filters, out_filters, strides, padding, relu=0.2):

    with tf.variable_scope(name):
        result = tf.layers.conv2d(x, out_filters, filter_size, (strides, strides), padding=padding, activation=None)
        if isinstance(relu, float):
            #print(name, 'relu', relu)
            return leaky_relu(result, alpha=relu)
        else:
            #print(name, 'nonrelu')
            return result
    

def C(name, x, filter_size, in_filters, out_filters, strides, relu=True):
    x = Conv(name, x, filter_size, in_filters, out_filters, strides, 'SAME', relu=relu)
    return x
    
    
def Conv_transpose(name, x, filter_size, out_filters, fraction = 2, padding = "SAME"):
    
    with tf.variable_scope(name):
        x = tf.layers.conv2d_transpose(x, out_filters, filter_size, (fraction,  fraction), padding)
        return x
    
def instance_norm(name, x):

    with tf.variable_scope(name):
        x = tf.contrib.layers.instance_norm(x)
    
    return x


def up_scaling_feature(name, x, n_feats):
    x = Conv_transpose(name = name + 'deconv', x = x, filter_size = 3,  out_filters = n_feats, fraction = 2, padding = 'SAME')
    x = instance_norm(name = name + 'instance_norm', x = x)
    x = leaky_relu(x)
    return x

def up_scaling(name, x, n_feats_in, n_feats_out):
    x = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], mode = 'REFLECT')
    x = Conv(name = name + 'shuffle_conv1', x = x, filter_size = 3, in_filters = n_feats_in, out_filters = n_feats_out*4, strides = 1, padding = 'VALID')
    x = tf.depth_to_space(x, 2) 
    #x = leaky_relu(x)
    return x

def U(name, x, n_feats_in, n_feats_out, conv_followed=True):
    x = up_scaling(name, x, n_feats_in, n_feats_out)
    x = leaky_relu(x)
    if conv_followed:
        x = C(name+"conv_in_upsampling", x, 3, n_feats_out, n_feats_out, 1, relu=False)
    return x

def res_block(name, x, n_feats):
    _res = x
    x = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], mode = 'REFLECT')
    x = Conv(name = name + 'conv1', x = x, filter_size = 3, in_filters = n_feats, out_filters = n_feats, strides = 1, padding = 'VALID')
    #x = instance_norm(name = name + 'instance_norm1', x = x)
    x = leaky_relu(x)

    x = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], mode = 'REFLECT')
    x = Conv(name = name + 'conv2', x = x, filter_size = 3, in_filters = n_feats, out_filters = n_feats, strides = 1, padding = 'VALID')
    #x = instance_norm(name = name + 'instance_norm2', x = x)
    x = x + _res 
    return x

def R(name, x, n_feats, blocks):
    for cnt in range(blocks):
        x = res_block(name+"res"+str(cnt), x, n_feats)
    return x


def down_scaling_feature(name, x, n_feats):
    x = Conv(name = name + 'conv', x = x, filter_size = 3, in_filters = n_feats, out_filters = n_feats * 2, strides = 2, padding = 'SAME')
    x = instance_norm(name = name + 'instance_norm', x = x)
    x = leaky_relu(x)

    return x

class nets:
    @staticmethod
    def deblur(x,scope='g_net', reuse=False):
        def ResnetBlock(x, dim, ksize, scope='rb'):
            with tf.variable_scope(scope):
                net = slim.conv2d(x, dim, [ksize, ksize], scope='conv1')
                net = slim.conv2d(net, dim, [ksize, ksize], activation_fn=None, scope='conv2')
                return net + x
            
        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                activation_fn=tf.nn.relu, padding='SAME', normalizer_fn=None,
                                weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                biases_initializer=tf.constant_initializer(0.0)):
                inp_pred = x
                inp_blur = x
                inp_pred = tf.stop_gradient(inp_pred)
                inp_all = tf.concat([inp_blur, inp_pred], axis=3, name='inp')
                # encoder
                conv1_1 = slim.conv2d(inp_all, 32, [5, 5], scope='enc1_1')
                conv1_2 = ResnetBlock(conv1_1, 32, 5, scope='enc1_2')
                conv1_3 = ResnetBlock(conv1_2, 32, 5, scope='enc1_3')
                conv1_4 = ResnetBlock(conv1_3, 32, 5, scope='enc1_4')
                conv2_1 = slim.conv2d(conv1_4, 64, [5, 5], stride=2, scope='enc2_1')
                conv2_2 = ResnetBlock(conv2_1, 64, 5, scope='enc2_2')
                conv2_3 = ResnetBlock(conv2_2, 64, 5, scope='enc2_3')
                conv2_4 = ResnetBlock(conv2_3, 64, 5, scope='enc2_4')
                conv3_1 = slim.conv2d(conv2_4, 128, [5, 5], stride=2, scope='enc3_1')
                conv3_2 = ResnetBlock(conv3_1, 128, 5, scope='enc3_2')
                conv3_3 = ResnetBlock(conv3_2, 128, 5, scope='enc3_3')
                conv3_4 = ResnetBlock(conv3_3, 128, 5, scope='enc3_4')
                deconv3_3 = ResnetBlock(conv3_4, 128, 5, scope='dec3_3')
                deconv3_2 = ResnetBlock(deconv3_3, 128, 5, scope='dec3_2')
                deconv3_1 = ResnetBlock(deconv3_2, 128, 5, scope='dec3_1')
                deconv2_4 = slim.conv2d_transpose(deconv3_1, 64, [4, 4], stride=2, scope='dec2_4')
                cat2 = deconv2_4 + conv2_4
                deconv2_3 = ResnetBlock(cat2, 64, 5, scope='dec2_3')
                deconv2_2 = ResnetBlock(deconv2_3, 64, 5, scope='dec2_2')
                deconv2_1 = ResnetBlock(deconv2_2, 64, 5, scope='dec2_1')
                deconv1_4 = slim.conv2d_transpose(deconv2_1, 32, [4, 4], stride=2, scope='dec1_4')
                cat1 = deconv1_4 + conv1_4
                deconv1_3 = ResnetBlock(cat1, 32, 5, scope='dec1_3')
                deconv1_2 = ResnetBlock(deconv1_3, 32, 5, scope='dec1_2')
                deconv1_1 = ResnetBlock(deconv1_2, 32, 5, scope='dec1_1')
                inp_pred = slim.conv2d(deconv1_1, 3, [5, 5], activation_fn=None, scope='dec1_0')

            return inp_pred
    
    @staticmethod
    def super_resolution(x, reuse=False):
        with tf.variable_scope(name_or_scope='super_resolution', reuse=reuse):
            x = C('super_resolution_conv1', x, 7, 3, 128, 1, relu=0.2)
            temp = x
            x = R('super_resolution_R1',x, 128, 8)
            x = C('super_resolution_conv2', x, 3, 128, 128, 1, relu=False)+temp
            return x
    
    @staticmethod
    def concatenate(x):
        sharp = nets.deblur(x)
        features = C('features_64', sharp, 5, 3, 64, 1, relu=False)
        super_resolved = nets.super_resolution(x)
        concat_vector = tf.concat([x, super_resolved, features], axis=3)
        gate_vector = nets.gate(concat_vector)+features
        return gate_vector, features, super_resolved, sharp
    
    @staticmethod
    def gate(x, reuse=False):
        with tf.variable_scope(name_or_scope='gate', reuse=reuse):
            x = C('gate_conv1', x, 3, 195, 64, 1, relu=0.2)
            x = C('gate_conv2', x, 1, 64, 64, 1, relu=False)+x
            return x
        
    @staticmethod
    def reconstruct(x, reuse=False):
        with tf.variable_scope(name_or_scope='reconstruct', reuse=reuse):
            x = R('reconstruct_R1',x, 64, 12)+x
            x = U('reconstruct_U1', x, 64, 64, conv_followed=False)
            scale2 = C('reconstruct_scale2_conv',x, 3, 64, 3, 1, relu=False)
            x = U('reconstruct_U2', x, 64, 64, conv_followed=False)
            x = C('reconstruct_conv1', x, 3, 64, 64, 1, relu=0.2)
            x = C('reconstruct_conv2', x, 3, 64, 3, 1, relu=False)
            return x, scale2
    
    @staticmethod
    def redundancy(scoremap, deblur_feature, sr_feature, reuse=False):
         with tf.variable_scope(name_or_scope='redundancy', reuse=reuse):
            return scoremap*deblur_feature+sr_feature
    
    @staticmethod
    def encapsulate(x, redundancy=False):
        scoremap, deblur_feature, super_resolved, sharp = nets.concatenate(x)
        if redundancy:
            input_lr = nets.redundancy(scoremap, deblur_feature, super_resolved)
        else:
            input_lr = scoremap
        output_hr, scale2 = nets.reconstruct(input_lr)
        return output_hr, scale2, deblur_feature, super_resolved, sharp
    

class assign:
    @staticmethod
    def get_deblur_var():
        trainablevars = tf.trainable_variables()
        deblurvars = [t for t in trainablevars if t.name.find('g_net')>=0]
        nondeblurvars = [t for t in trainablevars if t.name.find('g_net')<0]
        return deblurvars, nondeblurvars, trainablevars
        
    @staticmethod
    def get_assign_dict(deblurvars, pkl_path,printall=True):
        assign_dict=dict()
        graph = tf.get_default_graph()
        with open(pkl_path, 'rb') as pkl_file:
            weights=pk.load(pkl_file)
        for deblurvar in deblurvars:
            deblurvar_name = deblurvar.name
            if printall:
                print(deblurvar_name)
            assign_op = tf.assign(deblurvar, np.array(weights[deblurvar_name]))
            assign_dict[deblurvar_name] = assign_op
        return assign_dict
    
    @staticmethod
    def get_learning_rates(learning_rate1, learning_rate2, decay_step, decay_rate1, decay_rate2, now_epoch):
        decay1 = decay_rate1**(now_epoch/decay_step)
        decay2 = decay_rate2**(now_epoch/decay_step)
        return learning_rate1*decay1, learning_rate2*decay2
            
        
            
        
def readbatch(prefix_dirty = 'SRA/', prefix_sharp = 'SRB/', prefix_half='train_videos_scale2/', prefix_final='SRC/'):
    global batchcursor, d
    l = []
    overflow = False
    for cnt in range(batchsize):
        l.append((prefix_dirty + d[batchcursor], prefix_sharp + d[batchcursor], prefix_half+d[batchcursor],prefix_final+d[batchcursor]))
        batchcursor = (batchcursor+1)%len(d)
        if batchcursor == 0:
            overflow = True
    if overflow:
        random.shuffle(d)
    return l, overflow


def preprocessing(prefix_dirty = 'SRA/', prefix_sharp = 'SRB/', prefix_half='train_videos_scale2/', prefix_final='SRC/'):
    l, overflow = readbatch(prefix_dirty,prefix_sharp, prefix_half, prefix_final)
    dirtyarray = []
    sharparray = []
    halfarray = []
    finalarray = []
    k = random.randint(0, 3)
    for element in l:
        #print(element)
        pic_dirty = np.array(Image.open(element[0])).astype(np.float32)
        pic_sharp = np.array(Image.open(element[1])).astype(np.float32)
        pic_half = np.array(Image.open(element[2])).astype(np.float32)
        pic_final = np.array(Image.open(element[3])).astype(np.float32)
        x = random.randint(0, pic_dirty.shape[0] - sizeCropH)
        y = random.randint(0, pic_dirty.shape[1] - sizeCropW)
        pic_dirty = pic_dirty[x:x+sizeCropH, y:y+sizeCropW, :]
        pic_sharp = pic_sharp[x:x+sizeCropH, y:y+sizeCropW, :]
        pic_final= pic_final[x*scale:(x+sizeCropH)*scale, y*scale:(y+sizeCropW)*scale, :]
        pic_half = pic_half[x*scale//2:(x+sizeCropH)*scale//2, y*scale//2:(y+sizeCropW)*scale//2, :]
        if k!=0:
            pic_dirty = np.rot90(pic_dirty,k)
            pic_sharp = np.rot90(pic_sharp,k)
            pic_final = np.rot90(pic_final,k)
            pic_half = np.rot90(pic_half,k)
        dirtyarray.append(pic_dirty/255.0)
        sharparray.append(pic_sharp/255.0)
        halfarray.append(pic_half/255.0)
        finalarray.append(pic_final/255.0)
    dirtyarray = np.array(dirtyarray)
    sharparray = np.array(sharparray)
    halfarray = np.array(halfarray)
    finalarray = np.array(finalarray)
    return dirtyarray, sharparray, halfarray, finalarray, overflow


batchcursor = 0
if mode == 'train':
     with open('info.json') as f:
          d = list(json.load(f))
     random.shuffle(d)
initial_globalstep = 0
globalstep = tf.Variable(initial_globalstep, trainable=False)
blurred = tf.placeholder(tf.float32, [None, None, None, 3])
sharp_imgs = tf.placeholder(tf.float32, [None, None, None, 3])
scale2_imgs = tf.placeholder(tf.float32, [None, None, None, 3])
scale4_imgs = tf.placeholder(tf.float32, [None, None, None, 3])
lr1 = tf.placeholder(tf.float32)
lr2 = tf.placeholder(tf.float32)
final, scale2, deblur_feature, super_resolved, sharp = nets.encapsulate(blurred)
lambda1 = tf.placeholder(tf.float32)
lambda2 = tf.placeholder(tf.float32)
lambda3 = tf.placeholder(tf.float32)
if l1_or_l2 == 'l1':
    loss1 = tf.reduce_mean(tf.abs(sharp-sharp_imgs))
    loss2 = tf.reduce_mean(tf.abs(scale2-scale2_imgs))
    loss3 = tf.reduce_mean(tf.abs(final-scale4_imgs))
else:
    loss1 = tf.reduce_mean(tf.square(sharp-sharp_imgs))
    loss2 = tf.reduce_mean(tf.square(scale2-scale2_imgs))
    loss3 = tf.reduce_mean(tf.square(final-scale4_imgs))
print("loss")
deblurvars, nondeblurvars, trainablevars = assign.get_deblur_var()
G_loss = lambda1*loss1+lambda2*loss2+lambda3*loss3
original_optimizer1 = tf.train.AdamOptimizer(learning_rate = lr1)
optimizer1 = tf.contrib.estimator.clip_gradients_by_norm(original_optimizer1, clip_norm=5.0)
G_train1 = optimizer1.minimize(G_loss,global_step=globalstep, var_list=deblurvars)
original_optimizer2 = tf.train.AdamOptimizer(learning_rate = lr2)
optimizer2 = tf.contrib.estimator.clip_gradients_by_norm(original_optimizer2, clip_norm=5.0)
G_train2 = optimizer2.minimize(G_loss,global_step=globalstep, var_list=nondeblurvars)
G_train = tf.group(G_train1, G_train2)
PSNR = tf.reduce_mean(tf.image.psnr(final, scale4_imgs, max_val = 1.0))
ssim = tf.reduce_mean(tf.image.ssim(final, scale4_imgs, max_val = 1.0))
PSNR_deblur = tf.reduce_mean(tf.image.psnr(sharp, sharp_imgs, max_val = 1.0))
ssim_deblur = tf.reduce_mean(tf.image.ssim(sharp, sharp_imgs, max_val = 1.0))
PSNR_blur = tf.reduce_mean(tf.image.psnr(blurred, sharp_imgs, max_val = 1.0))
ssim_blur = tf.reduce_mean(tf.image.ssim(blurred, sharp_imgs, max_val = 1.0))
PSNR_scale2 = tf.reduce_mean(tf.image.psnr(scale2, scale2_imgs, max_val = 1.0))
ssim_scale2 = tf.reduce_mean(tf.image.ssim(scale2, scale2_imgs, max_val = 1.0))
error = final-scale4_imgs
print("assign")
try:
    if os.path.exists('weight.pkl'):
         assign_dict = assign.get_assign_dict(deblurvars, 'weights.pkl')
except:
    pass
def tensor_to_img(tensor):
    tensor = tensor*255.0
    tensor = np.clip(tensor, 0.0, 255.0)
    tensor = tensor.astype(np.uint8)
    image = Image.fromarray(tensor)
    return image

def tensor_to_img_reshape(tensor, shape):
    tensor = tensor*255.0
    tensor = np.clip(tensor, 0.0, 255.0)
    tensor = tensor.astype(np.uint8)
    image = Image.fromarray(tensor)
    if image.size != shape:
        image = image.resize(shape, Image.BICUBIC)
    return image

def get_time():
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

#pathtomodel = name+'/'
pathtomodel = name+'/'
saver = tf.train.Saver()
preacc = 0.0
preloss = 1e8


def test(sess, input_file_root, input_files, output_root, print_file=False, run_all=False, run_error=True, mode='final'):
    if not os.path.isdir(output_root):
        os.mkdir(output_root)
    start_time = time.time()
    for image_path in input_files:
        temp = image_path
        image_path = os.path.join(input_file_root, image_path)
        if print_file:
            print(image_path)
        image = Image.open(image_path)
        w, h= image.size
        origw = w
        origh = h
        if h%4 <= 2:
            h -= h%4
        else:
            h += 4-h%4
        if w%4 <= 2:
            w -= w%4
        else:
            w += 4-w%4
        if (w, h) != image.size:
            image = image.resize((w, h), Image.BICUBIC) 
        image = np.array(image).astype(np.float32)
        image = image[np.newaxis, :]
        image /= 255.0
        print(image.shape)
        if run_all or mode == 'all':
            sharp_img, scale2_img, final_img = sess.run([sharp, scale2, final], feed_dict={blurred:image})
            sharp_img = tensor_to_img(sharp_img[0])
            scale2_img = tensor_to_img(scale2_img[0])
            final_img = tensor_to_img(final_img[0])
            sharp_img.save(output_root+'/sharp_'+temp)
            scale2_img.save(output_root+'/scale2_'+temp)
            final_img.save(output_root+'/final_'+temp)
        elif mode=='final':
            neww, newh = origw*4, origh*4
            final_img = sess.run(final, feed_dict={blurred:image})
            final_img = tensor_to_img_reshape(final_img[0], (neww, newh))
            final_img.save(output_root+'/final_'+temp)
        elif mode == 'deblur':
            neww, newh = origw, origh
            sharp_img = sess.run(sharp, feed_dict={blurred:image})
            sharp_img = tensor_to_img_reshape(sharp_img[0], (neww, newh))
            sharp_img.save(output_root+'/sharp_'+temp)
        elif mode == 'middle':
            neww, newh = origw*2, origh*2
            middle_img = sess.run(scale2, feed_dict={blurred:image})
            middle_img = tensor_to_img_reshape(middle_img[0], (neww, newh))
            middle_img.save(output_root+'/middle_'+temp)

    end_time = time.time()
    print((end_time-start_time), len(input_files), (end_time-start_time)/len(input_files))

def testall(sess, rootdir, outdir, print_file=True, run_all=False, mode='final'):
    files = os.listdir(rootdir)
    test(sess, rootdir, files, outdir, print_file, run_all, mode=mode)

def write(string, path):
    with open(path, "a") as f:
         f.write(string+'\n')


with tf.Session() as sess:
    try:
        sess.run(tf.global_variables_initializer())
        try:
            saver.restore(sess, tf.train.latest_checkpoint(pathtomodel))
            write("Restore data successfully!", path)
        except:
            write("Error when reading model", path)
            write(traceback.format_exc(), path)
            for key in assign_dict.keys():
                sess.run(assign_dict[key])
            write("load_weights_successfully!", path)
            pass
        if mode == 'test':
            try:
                 mode = sys.argv[3]
            except:
                 mode = 'final'
            testall(sess, sys.argv[1], sys.argv[2], print_file=True, run_all=False, mode=mode)
            exit(0)
        while now_epoch < max_epochs:
            saver.save(sess, "%s/deblur"%pathtomodel, global_step=globalstep)
            write("now_epoch=%s"%now_epoch, path)
            feed_lr1, feed_lr2 = assign.get_learning_rates(learning_rate1, learning_rate2, decay_step, decay_rate1, decay_rate2, now_epoch)
            '''
            The config set below is used to finetune on REDS
            To train a REDS, please set:
            '''

            '''
            op = G_train
            if now_epoch < 40:
                feed_lambda1, feed_lambda2, feed_lambda3 = 0.5, 2.0, 1.0
            else:
                feed_lambda1, feed_lambda2, feed_lambda3 = 0.0, 1.0, 2.0
            '''
            if now_epoch < 8:
                feed_lambda1, feed_lambda2, feed_lambda3 = 1000.0, 5.0, 2.0
                op = G_train
            elif now_epoch >= 8 and now_epoch <= 12:
                op = G_train
                feed_lambda1, feed_lambda2, feed_lambda3 = 2.0, 1000.0, 5.0
            else:
                op = G_train
                feed_lambda1, feed_lambda2, feed_lambda3 = 2.0, 5.0, 1000.0
            overflow = False
            while not overflow:
                dirtyarray, sharparray, halfarray, finalarray, overflow = preprocessing()
                feed_dict = {lr1: feed_lr1, lr2: feed_lr2, blurred:dirtyarray, sharp_imgs: sharparray, scale2_imgs:halfarray, scale4_imgs:finalarray, lambda1:feed_lambda1, lambda2:feed_lambda2, lambda3:feed_lambda3}
                if step % 100 == 0:
                    _, run_G_loss, run_loss1, run_loss2, run_loss3, run_lambda1, run_lambda2, run_lambda3, run_PSNR, run_PSNR_deblur, run_PSNR_blur,run_PSNR_scale2, run_ssim, run_ssim_deblur, run_ssim_blur,run_ssim_scale2, g, run_lr1, run_lr2= sess.run([op, G_loss, loss1, loss2, loss3, lambda1, lambda2, lambda3, PSNR, PSNR_deblur, PSNR_blur,PSNR_scale2, ssim, ssim_deblur, ssim_blur, ssim_scale2, globalstep, lr1, lr2], feed_dict = feed_dict)
                    write(get_time()+' '+str(g)+' update_G(%s epoch), G_loss=%s=%s*%s+%s*%s+%s*%s, PSNR=%s, %s(%s, %s), ssim=%s, %s(%s, %s), step=%s, batchcursor=%s, run_lr=[%s,%s]'%(now_epoch, run_G_loss, feed_lambda1, run_loss1, feed_lambda2, run_loss2, feed_lambda3, run_loss3, run_PSNR, run_PSNR_scale2, run_PSNR_deblur, run_PSNR_blur, run_ssim, run_ssim_scale2, run_ssim_deblur, run_ssim_blur, step, batchcursor, run_lr1, run_lr2), path)
                    flag = False
                    if run_G_loss < preloss:
                        preloss = run_G_loss
                        saver.save(sess, "%s/deblur"%pathtomodel, global_step=globalstep)
                    elif run_ssim > preacc:
                        preacc = run_ssim
                        saver.save(sess, "%s/deblur"%pathtomodel, global_step=globalstep)

                else:
                    _ = sess.run(op, feed_dict=feed_dict)
                if not pyfile and step%40 == 0:
                    _, run_gene_img, run_G_loss, run_PSNR, run_ssim, run_middle, run_sharp= sess.run([G_train, final, G_loss, PSNR, ssim, scale2, sharp], feed_dict = feed_dict)
                    print(get_time(), 'update_G(%s epoch), G_loss=%s, PSNR=%s, ssim=%s, step=%s, batchcursor=%s'%(now_epoch, run_G_loss, run_PSNR, run_ssim, step, batchcursor))
                    run_gene_img = tensor_to_img(run_gene_img[-1])
                    final_img = tensor_to_img(finalarray[-1])
                    blurred_img = tensor_to_img(dirtyarray[-1])
                    sharp_img = tensor_to_img(run_sharp[-1])
                    run_middle = tensor_to_img(run_middle[-1])
                    middle_img = tensor_to_img(halfarray[-1])
                    #print(run_residual)
                    plt.imshow(run_gene_img)
                    plt.show()
                    plt.imshow(final_img)
                    plt.show()
                    plt.imshow(blurred_img)
                    plt.show()
                    plt.imshow(sharp_img)
                    plt.show()
                    plt.imshow(run_middle)
                    plt.show()
                    plt.imshow(middle_img)
                    plt.show()
                step += 1
            now_epoch += 1 
    except:
        pass


