from __future__ import print_function

import os
import time
import random
import vgg
from PIL import Image
import tensorflow as tf
import numpy as np
#import unet_RS
import LH_net3
from utils import *

def concat(layers):
    return tf.concat(layers, axis=3)

def DecomNet(input_im):
    with tf.variable_scope('DecomNet', reuse=tf.AUTO_REUSE):
        out = LH_net3.fusion3(input_im)
    return out
class lowlight_enhance(object):
    def __init__(self, sess):
        self.sess = sess
        # build the model
        self.input_low = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low')
        self.input_high = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high')

        output = DecomNet(self.input_low)#输入低光图像
        self.output_S = output

        #metric
        self.psnr_R = tf.reduce_mean(tf.image.psnr(self.output_S * 255, self.input_high* 255, max_val=255))
        self.ssim_R = tf.reduce_mean(tf.image.ssim(self.output_S * 255, self.input_high* 255, max_val=255))
        # loss

        #Decom
        self.recon_loss_high = tf.reduce_mean(tf.abs(self.output_S - self.input_high))
        self.loss_ssim = 1-tf.reduce_mean(tf.image.ssim(self.output_S * 255, self.input_high* 255, max_val=255))
        self.loss_color = tf.reduce_mean(tf.abs(self.color_s(output) - self.color_s(self.input_high)))
        self.recon_local = tf.reduce_mean(tf.square(self.output_S - self.input_high))
        self.loss_tv = tf.reduce_mean(tf.square(self.tv_grad(self.output_S) -self.tv_grad(self.input_high)))
        # 2) content loss
        CONTENT_LAYER = 'relu5_4'
        enhanced_vgg = vgg.net('vgg_pretrained/imagenet-vgg-verydeep-19.mat', vgg.preprocess(self.output_S * 255))
        dslr_vgg = vgg.net('vgg_pretrained/imagenet-vgg-verydeep-19.mat', vgg.preprocess(self.input_high * 255))
        #content_size = utils_best._tensor_size(dslr_vgg[CONTENT_LAYER]) * batch_size
        self.loss_content = 2 * tf.nn.l2_loss(enhanced_vgg[CONTENT_LAYER] - dslr_vgg[CONTENT_LAYER]) #/ content_size
        #self.loss_final =self.recon_local
        #self.loss_final = self.recon_loss_high+self.recon_local
        #self.loss_final = 0.32*self.recon_loss_high+self.loss_ssim+0.01*self.loss_color+0.68*self.recon_local+0.01*self.loss_tv
        #self.loss_final = 0.1*self.recon_loss_high+self.loss_ssim+0.01*self.loss_color+0.9*self.recon_local+0.01*self.loss_tv
        #self.loss_final = 0.68*self.recon_loss_high+0.25*self.loss_ssim+0.05*self.loss_color+2*self.recon_local+0.05*self.loss_tv
        #self.loss_final = 0.32*self.recon_loss_high+self.loss_ssim+0.05*self.loss_color+0.68*self.recon_local+0.05*self.loss_tv
        #self.loss_final =self.recon_local+self.recon_loss_high+0.1*self.loss_tv
        self.loss_final =self.recon_local+0.01*self.loss_tv+self.recon_loss_high+0.05*self.loss_color
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')

        self.var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
        self.train_op_Decom = optimizer.minimize(self.loss_final, var_list = self.var_Decom)
        self.sess.run(tf.global_variables_initializer())
        self.saver_Decom = tf.train.Saver(var_list = self.var_Decom, max_to_keep=0)
        print("[*] Initialize model successfully...")


    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = tf.reshape(tf.constant([[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])
        self.smooth_kernel_y = tf.transpose(self.smooth_kernel_x, [1, 0, 2, 3])

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        return tf.abs(tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))#154
    def tv_grad(self,input_R):
        input_R = tf.image.rgb_to_grayscale(input_R)
        return self.gradient(input_R,"x")+self.gradient(input_R,"y")
    def tv_grad2(self,input_I):
        return self.gradient(input_I,"x")+self.gradient(input_I,"y")
    def ave_gradient(self, input_tensor, direction):
        return tf.layers.average_pooling2d(self.gradient(input_tensor, direction), pool_size=3, strides=1, padding='SAME')

    def smooth(self, input_I, input_R):
        input_R = tf.image.rgb_to_grayscale(input_R)
        return tf.reduce_mean(self.gradient(input_I, "x") * tf.exp(-10 * self.ave_gradient(input_R, "x")) + self.gradient(input_I, "y") * tf.exp(-10 * self.ave_gradient(input_R, "y")))

    def color_s(self, input_c): 
        max_channel = tf.reduce_max(input_c,axis=-1,keep_dims=True)
        min_channel = tf.reduce_min(input_c,axis=-1,keep_dims=True)
        res_channel = (max_channel- min_channel)/(max_channel+0.01)
        return res_channel
    def evaluate(self, epoch_num, eval_low_data, eval_high_data, sample_dir, train_phase):
        print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))
        psnr1=0.0
        ssim1=0.0
        psnr3=0.0
        ssim3=0.0
        avg_psnr=0.0
        avg_ssim=0.0
        i=0
        psnr_R=self.psnr_R
        ssim_R=self.ssim_R

        for idx in range(len(eval_low_data)):
            input_low_eval = np.expand_dims(eval_low_data[idx], axis=0)
            input_high_eval = np.expand_dims(eval_high_data[idx], axis=0)
            if train_phase == "Decom_total_LH":
                result_1= self.sess.run(self.output_S, feed_dict={self.input_low: input_low_eval})
                #save_images(os.path.join(sample_dir, 'eval_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), result_1)
                psnr,ssim= self.sess.run([psnr_R,ssim_R],feed_dict={self.input_low: input_low_eval,self.input_high: input_high_eval})
                psnr1+=psnr
                ssim1+=ssim
        avg_psnr=psnr1/(len(eval_low_data))
        avg_ssim=ssim1/(len(eval_low_data))
        print("psnr: %.4f,ssim: %.4f" \
                   % (avg_psnr,avg_ssim))
            #write txt
        f = open("psnr_TV_01.txt", "a+")
        #f = open("psnr_color_0.05.txt", "a+")
        #f = open("psnr_total_LH9.txt", "a+")
        print("i %d---PSNR %.4f , SSIM  %.4f ---" % (epoch_num,avg_psnr,avg_ssim), file=f)
        f.write('\n')
        f.close()
        i=i+1

    def train(self, train_low_data, train_high_data, eval_low_data, eval_high_data, batch_size, patch_size, epoch, lr, sample_dir, ckpt_dir, eval_every_epoch, train_phase):
        assert len(train_low_data) == len(train_high_data)
        print("total train number")
        print(len(train_low_data))
        numBatch = len(train_low_data) // int(batch_size)

        # load pretrained model
        if train_phase == "Decom_total_LH":
            train_op = self.train_op_Decom
            train_loss = self.loss_final
            loss1=0.68*self.recon_loss_high
            loss2=0.25*self.loss_ssim
            loss3=0.05*self.loss_color
            loss4=2*self.recon_local
            loss5=0.05*self.loss_tv
            saver = self.saver_Decom
        load_model_status, global_step = self.load(saver, ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")

        print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))

        start_time = time.time()
        image_id = 0

        for epoch in range(start_epoch, epoch):
            for batch_id in range(start_step, numBatch):
                # generate data for a batch
                batch_input_low = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                batch_input_high = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                for patch_id in range(batch_size):
                    h, w, _ = train_low_data[image_id].shape
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)
            
                    rand_mode = random.randint(0, 7)
                    batch_input_low[patch_id, :, :, :] = data_augmentation(train_low_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
                    batch_input_high[patch_id, :, :, :] = data_augmentation(train_high_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
                    
                    image_id = (image_id + 1) % len(train_low_data)
                    if image_id == 0:
                        tmp = list(zip(train_low_data, train_high_data))
                        random.shuffle(list(tmp))
                        train_low_data, train_high_data  = zip(*tmp)

                # train
                _, loss,loss11,loss12,loss13,loss14,loss15 = self.sess.run([train_op, train_loss,loss1,loss2,loss3,loss4,loss5], feed_dict={self.input_low: batch_input_low, \
                                                                           self.input_high: batch_input_high, \
                                                                           self.lr: lr[epoch]})

                print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.4f, loss1: %.4f, loss2: %.4f, loss3: %.4f, loss4: %.4f, loss5: %.4f" \
                      % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss,loss11,loss12,loss13,loss14,loss15))
                iter_num += 1

            # evalutate the model and save a checkpoint file for it
            if (epoch + 1) % eval_every_epoch == 0:
                self.evaluate(epoch + 1, eval_low_data,eval_high_data, sample_dir=sample_dir, train_phase=train_phase)
                self.save(saver, iter_num, ckpt_dir, "RetinexNet_LH9-%s" % train_phase)

        print("[*] Finish training for phase %s." % train_phase)

    def save(self, saver, iter_num, ckpt_dir, model_name):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        print("[*] Saving model %s" % model_name)
        saver.save(self.sess, \
                   os.path.join(ckpt_dir, model_name), \
                   global_step=iter_num)

    def load(self, saver, ckpt_dir):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(ckpt_dir)
            try:
                global_step = int(full_path.split('/')[-1].split('-')[-1])
            except ValueError:
                global_step = None
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            print("[*] Failed to load model from %s" % ckpt_dir)
            return False, 0

    def test(self, test_low_data, test_high_data, test_low_data_names, test_high_data_names,save_dir, decom_flag):
        tf.global_variables_initializer().run()

        print("[*] Reading checkpoint...")
        #load_model_status_Decom, _ = self.load(self.saver_Decom, './checkpoint_test/21.3192_0.8193')
        load_model_status_Decom, _ = self.load(self.saver_Decom, './checkpoint/mseOnly')
        #load_model_status_Relight, _ = self.load(self.saver_Relight, './model/Relight')
        #if load_model_status_Decom and load_model_status_Relight:
            #print("[*] Load weights successfully...")
        
        print("[*] Testing...")
        psnr1=0.0
        ssim1=0.0
        avg_psnr=0.0
        avg_ssim=0.0
        psnr_R=self.psnr_R
        ssim_R=self.ssim_R
        #test_low_data_names.sort()
        for idx in range(len(test_low_data)):
            [_, name] = os.path.split(test_low_data_names[idx])
            suffix = name[name.find('.') + 1:]
            name = name[:name.find('.')]
            #print(name)
            input_low_test = np.expand_dims(test_low_data[idx], axis=0)
            input_high_test = np.expand_dims(test_high_data[idx], axis=0)
            #[R_low, I_low] = self.sess.run([self.output_R_low, self.output_I_low], feed_dict = {self.input_low: input_low_test})
            output=self.sess.run(self.output_S,feed_dict = {self.input_low: input_low_test,self.input_high: input_high_test})            
            save_images(os.path.join(save_dir, name + "_TV_01_.png"), output)
            psnr,ssim= self.sess.run([psnr_R,ssim_R],feed_dict={self.input_low: input_low_test,self.input_high: input_high_test})
            psnr1+=psnr
            ssim1+=ssim

            print("%s, psnr: %.4f,ssim: %.4f" \
                   % (name,psnr,ssim))
            f = open("psnr_test_real.txt", "a+")
            print("%s---PSNR %.4f , SSIM  %.4f ---" % (name,psnr,ssim), file=f)
            f.write('\n')
            f.close()
        avg_psnr=psnr1/(len(test_low_data))
        avg_ssim=ssim1/(len(test_low_data))
        print("psnr: %.4f,ssim: %.4f" \
                   % (avg_psnr,avg_ssim))


