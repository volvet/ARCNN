#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 13:49:26 2018

@author: volvetzhang
"""


import argparse
import tarfile
import os
import shutil
import random
import tensorflow as tf
import cv2
import numpy as np
from tqdm import tqdm
import math


class Data(object):
    def __init__(self, data_root, data_path, quality, block_size, block_channel, batch_size, stride):
        self.quality = quality
        self.data_root = data_root
        self.data_path = data_path
        self.batch_size = batch_size
        self.size = block_size
        self.channel = block_channel
        self.compressed_samples = []
        self.origin_samples = []
        self.indexs = []
        self.stride = stride
        self.pos = 0
        return
    
    def next_batch(self):
        X = np.zeros([self.batch_size, self.size, self.size, self.channel], np.float32)
        Y = np.zeros([self.batch_size, self.size, self.size, self.channel], np.float32)
        
        if self.pos > len(self.compressed_samples) - self.batch_size:
            return None, None
        for i in range(self.batch_size):
            X[i, :, :, :] = self.compressed_samples[self.indexs[i+self.pos]]/255.0
            Y[i, :, :, :] = self.origin_samples[self.indexs[i+self.pos]]/255.0
        
        self.pos += self.batch_size
        return X, Y
    
    def num_batches(self):
        return len(self.compressed_samples) // self.batch_size
    
    def shuffle(self):
        #random.shuffle(self.indexs)
        self.pos = 0
        return
    
    def load(self):
        filelist = os.listdir(self.data_root)
        for fn in filelist:
            if fn.endswith('tgz'):
                tgz_path = os.path.join(self.data_root, fn)
                #print(tgz_path)
                data_tgz = tarfile.open(tgz_path)
                data_tgz.extractall(path=self.data_root)
                data_tgz.close()
        
        Data.prepare(self)
        return
    
    def unload(self):
        filelist = os.listdir(self.data_root)
        for fn in filelist:
            path = os.path.join(self.data_root, fn)
            if os.path.isdir(path):
                shutil.rmtree(path)
        return

    def prepare(self):
        new_list = os.listdir(self.data_path)
        img_list = []
        for item in new_list:
            if item.endswith('jpg'):
                img_list.append(item)
        
        #print(img_list)
        #random.shuffle(img_list)
        for img_file in tqdm(img_list, desc='Prepare train samples'):
            origin_img = cv2.imread(self.data_path + img_file)
            
            cv2.imwrite('tmp.jpg', origin_img, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
            compressed_img = cv2.imread('tmp.jpg')
            os.remove('tmp.jpg')
            Data.build_train_samples(self, self.origin_samples, origin_img)
            Data.build_train_samples(self, self.compressed_samples, compressed_img)
            
        for i in range(len(self.origin_samples)):
            self.indexs.append(i)
            
        if self.batch_size > len(self.origin_samples):
            self.batch_size = len(self.origin_samples)
        
        return

    def build_train_samples(self, data, img):
        row_num = (img.shape[0] - self.size) // self.stride + 1
        col_num = (img.shape[1] - self.size) // self.stride + 1
        
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                img[y, x, 0] = 0.299*img[y, x, 0] + 0.587*img[y, x, 1] + 0.114*img[y, x, 2]
        
        for y in range(row_num):
            for x in range(col_num):
                x_start = x*self.stride
                x_end = x_start+self.size
                y_start = y*self.stride
                y_end = y_start+self.size
                block = img[y_start:y_end, x_start:x_end, 0:1]
                data.append(block)
        return

class ARCNN(object):

    def __init__(self, config):
        self.config = config
        return

    def train(self, train_data, test_data):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.config.epoch):
                train_data.shuffle()
                test_data.shuffle()
                for j in tqdm(range(train_data.num_batches()), desc='epoch ' + str(i)):
                    batch_x, batch_y = train_data.next_batch()
                    _, cost, origin_cost = sess.run([self.optimizer, self.loss, self.origin_loss], 
                                                       feed_dict={self.X: batch_x, self.Y: batch_y})
                
                
                #test
                test_x, test_y = test_data.next_batch()
                cost, origin_cost = sess.run([self.loss, self.origin_loss], 
                                             feed_dict={self.X: test_x, self.Y: test_y})
                PSNR = 10.0 * math.log(1.0/cost)/math.log(10.0)
                PSNR_original = 10.0 * math.log(1.0/origin_cost)/math.log(10.0)
                print('\nEpoch: ', i, 'PSNR: ', PSNR, '/', PSNR_original)
            
        return
    
    def build_net(self):
        shapes = {
            'in': [None, config.block_size, config.block_size, config.block_channel],
            'extract': [9, 9, config.block_channel, 64],
            'enhance': [7, 7, 64, 32],
            'restore': [1, 1, 32, 16],
            'recon': [5, 5, 16, config.block_channel]
        }
        
        self.X = tf.placeholder(tf.float32, shapes['in'])
        self.Y = tf.placeholder(tf.float32, shapes['in'])
        
        with tf.name_scope('feature_extraction'):
            self.W1 = ARCNN.weight_variable(self, shapes['extract'], stddev=0.001)
            conv1 = tf.nn.conv2d(self.X, self.W1, strides=[1,1,1,1], padding='SAME')
            self.b1 = ARCNN.bias_variable(self, [shapes['extract'][-1]], stddev=0.001)
            conv1 = tf.nn.relu(tf.nn.bias_add(conv1, self.b1))
        
        with tf.name_scope('featue_enhancement'):
            self.W2 = ARCNN.weight_variable(self, shapes['enhance'], stddev=0.001)
            conv2 = tf.nn.conv2d(conv1, self.W2, strides=[1,1,1,1], padding='SAME')
            self.b2 = ARCNN.bias_variable(self, [shapes['enhance'][-1]], stddev=0.001)
            conv2 = tf.nn.relu(tf.nn.bias_add(conv2, self.b2))
            
        with tf.name_scope('restore'):
            self.W3 = ARCNN.weight_variable(self, shapes['restore'], stddev=0.001)
            conv3 = tf.nn.conv2d(conv2, self.W3, strides=[1,1,1,1], padding='SAME')
            self.b3 = ARCNN.bias_variable(self, [shapes['restore'][-1]], stddev=0.001)
            conv3 = tf.nn.relu(tf.nn.bias_add(conv3, self.b3))
            
        with tf.name_scope('reconstruction'):
            self.W4 = ARCNN.weight_variable(self, shapes['recon'], stddev=0.001)
            conv4 = tf.nn.conv2d(conv3, self.W4, strides=[1,1,1,1], padding='SAME')
            self.b4 = ARCNN.bias_variable(self, [shapes['recon'][-1]], stddev=0.001)
            self.recon = tf.nn.bias_add(conv4, self.b4)
            
        offset_height = int(config.block_size - config.valid_size)//2
        offset_width = int(config.block_size - config.valid_size)//2
        
        mid_Y = tf.strided_slice(self.Y, [0, offset_height, offset_width, 0], 
                                 [-1, offset_height+config.valid_size, offset_width+config.valid_size, config.block_channel])
        mid_X = tf.strided_slice(self.X, [0, offset_height, offset_width, 0], 
                                 [-1, offset_height+config.valid_size, offset_width+config.valid_size, config.block_channel])
        mid_recon = tf.strided_slice(self.recon, [0, offset_height, offset_width, 0], 
                                 [-1, offset_height+config.valid_size, offset_width+config.valid_size, config.block_channel])
        
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.square(mid_recon - mid_Y))
        
        with tf.name_scope('origin_loss'):
            self.origin_loss = tf.reduce_mean(tf.square(mid_X - mid_Y))
            
        self.optimizer = tf.train.RMSPropOptimizer(config.learning_rate)
        gradients = self.optimizer.compute_gradients(self.loss)
        for i, (g, v) in enumerate(gradients):
            if g is not None:
                gradients[i] = (tf.clip_by_value(g, -config.gradient_clip, config.gradient_clip), v)
        self.optimizer = self.optimizer.apply_gradients(gradients)
        return
    
    def weight_variable(self, shape, stddev=0.01, name=None):
        initial = tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32)
        return tf.Variable(initial, name=name, dtype=tf.float32)
    
    def bias_variable(self, shape, stddev=0.01, name=None):
        initial = tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32)
        return tf.Variable(initial, name=name, dtype=tf.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--quality', type=int, default=60)
    parser.add_argument('--test_size', type=int, default=256)
    parser.add_argument('--block_size', type=int, default=32)
    parser.add_argument('--block_channel', type=int, default=1)
    parser.add_argument('--valid_size', type=int, default=20)
    parser.add_argument('--data_root', type=str, default='../data/')
    parser.add_argument('--data_path', type=str, default='../data/BSR/BSDS500/data/images/train/')
    parser.add_argument('--test_path', type=str, default='../data/BSR/BSDS500/data/images/test/')
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--gradient_clip', type=int, default=1)
    parser.add_argument('--train_stride', type=int, default=10)
    parser.add_argument('--test_stride', type=int, default=32)
    
    config = parser.parse_args()
    
    print(config)
    
    train_data = Data(config.data_root, config.data_path, config.quality, config.block_size, config.block_channel, 
                config.batch_size, config.train_stride)
    
    test_data = Data(config.data_root, config.test_path, config.quality, config.block_size, config.block_channel, 
                config.test_size, config.test_stride)
    
    train_data.load()
    test_data.load()
       
    arcnn = ARCNN(config)
    arcnn.build_net()
    arcnn.train(train_data, test_data)
    
    train_data.unload()
    
