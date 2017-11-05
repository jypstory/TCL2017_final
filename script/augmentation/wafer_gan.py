#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:59:40 2017
@author: jyp
"""

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from os import path

## 옵션 설정
total_epoch = 100
batch_size = 50
learning_rate = 0.0002
n_hidden = 30
n_input = 100*100
n_noise = 32
n_width = 100
n_height = 100

total_batch = 100

## input data 구성
path_src = "../../wafer_data/bad_image/line"
path_out = "../../wafer_data/bad_image/gan_line"

abspath_src = path.abspath(path_src)
abspath_out = path.abspath(path_out)

## to delete .Ds_store file
def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]

print(mylistdir(path_src))

file_paths = [path.join(abspath_src, name) for name in mylistdir(path_src) if path.isfile(path.join(abspath_src, name))]

img_list = []
cnt = 0
img = ""
for file in file_paths:
    if cnt == 0:
        img = Image.open(file).convert("L")
        img = img.resize((n_width, n_height), Image.ANTIALIAS)
        arr = np.array(img)
        img_list = arr.reshape(1, n_width * n_height)

    img = Image.open(file).convert("L")
    img = img.resize((n_width, n_height), Image.ANTIALIAS)
    #img = scale( img, axis=0, with_mean=True, with_std=True, copy=True)
    arr = np.array(img)
    arr = arr.reshape(1, n_width*n_height)
    img_list = np.append(img_list, arr, axis=0)
    cnt = cnt + 1

batch_image_list = []
cnt = 0
print(len(img_list))

for i in range(batch_size):
    if cnt == 0:
        batch_image_list = img_list
    batch_image_list = np.append(batch_image_list, img_list, axis=0)
    cnt = cnt + 1
batch_image_list = batch_image_list[:batch_size]


## 신경망 모델구성
X = tf.placeholder(tf.float32, [ None, n_input])
Z = tf.placeholder(tf.float32, [None, n_noise])

G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev = 0.01))
G_b1 = tf.Variable(tf.zeros([n_hidden]))
G_W2 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev = 0.01))
G_b2 = tf.Variable(tf.zeros([n_input]))

D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
D_b1 = tf.Variable(tf.zeros([n_hidden]))
D_W2 = tf.Variable(tf.random_normal([n_hidden,1], stddev=0.01))
D_b2 = tf.Variable(tf.zeros([1]))

def generator(noise_z):
    hidden = tf.nn.relu(
        tf.matmul(noise_z, G_W1)
    )

    output = tf.nn.sigmoid(
        tf.matmul(hidden, G_W2) + G_b2
    )

    return output

def discriminator(inputs):
    hidden = tf.nn.relu(
        tf.matmul(inputs, D_W1) + D_b1
    )
    output = tf.nn.sigmoid(tf.matmul(hidden, D_W2)+ D_b2)

    return output

def get_noise(batch_size, n_noise):
    return np.random.normal(size=(batch_size, n_noise))

G = generator(Z)
D_gene = discriminator(G)
D_real = discriminator(X)


loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1-D_gene))
loss_G = tf.reduce_mean(tf.log(D_gene))

D_var_list = [D_W1, D_b1, D_W2, D_b2]
G_var_list = [G_W1, G_b1, G_W2, G_b2]

train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D, var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G, var_list=G_var_list)


## 신경망 모델 학습
sess = tf.Session()
sess.run(tf.global_variables_initializer())

loss_val_D, loss_val_G = 0,0

for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs = batch_image_list
        noise = get_noise(batch_size, n_noise)

        _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Z: noise})
        _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Z: noise})

    print('Epoch', '%04d' % epoch,
          'D loss: {:.4}'.format(loss_val_D),
          'G loss: {:.4}'.format(loss_val_G))


    ## 확인용 이미지 생성
    if epoch == 0 or (epoch + 1) % 10 ==0:
        sample_size = 1
        noise = get_noise(sample_size, n_noise)
        samples = sess.run(G, feed_dict={Z: noise})
        samples = samples.reshape(n_width, n_height)
        formatted = (samples * 255 / np.max(samples)).astype('uint8')
        img = Image.fromarray(formatted)
        img_filename = abspath_out+'/' + str(epoch) + '.png'
        img.save(img_filename)

print('최적화 완료')