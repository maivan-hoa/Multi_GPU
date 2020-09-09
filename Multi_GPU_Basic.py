# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 09:32:25 2020

@author: Mesoco
"""
import tensorflow.compat.v1 as tf
import numpy as np
import datetime

tf.disable_eager_execution()

# Num of multiplications to perform
n = 10

# Create random large matrix
A = np.random.rand(10000, 10000).astype('float32')
B = np.random.rand(10000, 10000).astype('float32')

# Create a graph to store results
c1 = []
c2 = []

def matpow(M, n):
    if n<1:
        return M
    else: 
        return tf.matmul(M, matpow(M, n-1))  # đệ quy nhân ma trận
    

'''
Placeholders are nodes whose value is fed in at execution time. In fact, you can build the graph without needing the data
--> phải tạo dict chỉ định dữ liệu khi thực thi (khi gọi sess.run())
'''


'''
Single GPU computing
'''

with tf.device('/gpu:0'):
    a = tf.placeholder(tf.float32, [10000, 10000])
    b = tf.placeholder(tf.float32, [10000, 10000])
    # Compute and store results in c1
    c1.append(matpow(a, n))
    c1.append(matpow(b, n))
    
with tf.device('/cpu:0'):
    sum = tf.add_n(c1) # add element-wise
    
t1 = datetime.datetime.now()

with tf.Session(config= tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(sum, {a: A, b:B})

t2 = datetime.datetime.now()


'''
Multi GPU computing
'''

with tf.device('/gpu:0'):
    # Compute A^n and store result in c2
    a = tf.placeholder(tf.float32, [10000, 10000])
    c2.append(matpow(a, n))
    
with tf.device('/gpu:1'):
    # Compute B^n and store result in c2
    b = tf.placeholder(tf.float32, [10000, 10000])
    c2.append(matpow(b, n))

with tf.device('/cpu:0'):
    sum.add_n(c2)
    
t3 = datetime.datetime.now()

with tf.Session(config= tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(sum, {a: A, b:B})

t4 = datetime.datetime.now()

print("Single GPU computation time: "+ str(t2 - t1))
print("Multi GPU computation time: "+ str(t4 - t3))











































