import tensorflow as tf 
import numpy as np 
from utils import * 

class linear(object):
    def __init__(self, information):
        params = information['params']
        self.NT = 2*params['K']
        self.NR = 2*params['N']
        self.batch_size = information['batchsize_placeholder']

    def MMNet(self, shatt, rt, features):
        batch_size = self.batch_size
        
        Wr = tf.Variable(tf.random_normal(shape=[1, self.NT//2, self.NR//2], mean=0., stddev=0.01))
        Wi = tf.Variable(tf.random_normal([1, self.NT//2, self.NR//2], 0., 0.01))
        W = tf.concat([tf.concat([Wr, -Wi], axis=2), tf.concat([Wi, Wr], axis=2)], axis=1)
        W = tf.tile(W, [batch_size, 1, 1])
        H = features['H'] 
        zt = shatt + batch_matvec_mul(W, rt)
         
        helper = {'W': W, 'I_WH': tf.eye(self.NT, batch_shape=[batch_size])-tf.matmul(W, H)}
        return zt, helper


    def MMNet_iid(self, shatt, rt, features):
        batch_size = self.batch_size
        H = features['H']
        W = tf.matrix_transpose(H)
        helper = {'W': W}
        zt = shatt + tf.square(tf.Variable(1.)) * batch_matvec_mul(W, rt) 
        return zt, helper

    def Ht(self, shatt, rt, features):
        batch_size = self.batch_size
        H = features['H']
        W = tf.matrix_transpose(H)
        helper = {'W': W}
        zt = shatt + batch_matvec_mul(W, rt)
        return zt, helper

    def lin_DetNet(self, shatt, rt, features):
        batch_size = self.batch_size
        H = features['H']
        W = tf.matrix_transpose(H)
        helper = {'W': W}
        Theta1 = tf.Variable(tf.random_normal([1,self.NT, self.NT], 0., 0.001))
        theta2 = tf.Variable(1.)
        theta3 = tf.Variable(tf.random_normal([1,self.NT], 0., 0.001))
        
        zt = shatt + batch_matvec_mul(tf.matmul(tf.tile(Theta1, [batch_size,1,1]), W), features['y'] - batch_matvec_mul(H, shatt)) + theta3

        return zt, helper
    
    def identity(self, shatt, rt, features):
        helper = {'W': [[0.]]}
        zt = shatt
        return zt, helper

    def OAMPNet(self, shatt, rt, features):
        noise_sigma = features['noise_sigma']
        H = features['H']
        HTH = tf.matmul(H, H, transpose_a=True)
        HHT = tf.matmul(H, H, transpose_b=True)
        batch_size = self.batch_size
        gamma_t = tf.Variable(1., trainable=True)
        v2_t = tf.divide((tf.reduce_sum(tf.square(rt), axis=1, keep_dims=True) - self.NR * tf.square(noise_sigma) / 2.), tf.expand_dims(tf.trace(HTH), axis=1))
        v2_t = tf.maximum(v2_t , 1e-9)
        v2_t = tf.expand_dims(v2_t, axis=2)
        lam = tf.square(tf.expand_dims(noise_sigma, axis=2))/2. 
        inv_term = tf.matrix_inverse(v2_t * HHT +  lam * tf.eye(self.NR, batch_shape=[tf.shape(H)[0]]))
        What_t = v2_t * tf.matmul(H, inv_term, transpose_a=True)
        W_t = self.NT * What_t / tf.reshape(tf.trace(tf.matmul(What_t, H)), [-1,1,1])
        zt = shatt + gamma_t * batch_matvec_mul(W_t, rt)
        helper = {'W': gamma_t * W_t, 'I_WH': tf.eye(self.NT, batch_shape=[batch_size])-tf.matmul(gamma_t*W_t, H)}
        return zt, helper

