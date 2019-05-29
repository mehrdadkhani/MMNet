import tensorflow as tf
import numpy as np
from utils import *

class generator(object):
    def __init__(self, params, batch_size): 
        modulation = params['modulation']
        self.batch_size = batch_size
        if params['data']:
            self.Hdataset_powerdB = params['Hdataset_powerdB']
        else:
            self.Hdataset_powerdB = np.inf
        self.NT = params['K']
        self.NR = params['N']
        self.mod_scheme = modulation.split('_')[0]
        self.alpha4 = 1.
        self.alpha16 = 1.
        self.alpha64 = 1.

        self.mix_mapping = None

        if self.mod_scheme == 'QAM':
            self.mod_n = int(modulation.split('_')[1])
        if self.mod_scheme == 'MIX':
            self.mod_n = int(modulation.split('_')[1])
        else:
            print("The modulation is not supported yet")

        self.constellation, _ = self.QAM_N_const()


    def QAM_N_const(self, n=None):
        if n==None:
            n = self.mod_n
        constellation = np.linspace(-np.sqrt(n)+1, np.sqrt(n)-1, np.sqrt(n))
        alpha = np.sqrt((constellation ** 2).mean())
        constellation /= (alpha * np.sqrt(2))
        constellation = tf.Variable(constellation, trainable=False, dtype=tf.float32)
        return constellation, float(alpha)

    def QAM_N_ind(self): 
        indices_QAM = tf.random_uniform(
            shape=[self.batch_size, 2*self.NT], 
            minval=0, 
            maxval=np.sqrt(self.mod_n),
            dtype=tf.int32)
        if self.mod_scheme == 'MIX':
            print("MIXED CONSTELLATION MODE ACTIVATED")
            mod_names = tf.random_uniform(
            shape=[self.batch_size, self.NT], 
            minval=0, 
            maxval=3,
            dtype=tf.int32)
            mod_names = tf.tile(mod_names, [1,2])
            # mod_names: 0-->QAM64, 1-->QAM16, 2-->QAM64
            mapping = tf.one_hot(mod_names, depth=3, dtype=tf.int32)
            indices_QAM64 = tf.random_uniform(
                shape=[self.batch_size, 2*self.NT,1], 
                minval=0, 
                maxval=8,
                dtype=tf.int32)
            indices_QAM16 = tf.random_uniform(
                shape=[self.batch_size, 2*self.NT,1], 
                minval=0, 
                maxval=4,
                dtype=tf.int32)
            indices_QAM4 = tf.random_uniform(
                shape=[self.batch_size, 2*self.NT,1], 
                minval=0, 
                maxval=2,
                dtype=tf.int32)
            indices_QAM = mapping * tf.concat([indices_QAM64,tf.gather([0,2,5,7],indices_QAM16),indices_QAM4*7], axis=2)
            indices_QAM = tf.reduce_sum(indices_QAM,axis=2)
            self.mix_mapping = mapping

        return indices_QAM

    def random_indices(self):
        '''Generate random constellation symbol indices'''
        if  self.mod_scheme == 'QAM':
            indices = self.QAM_N_ind()
        elif self.mod_scheme == 'MIX':
            indices = self.QAM_N_ind()
        else: print("Modulation is not supported")
        return indices
    
    def modulate(self, indices):
        x = tf.gather(self.constellation, indices)
        return x   
            
    def exp_correlation(self, rho):
        ranget = np.reshape(np.arange(1,self.NT+1), (-1,1))
        ranger = np.reshape(np.arange(1,self.NR+1), (-1,1))
        Rt = rho ** (np.abs(ranget - ranget.T))
        Rr = rho ** (np.abs(ranger - ranger.T))
        R1 = LA.sqrtm(Rr)
        R2 = LA.sqrtm(Rt)
        return R1, R2
        
    def channel(self, x, snr_db_min, snr_db_max, H, dataset_flag, correlation_flag):
        '''Simulate transmission over a random or static channel with SNRs draw uniformly
           from the range [snr_db_min, snr_db_max] 
        '''
        # Channel Matrix
        if dataset_flag:
            H = H
        else:
            print("iid channels are generated")
            Hr = tf.random_normal(shape=[self.batch_size, self.NR, self.NT], stddev=1./np.sqrt(2.*self.NR), dtype=tf.float32)
            Hi = tf.random_normal(shape=[self.batch_size, self.NR, self.NT], stddev=1./np.sqrt(2.*self.NR), dtype=tf.float32)
            h1 = tf.concat([Hr, -1. * Hi], axis=2)
            h2 = tf.concat([Hi, Hr], axis=2)
            H = tf.concat([h1, h2], axis=1)
            self.Hdataset_powerdB = 0.
        # Channel Noise
        snr_db = tf.random_uniform(shape=[self.batch_size, 1], minval=snr_db_min, maxval=snr_db_max, dtype=tf.float32)
        if self.mod_scheme == 'MIX':
            print(self.constellation)
            powQAM4 = 10. * tf.log((tf.square(self.constellation[0]) + tf.square(self.constellation[7]))/2.) / tf.log(10.)
            powQAM16 = 10. * tf.log((self.constellation[0] ** 2 + self.constellation[2] ** 2 + self.constellation[5] ** 2 + self.constellation[7] ** 2)/4.)/tf.log(10.)
            snr_adjusments = tf.cast(self.mix_mapping, tf.float32) * [[[0,-powQAM16-7.,-powQAM4-14.]]]
            snr_adjusments = tf.reduce_sum(snr_adjusments, axis=2) 
            snr_adjusments = tf.expand_dims(snr_adjusments, axis=1)
            H = H * (10. ** (snr_adjusments/10.))
            print('seessssse', H)
        
        wr = tf.random_normal(shape=[self.batch_size, self.NR], stddev=1./np.sqrt(2.), dtype=tf.float32, name='w')
        wi = tf.random_normal(shape=[self.batch_size, self.NR], stddev=1./np.sqrt(2.), dtype=tf.float32, name='w')
        w = tf.concat([wr, wi], axis=1)
        # SNR
        print("Controlling the SNR")
        H_powerdB = 10. * tf.log(tf.reduce_mean(tf.reduce_sum(tf.square(H), axis=1), axis=0)) / tf.log(10.)
        average_H_powerdB = tf.reduce_mean(H_powerdB)
        average_x_powerdB = 10. * tf.log(tf.reduce_mean(tf.reduce_sum(tf.square(x), axis=1))) / tf.log(10.)
        #temp_powwdB = 10. * tf.log(tf.reduce_mean(tf.reduce_sum(tf.square(w), axis=1))) / tf.log(10.)#10. * np.log(self.NR) / np.log(10.)
        #w *= tf.pow(10., ((average_x_powerdB + average_H_powerdB - snr_db - temp_powwdB)/20.))  
        #print "new noise model"
        w *= tf.pow(10., (10.*np.log10(self.NT) + self.Hdataset_powerdB - snr_db - 10.*np.log10(self.NR))/20.)  
        #complexnoise_sigma = tf.sqrt(2. * tf.reduce_mean(tf.square(w * tf.pow(10., snr_db/20.)), axis=[0,1], keep_dims=True)) * tf.pow(10., -snr_db/20.)
        complexnoise_sigma = tf.pow(10., (10.*np.log10(self.NT) + self.Hdataset_powerdB - snr_db - 10.*np.log10(self.NR))/20.) #tf.pow(10., -snr_db/20.) 
        # Channel Output
        y = batch_matvec_mul(H, x) + w
        sig_powdB = 10. * tf.log(tf.reduce_mean(tf.reduce_sum(tf.square(batch_matvec_mul(H,x)), axis=1))) / tf.log(10.)
        noise_powdB = 10. * tf.log(tf.reduce_mean(tf.reduce_sum(tf.square(w), axis=1))) / tf.log(10.)
        actual_snrdB = sig_powdB - noise_powdB 

        #if self.mod_scheme=='MIX':
        #    return y, H, complexnoise_sigma, actual_snrdB

        if dataset_flag:
            return y, complexnoise_sigma, actual_snrdB
        else:
            return y, H, complexnoise_sigma, actual_snrdB
    
