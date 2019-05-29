import tensorflow as tf 
from utils import * 

class denoiser(object):
    def __init__(self, information):
        params = information['params']
        lgst_constel = information['constellation']
        self.NT = 2*params['K']
        self.NR = 2*params['N']
        self.L = params['L']
        self.lgst_constel = tf.cast(lgst_constel, tf.float32)
        self.M = int(lgst_constel.shape[0])
        self.information = information

    def gaussian(self, zt, features):
        tau2_t = features['tau2_t'] 
        arg = tf.reshape(zt,[-1,1]) - self.lgst_constel 
        arg = tf.reshape(arg, [-1, self.NT, self.M]) 
        arg = - tf.square(arg) / 2. / tau2_t 
        arg = tf.reshape(arg, [-1, self.M]) 
        shatt1 = tf.nn.softmax(arg, axis=1) 
        shatt1 = tf.matmul(shatt1, tf.reshape(self.lgst_constel, [self.M,1])) 
        shatt1 = tf.reshape(shatt1, [-1, self.NT]) 
        denoiser_helper = {}
        return shatt1, denoiser_helper
                                         
    def DetNet(self, zt, xhatt, rt, features, linear_helper):        
        H = features['H']
        shatt1 = fc_layer(tf.nn.relu(fc_layer(zt, 8 * self.NT)), self.NT)
        denoiser_helper = {'onsager':tf.Variable(0., trainable=False)}
        return shatt1, denoiser_helper, shatt1

    def MMNet(self, zt, xhatt, rt, features, linear_helper):        
        H = features['H']
        noise_sigma = features['noise_sigma']
        W_t = linear_helper['W']
        HTH = tf.matmul(H, H, transpose_a=True)
        v2_t = tf.divide((tf.reduce_sum(tf.square(rt), axis=1, keep_dims=True) - self.NR * tf.square(noise_sigma) / 2.), tf.expand_dims(tf.trace(HTH), axis=1))
        v2_t = tf.maximum(v2_t , 1e-9)
        v2_t = tf.expand_dims(v2_t, axis=2)
        C_t = tf.eye(self.NT, batch_shape=[tf.shape(H)[0]]) - tf.matmul(W_t, H)
        tau2_t = 1./self.NT * tf.reshape(tf.trace(tf.matmul(C_t, C_t, transpose_b=True)), [-1,1,1]) * v2_t + tf.square(tf.reshape(noise_sigma,[-1,1,1])) / (2.*self.NT) * tf.reshape(tf.trace(tf.matmul(W_t, W_t, transpose_b=True)), [-1,1,1])
        shatt1, _ = self.gaussian(zt, {'tau2_t':tau2_t/tf.Variable(tf.random_normal([1,self.NT,1], 1., 0.1))})
        denoiser_helper = {'onsager':tf.Variable(0., trainable=False)}
        return shatt1, denoiser_helper, shatt1

    def OAMPNet(self, zt, xhatt, rt, features, linear_helper):        
        H = features['H']
        noise_sigma = features['noise_sigma']
        W_t = linear_helper['W']
        HTH = tf.matmul(H, H, transpose_a=True)
        v2_t = tf.divide((tf.reduce_sum(tf.square(rt), axis=1, keep_dims=True) - self.NR * tf.square(noise_sigma) / 2.), tf.expand_dims(tf.trace(HTH), axis=1))
        v2_t = tf.maximum(v2_t , 1e-9)
        v2_t = tf.expand_dims(v2_t, axis=2)
        theta_t = tf.Variable(1., trainable=True)
        C_t = tf.eye(self.NT, batch_shape=[tf.shape(H)[0]]) - theta_t * tf.matmul(W_t, H)
        tau2_t = 1./self.NT * tf.reshape(tf.trace(tf.matmul(C_t, C_t, transpose_b=True)), [-1,1,1]) * v2_t + tf.square(tf.reshape(theta_t,[-1,1,1])*tf.reshape(noise_sigma,[-1,1,1])) / (2.*self.NT) * tf.reshape(tf.trace(tf.matmul(W_t, W_t, transpose_b=True)), [-1,1,1])
        shatt1, _ = self.gaussian(zt, {'tau2_t':tau2_t})
        denoiser_helper = {'onsager':tf.Variable(0., trainable=False)}
        return shatt1, denoiser_helper, shatt1

    def identity(self, zt, xhatt, rt, features, linear_helper):
        shatt1 = zt
        denoiser_helper = {'onsager':tf.Variable(0.)}
        return shatt1, denoiser_helper

    def naive_nn(self, zt, xhatt, rt, features, linear_helper):
        nhidden = 30
        den = tf.nn.relu(fc_layer(tf.reshape(zt, [-1,1]), nhidden))
        den = tf.nn.relu(fc_layer(den, 10))
        den = fc_layer(den, 1)
        den_logit = fc_layer(den, self.M)
        den = tf.nn.softmax(den_logit, dim=1)
        den = tf.matmul(den, tf.reshape(self.lgst_constel, [self.M,1]))    
        shatt1 = tf.reshape(den, [-1, self.NT]) 
        denoiser_helper = {'onsager': 0.}
        return shatt1, denoiser_helper

    def featurous_nn(self, zt, xhatt, rt, features, linear_helper):
        H = features['H']
        feature1 = tf.reduce_sum(tf.square(rt), axis=1, keep_dims=True)
        feature2 = 1./tf.trace(tf.matmul(H,H, transpose_a=True))
        feature2 = tf.reshape(feature2, [-1,1])
        feature = tf.concat([feature1, feature2], axis=1)
        tau2_t = tf.nn.relu(fc_layer(feature, 10))
        tau2_t = tf.nn.relu(fc_layer(tau2_t, 4))
        tau2_t = tf.square(fc_layer(tau2_t, 1))
        tau2_t = tf.expand_dims(tau2_t, axis=2) 
        tau2_t = tf.maximum(tau2_t, 1e-10)
        shatt1, _ = self.gaussian(zt, {'tau2_t': tau2_t})
        denoiser_helper = {'onsager':0.}
        return shatt1, denoiser_helper, shatt1

