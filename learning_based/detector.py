import tensorflow as tf
from layer import layer

class detector(object):
    def __init__(self, params, lgst_constel, x, y, H, noise_sigma, indices, batchsize_placeholder):
        self.information = {'x':x, 'indices': indices, 'params': params, 'constellation': lgst_constel, 'batchsize_placeholder': batchsize_placeholder}
        self.features = {'y': y, 'H': H, 'noise_sigma': noise_sigma}
        self.NT = 2 * params['K']
        self.NR = 2 * params['N']
        self.L = params['L']
        self.linear_name = params['linear_name']
        self.denoiser_name = params['denoiser_name']

    def create_graph(self): 
        batch_size = self.information['batchsize_placeholder']
        xhatk = tf.zeros(shape=[batch_size, self.NT], dtype=tf.float32)
        rk = tf.cast(self.features['y'], tf.float32)
        onsager = tf.zeros(shape=[batch_size, self.NR], dtype=tf.float32)
        xhat = []
        helper = {}
        for k in range(1,self.L+1):  
            print ("FIRST LAYER DIFFERENT LINEAR!!!!!!!!!!!!!!!!!!!!!!")
            #if k==1:
            #    xhatk, rk, onsager, helperk, den_output = layer(xhatk, rk, onsager, self.features, 'mmse2', self.denoiser_name, self.information)   
            if False:
                xhatk, rk, onsager, helperk, den_output = layer(xhatk, rk, onsager, self.features, 'lin_oamp', self.denoiser_name, self.information)   
            if True:
                xhatk, rk, onsager, helperk, den_output = layer(xhatk, rk, onsager, self.features, self.linear_name, self.denoiser_name, self.information)   
            print("detector: den_ouput instead of xk")
            xhat.append(den_output)
            #xhat.append(xhatk)
            helper['layer'+str(k)] = helperk
        print("total number of trainable variables:", self.get_n_vars())
        return xhat, helper

    def get_n_vars(self):         
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters 
