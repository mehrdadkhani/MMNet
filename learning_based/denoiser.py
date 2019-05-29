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

    def old_big(self, zt, xhatt, rt, features, linear_helper):        
        H = features['H']
        noise_sigma = features['noise_sigma']
        W_t = linear_helper['W']
        feature1 = tf.reduce_sum(tf.square(rt), axis=1, keepdims=True)
        feature2 = 1./tf.trace(tf.matmul(H,H, transpose_a=True))
        feature2 = tf.reshape(feature2, [-1,1])
        feature3 = tf.trace(tf.matmul(W_t, W_t, transpose_b=True))
        feature3 = tf.reshape(feature3, [-1,1])
        WH = tf.matmul(W_t, H)
        feature4 = tf.expand_dims(tf.trace(tf.matmul(WH, WH, transpose_b=True)), axis=1)
        feature5 = tf.expand_dims(tf.trace(WH), axis=1)
        feature = tf.concat([feature1, feature2, feature3, feature4, feature5], axis=1)
        tau2_t = tf.nn.relu(fc_layer(feature, 10))
        tau2_t = tf.nn.relu(fc_layer(tau2_t, 4))
        tau2_t = tf.square(fc_layer(tau2_t, 1))
        tau2_t = tf.expand_dims(tau2_t, axis=2) 
        tau2_t = tf.maximum(tau2_t, 1e-10)
        shatt1, _ = self.gaussian(zt, {'tau2_t': tau2_t})

        alpha = tf.Variable(1.)
        beta = tf.square(tf.Variable(0.001))
        shatt1 = alpha *( shatt1 - beta * zt )
        denoiser_helper = {'onsager':tf.Variable(0., trainable=False), 'alpha':alpha, 'beta':beta}
        return shatt1, denoiser_helper

    def divfree2(self, zt, xhatt, rt, features, linear_helper):        
        H = features['H']
        noise_sigma = features['noise_sigma']
        W_t = linear_helper['W']
        #v2_t = linear_helper['v2_t']
        HTH = tf.matmul(H, H, transpose_a=True)
        v2_t = tf.divide((tf.reduce_sum(tf.square(rt), axis=1, keep_dims=True) - self.NR * tf.square(noise_sigma) / 2.), tf.expand_dims(tf.trace(HTH), axis=1))
        v2_t = tf.maximum(v2_t , 1e-9)
        v2_t = tf.expand_dims(v2_t, axis=2)
        theta_t = tf.Variable(1., trainable=True)
        C_t = tf.eye(self.NT, batch_shape=[tf.shape(H)[0]]) - theta_t * tf.matmul(W_t, H)
        tau2_t = tf.Variable(1.) * 1./self.NT * tf.reshape(tf.trace(tf.matmul(C_t, C_t, transpose_b=True)), [-1,1,1]) * v2_t + tf.Variable(1.) * tf.square(tf.reshape(theta_t,[-1,1,1])*tf.reshape(noise_sigma,[-1,1,1])) / (2.*self.NT) * tf.reshape(tf.trace(tf.matmul(W_t, W_t, transpose_b=True)), [-1,1,1])
        shatt1, _ = self.gaussian(zt, {'tau2_t':tau2_t})
        alpha = tf.Variable(1.)
        beta = tf.square(tf.Variable(0.001))
        den_output = shatt1
        shatt1 = alpha *( shatt1 - beta * zt )
        denoiser_helper = {'onsager':tf.Variable(0., trainable=False), 'alpha':alpha, 'beta':beta}
        return shatt1, denoiser_helper, den_output
    def oamp_divfree(self, zt, xhatt, rt, features, linear_helper):        
        H = features['H']
        noise_sigma = features['noise_sigma']
        W_t = linear_helper['W']
        #v2_t = linear_helper['v2_t']
        HTH = tf.matmul(H, H, transpose_a=True)
        v2_t = tf.divide((tf.reduce_sum(tf.square(rt), axis=1, keep_dims=True) - self.NR * tf.square(noise_sigma) / 2.), tf.expand_dims(tf.trace(HTH), axis=1))
        v2_t = tf.maximum(v2_t , 1e-9)
        v2_t = tf.expand_dims(v2_t, axis=2)
        theta_t = tf.Variable(1., trainable=True)
        C_t = tf.eye(self.NT, batch_shape=[tf.shape(H)[0]]) - theta_t * tf.matmul(W_t, H)
        tau2_t = 1./self.NT * tf.reshape(tf.trace(tf.matmul(C_t, C_t, transpose_b=True)), [-1,1,1]) * v2_t + tf.square(tf.reshape(theta_t,[-1,1,1])*tf.reshape(noise_sigma,[-1,1,1])) / (2.*self.NT) * tf.reshape(tf.trace(tf.matmul(W_t, W_t, transpose_b=True)), [-1,1,1])
        shatt1, _ = self.gaussian(zt, {'tau2_t':tau2_t})
        alpha = tf.Variable(1.)
        beta = tf.square(tf.Variable(0.001))
        den_output = shatt1
        shatt1 = alpha *( shatt1 - beta * zt )
        denoiser_helper = {'onsager':tf.Variable(0., trainable=False), 'alpha':alpha, 'beta':beta}
        return shatt1, denoiser_helper, den_output

    def mmnet_divfree(self, zt, xhatt, rt, features, linear_helper):        
        H = features['H']
        noise_sigma = features['noise_sigma']
        batch_size = tf.shape(H)[0]
        W_t = linear_helper['W']
        I_WH = tf.eye(self.NT, batch_shape=[batch_size]) - tf.matmul(W_t, H)
        theta4 = tf.square(tf.Variable(0.01, trainable=True))
        theta5 = tf.square(tf.Variable(1.0, trainable=True))
        e_est = tf.maximum(tf.reduce_sum(tf.square(rt), axis=1, keep_dims=True) - self.NR * tf.square(noise_sigma), 1e-10)
        e_est = tf.expand_dims(e_est, axis=2)
        HF = tf.norm(H, 'fro', axis=[1,2], keepdims=True)
        I_WHF = tf.norm(I_WH, 'fro', axis=[1,2], keepdims=True)
        WF = tf.norm(W_t, 'fro', axis=[1,2], keepdims=True) 
        tau2_t = theta4 * I_WHF / HF * e_est + theta5 * WF / HF * tf.reshape(tf.square(noise_sigma), [-1,1,1])
        shatt1, _ = self.gaussian(zt, {'tau2_t':tau2_t})
        alpha = tf.Variable(1.)
        beta = tf.square(tf.Variable(0.001))
        den_output = shatt1
        shatt1 = alpha *( shatt1 - beta * zt )
        denoiser_helper = {'onsager':tf.Variable(0., trainable=False), 'alpha':alpha, 'beta':beta, 'theta4': theta4, 'theta5': theta5}
        return shatt1, denoiser_helper, den_output
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
    def oampI(self, zt, xhatt, rt, features, linear_helper):        
        #features = x, r_t, H, noise_sigma, W_t
        #print "oamp denoiser loaded"
        #H = features['H']
        
        #print "denoiser is also using estimation of H"
        #print "Using Hlearn in oamp"
        H = features['H']
        noise_sigma = features['noise_sigma']
        W_t = linear_helper['W']
        #v2_t = linear_helper['v2_t']
        HTH = tf.matmul(H, H, transpose_a=True)
        v2_t = tf.divide((tf.reduce_sum(tf.square(rt), axis=1, keep_dims=True) - self.NR * tf.square(noise_sigma) / 2.), tf.expand_dims(tf.trace(HTH), axis=1))
        #v2_t = tf.divide((tf.norm(rt, ord=2, axis=1, keep_dims=True) - self.NR * tf.square(noise_sigma) / 2.), tf.expand_dims(tf.trace(HTH), axis=1))
        v2_t = tf.maximum(v2_t , 1e-9)
        v2_t = tf.expand_dims(v2_t, axis=2)
        #theta_t = tf.Variable(1., trainable=True)
        print('not trainable theta=1')
        theta_t = tf.Variable(1., trainable=False)
        C_t = tf.eye(self.NT, batch_shape=[tf.shape(H)[0]]) - theta_t * tf.matmul(W_t, H)
        tau2_t = 1./self.NT * tf.reshape(tf.trace(tf.matmul(C_t, C_t, transpose_b=True)), [-1,1,1]) * v2_t + tf.square(tf.reshape(theta_t,[-1,1,1])*tf.reshape(noise_sigma,[-1,1,1])) / (2.*self.NT) * tf.reshape(tf.trace(tf.matmul(W_t, W_t, transpose_b=True)), [-1,1,1])
        shatt1, _ = self.gaussian(zt, {'tau2_t':tau2_t/tf.Variable(tf.random_normal([1,32,1], 1., 0.1))})
        denoiser_helper = {'onsager':tf.Variable(0., trainable=False)}
        return shatt1, denoiser_helper, shatt1
    def oamp(self, zt, xhatt, rt, features, linear_helper):        
        #features = x, r_t, H, noise_sigma, W_t
        #print "oamp denoiser loaded"
        #H = features['H']
        
        #print "denoiser is also using estimation of H"
        #print "Using Hlearn in oamp"
        H = features['H']
        noise_sigma = features['noise_sigma']
        W_t = linear_helper['W']
        #v2_t = linear_helper['v2_t']
        HTH = tf.matmul(H, H, transpose_a=True)
        v2_t = tf.divide((tf.reduce_sum(tf.square(rt), axis=1, keep_dims=True) - self.NR * tf.square(noise_sigma) / 2.), tf.expand_dims(tf.trace(HTH), axis=1))
        #v2_t = tf.divide((tf.norm(rt, ord=2, axis=1, keep_dims=True) - self.NR * tf.square(noise_sigma) / 2.), tf.expand_dims(tf.trace(HTH), axis=1))
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
        #feature3 = tf.trace(tf.matmul(W_t, W_t, transpose_b=True))
        #feature3 = tf.reshape(feature3, [-1,1])
        #WH = tf.matmul(W_t, H)
        #feature4 = tf.expand_dims(tf.trace(tf.matmul(WH, WH, transpose_b=True)), axis=1)
        #feature5 = tf.expand_dims(tf.trace(WH), axis=1)
        #feature = tf.concat([feature1, feature2, feature3, feature4, feature5], axis=1)
        feature = tf.concat([feature1, feature2], axis=1)
        tau2_t = tf.nn.relu(fc_layer(feature, 10))
        tau2_t = tf.nn.relu(fc_layer(tau2_t, 4))
        tau2_t = tf.square(fc_layer(tau2_t, 1))
        tau2_t = tf.expand_dims(tau2_t, axis=2) 
        tau2_t = tf.maximum(tau2_t, 1e-10)
        shatt1, _ = self.gaussian(zt, {'tau2_t': tau2_t})
        denoiser_helper = {'onsager':0.}
        return shatt1, denoiser_helper, shatt1

    def featurous_nn2(self, zt, xhatt, rt, features, linear_helper):
        #print "noisy estimation of channels activated"
        H = features['H']
        pH = tf.matmul(tf.matrix_inverse(tf.matmul(H, H, transpose_a=True)), H , transpose_b=True)
        et = features['y'] - batch_matvec_mul(H, zt)
        feature1 = tf.expand_dims(batch_matvec_mul(pH, et), axis=2)
        feature2 = tf.reduce_max(tf.abs(pH), axis=2, keep_dims=True)
        #feature3 = tf.reduce_mean(tf.square(pH), axis=2, keep_dims=True)
        #feature4 = tf.expand_dims(batch_matvec_mul(pH, features['onsager']), axis=2)
        #feature = tf.concat([feature1, feature4], axis=2)
        feature = tf.concat([feature1, feature2], axis=2)
        tau2_t = tf.nn.relu(fc_layer(tf.reshape(feature,[-1,2]), 10))
        #feature = tf.concat([feature1, feature2, feature3, feature4], axis=2)
        #tau2_t = tf.nn.relu(fc_layer(tf.reshape(feature,[-1,4]), 10))
        tau2_t = tf.nn.relu(fc_layer(tau2_t, 3))
        tau2_t = tf.square(fc_layer(tau2_t, 1))
        tau2_t = tf.expand_dims(tau2_t, axis=2) 
        tau2_t = tf.maximum(tau2_t, 1e-10)
        tau2_t = tf.reshape(tau2_t, [-1, self.NT,1])
        flatten_z = tf.reshape(zt,[-1,1])
        flatten_z = tf.split(flatten_z, 1, axis=0)
        shatt1, _ = self.gaussian(flatten_z, {'tau2_t': tau2_t})
        print("Onsager term added")
        flatten_s = tf.reshape(shatt1, [-1,1])
        flatten_s = tf.split(flatten_s, 1, axis=0)
        dsdz = [tf.gradients(flatten_s[i], flatten_z[i], stop_gradients=flatten_z[i]) for i in range(len(flatten_s))]
        dsdz = tf.reshape(tf.concat(dsdz, axis=0), [-1, self.NT])
        b = tf.reduce_max(dsdz)
        dsdz = tf.clip_by_value(dsdz, -10.,10.)
        onsager = rt * tf.reduce_mean(dsdz, axis=1, keepdims=True) * self.NR / self.NT
        denoiser_helper = {'dsdz': [tf.reduce_max(dsdz), b], 'onsager':onsager}
        print("Onsager removed")
        return shatt1, denoiser_helper, shatt1


    def Damp(self, zt, xhatt, rt, features, linear_helper):
        H = features['H']
        flatten_z = tf.reshape(zt,[-1,1])
        tau2_t = tf.square(tf.Variable(1.)) + tf.reduce_sum(tf.square(rt), axis=1) / self.NT
        tau2_t = tf.reshape(tau2_t, [-1,1,1])
        shatt1, _ = self.gaussian(flatten_z, {'tau2_t': tau2_t})
        print("Onsager term added")
        flatten_s = tf.reshape(shatt1, [-1,32])
        dsdz = tf.gradients(flatten_s, flatten_z, stop_gradients=flatten_z)
        dsdz = tf.reshape(dsdz, [-1, self.NT])
        onsager = rt * tf.reduce_mean(dsdz, axis=1, keepdims=True) * self.NR / self.NT
        denoiser_helper = {'dsdz': dsdz, 'onsager':onsager}
        return shatt1, denoiser_helper


    def nips(self, zt, xhatt, rt, features, linear_helper):
        H = features['H']
        y = features['y']

        thetat = tf.concat([batch_matvec_mul(H, xhatt), batch_matvec_mul(H, rt, transpose_a=True)], axis=1)
        thetat = tf.nn.relu(fc_layer(thetat, 100))
        thetat = tf.nn.relu(fc_layer(thetat, 50))
        thetat = fc_layer(thetat, 5)
        thetat = tf.expand_dims(thetat, axis=1)
        thetat = tf.tile(thetat, [1, 1, self.NT])
        thetat = tf.reshape(thetat, [-1, 5])
        
        zzt = tf.reshape(zt, [-1, 1])
        
        w1, b1 = get_weights(1, 100)
        w2, b2 = get_weights(100, 30)
        w3, b3 = get_weights(30, self.M)	
        
        x2 = tf.nn.relu(tf.add(fc_layer(thetat, 100), tf.matmul(zzt, w1)))
        x3 = tf.nn.relu(tf.add(tf.matmul(x2, w2), b2))
        x4 = tf.add(tf.matmul(x3, w3), b3)
        x5 = tf.nn.softmax(x4, dim=1)
        x6 = tf.multiply(x5, tf.reshape(self.lgst_constel, [-1, self.M]))
        shatt1 = tf.reduce_sum(x6, axis=1)
        arrest = tf.concat([zzt, tf.reshape(shatt1, [-1,1])], axis=1)
        g = tf.multiply(w1, 0.5 * tf.sign(x2) + 0.5)
        g = tf.matmul(g, w2)
        g = tf.multiply(g, 0.5 * tf.sign(x3) + 0.5)
        g = tf.matmul(g, w3)
        
        SiSj = tf.matmul(tf.reshape(x5, [-1, self.M, 1]), tf.reshape(x5, [-1, 1, self.M]))
        Si = tf.matmul(tf.reshape(x5, [-1, 1, self.M]), tf.eye(self.M, batch_shape=[tf.shape(y)[0] * self.NT]))
        softmax_g = tf.add(Si, -SiSj)
        
        g = tf.matmul(tf.expand_dims(g, axis=1), softmax_g)
        g = tf.reshape(g, [-1, self.M])
        g = tf.multiply(g, tf.reshape(self.lgst_constel, [-1, self.M]))
        g = tf.reduce_sum(g, axis=1)
        
        g = tf.reshape(g, [-1, self.NT])
        shatt1 = tf.reshape(shatt1, [-1, self.NT])
        onsager = self.NT/float(self.NR) * tf.multiply(rt, tf.expand_dims(tf.reduce_mean(g, axis=1), axis=1))
        denoiser_helper = {'onsager':tf.stop_gradient(onsager)}

        return shatt1, denoiser_helper, shatt1
    def featurous_nn3(self, zt, xhatt, rt, features, linear_helper):
        H = features['H']
        pH = tf.matmul(tf.matrix_inverse(tf.matmul(H, H, transpose_a=True)), H , transpose_b=True)
        et = features['y'] - batch_matvec_mul(H, zt)
        feature1 = tf.expand_dims(batch_matvec_mul(pH, et), axis=2)
        feature2 = tf.reduce_max(tf.abs(pH), axis=2, keep_dims=True)
        feature3 = tf.reduce_mean(tf.square(pH), axis=2, keep_dims=True)
        feature = tf.concat([feature1, feature2, feature3], axis=2)
        tau2_t = tf.nn.relu(fc_layer(tf.reshape(feature,[-1,3]), 10))
        tau2_t = tf.nn.relu(fc_layer(tau2_t, 3))
        f = tf.concat([tf.reshape(zt, [-1,1]), tau2_t], axis=1)
        f = tf.nn.relu(fc_layer(f, 10))
        f = tf.nn.relu(fc_layer(f, 3))
        shatt1 = fc_layer(f, 1)
        shatt1 = tf.reshape(shatt1, [-1, self.NT])
        denoiser_helper = {}
        return shatt1, denoiser_helper

    def featurous_nn4(self, zt, xhatt, rt, features, linear_helper):
        H = features['H']
        pH = tf.matmul(tf.matrix_inverse(tf.matmul(H, H, transpose_a=True)), H , transpose_b=True)
        et = features['y'] - batch_matvec_mul(H, zt)
        feature1 = tf.expand_dims(batch_matvec_mul(pH, et), axis=2)
        feature2 = tf.reduce_max(tf.abs(pH), axis=2, keep_dims=True)
        feature3 = tf.reduce_mean(tf.square(pH), axis=2, keep_dims=True)
        feature = tf.concat([feature1, feature2, feature3], axis=2)
        tau2_t = fc_layer(tf.reshape(tf.nn.relu(feature),[-1,3]), 10) + fc_layer(tf.reshape(tf.nn.relu(- feature),[-1,3]), 10)
        tau2_t = fc_layer(tf.nn.relu(tau2_t), 3) + fc_layer(tf.nn.relu(- tau2_t), 3)
        tau2_t = tf.square(fc_layer(tau2_t, 1))
        tau2_t = tf.expand_dims(tau2_t, axis=2) 
        tau2_t = tf.maximum(tau2_t, 1e-10)
        tau2_t = tf.reshape(tau2_t, [-1, self.NT,1])
        shatt1, _ = self.gaussian(zt, {'tau2_t': tau2_t})
        denoiser_helper = {}
        return shatt1, denoiser_helper
    def punctual(self, zt, xhatt, rt, features, linear_helper):
        tp_prob = 1.
        fp_prob = .0
        x = self.information['x']
        #alpha = 1. - 1. * tf.cast(tf.abs(zt-x) < 0.5 * tf.abs(self.lgst_constel[1] - self.lgst_constel[0]), tf.float32)
        nt = zt - x
        goods = tf.cast(tf.abs(zt-x) < 0.5 * tf.abs(self.lgst_constel[1] - self.lgst_constel[0]), tf.float32)

        closest = tf.argmin(tf.abs(tf.expand_dims(zt, axis=2) - tf.reshape(self.lgst_constel, [1,1,-1])), axis=2)
        closest = tf.gather(self.lgst_constel, closest)

        true_positive_bool = tf.cast(tf.contrib.distributions.Bernoulli(probs=tp_prob).sample(sample_shape=tf.shape(zt)), tf.float32)
        false_positive_bool = tf.cast(tf.contrib.distributions.Bernoulli(probs=fp_prob).sample(sample_shape=tf.shape(zt)), tf.float32)

        det = goods * true_positive_bool + (1.-goods) * false_positive_bool
        shatt1 = tf.where(tf.abs(det-1.) < 0.001, closest, zt)
#        alpha = 1. - tf.cast(tf.random_uniform([500, self.NT], 0, 2, tf.int32), tf.float32) * goods
#        shatt1 = (1.-alpha) * x + alpha * zt
        denoiser_helper = {'onsager':tf.Variable([0.], trainable=False)}
        #denoiser_helper = {'z': zt, 'det':tf.reduce_mean(goods)}
        return shatt1, denoiser_helper
