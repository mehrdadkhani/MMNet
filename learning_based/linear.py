import tensorflow as tf 
import numpy as np 
from utils import * 

class linear(object):
    def __init__(self, information):
        params = information['params']
        self.NT = 2*params['K']
        self.NR = 2*params['N']
        self.batch_size = information['batchsize_placeholder']

    def fixed_WpH(self, shatt, rt, features):
        batch_size = self.batch_size
        H = features['H']
        varn = tf.square(features['noise_sigma'])
        I = tf.eye(self.NT, batch_shape=[batch_size])
        #W1 = tf.Variable(tf.random_normal([1,self.NT,self.NT], 0., 0.00001))
        #W1 = tf.tile(W1,[batch_size, 1, 1])
        #W2 = tf.Variable(tf.random_normal([1,self.NR,self.NR], 0., 0.00001))
        #W2 = tf.tile(W2,[batch_size, 1, 1])
        #H = tf.matmul(W2, tf.matmul(H, W1))
        pH = tf.matmul(tf.matrix_inverse(tf.Variable(1.) * tf.matmul(H, H, transpose_a=True) + tf.reshape(varn, [-1,1,1]) * I), H , transpose_b=True)
        
        #W1 = tf.Variable(tf.random_normal([1,self.NT,self.NR], 0., 0.00001))
        V1 = tf.Variable(tf.random_normal([1,self.NT,1], 0., 0.00001))
        V2 = tf.Variable(tf.random_normal([1,1,self.NR], 0., 0.00001))
        W1 = tf.matmul(V1, V2)
        W1 = tf.tile(W1,[batch_size, 1, 1])
        #V1 = tf.Variable(tf.random_normal([1,self.NT,1], 0., 0.001))
        #V2 = tf.Variable(tf.random_normal([1,1,self.NR], 0., 0.001))
        #W1 = tf.matmul(V1,V2)
        #V3 = tf.Variable(tf.random_normal([1,self.NT,1], 0., 0.001))
        #V4 = tf.Variable(tf.random_normal([1,1,self.NR], 0., 0.001))
        #W2 = tf.matmul(V3,V4)
        #W2 = tf.tile(W1,[batch_size, 1, 1])
        #W2 = tf.Variable(tf.random_normal([1,self.NT,self.NR], 0., 0.00001))
        #W2 = tf.tile(W2,[batch_size, 1, 1])
        #W3 = tf.Variable(tf.random_normal([1,self.NT,self.NR], 0., 0.00001))
        #W3 = tf.tile(W3,[batch_size, 1, 1])
        #W = W + W2 * pH + W3 * tf.square(pH)
        #W =  tf.Variable(1.) * pH + tf.Variable(0.001)
        alpha = tf.Variable(1.) 
        W = alpha * pH + W1 
        zt = shatt + batch_matvec_mul(W, rt)
        helper = {'W': W, 'W1': W1, 'alpha': alpha, 'normHt': tf.norm(tf.matrix_transpose(H), axis=[1,2]), 'normW': tf.norm(W, axis=[1,2]), 'normpH': tf.reduce_mean(tf.norm(pH, axis=[1,2])), 'normW1': tf.norm(W1, axis=[1,2])}
        return zt, helper

    def fixed_driv3(self, shatt, rt, features):
        batch_size = self.batch_size
        H = features['H'] 
        WsetRT = tf.Variable(tf.random_normal([1, self.NT/2, self.NT/2, self.NR/2], 0., 1e-4))
        WsetRR = tf.Variable(tf.random_normal([1, self.NR/2, self.NT/2, self.NR/2], 0., 1e-4))
        Hr = tf.slice(H,[0,0,0],[-1,self.NR/2,self.NT/2])
        Wrt = tf.reshape(tf.reduce_sum(Hr, axis=1), [-1, self.NT/2, 1, 1]) * WsetRT
        Wrt = tf.reduce_mean(Wrt, axis=1)
        Wrr = tf.reshape(tf.reduce_sum(Hr, axis=2), [-1, self.NR/2, 1, 1]) * WsetRR
        Wrr = tf.reduce_mean(Wrr, axis=1)

        WsetIT = tf.Variable(tf.random_normal([1, self.NT/2, self.NT/2, self.NR/2], 0., 1e-4))
        WsetIR = tf.Variable(tf.random_normal([1, self.NR/2, self.NT/2, self.NR/2], 0., 1e-4))
        Hi = tf.slice(H,[0,self.NR/2,0],[-1,self.NR/2,self.NT/2])
        Wit = tf.reshape(tf.reduce_sum(Hi, axis=1), [-1, self.NT/2, 1, 1]) * WsetIT
        Wit = tf.reduce_mean(Wit, axis=1)
        Wir = tf.reshape(tf.reduce_sum(Hi, axis=2), [-1, self.NR/2, 1, 1]) * WsetIR
        Wir = tf.reduce_mean(Wir, axis=1)
        Wr = Wrt + Wrr
        Wi = Wit + Wir
        Wtotal = self.complex_to_real(tf.complex(Wr,Wi))
        varn = tf.expand_dims(tf.square(features['noise_sigma']), axis=2)
        I = tf.eye(self.NT, batch_shape=[batch_size])
        pH = tf.Variable(1.) * tf.matmul(tf.matrix_inverse(tf.matmul(H, H, transpose_a=True) + (varn + tf.square(tf.Variable(0.01))) * I), H , transpose_b=True)
        W =  Wtotal + tf.tile(tf.Variable(tf.random_normal([1, self.NT, self.NR], 0., 0.01)), [batch_size,1,1]) 
        zt = shatt + batch_matvec_mul(W, rt)
         
        helper = {'W': W}#, 'WsetR': WsetR, 'WsetI': WsetI, 'H':H}
        return zt, helper
    def fixed_driv2(self, shatt, rt, features):
        batch_size = self.batch_size
        H = features['H'] 
        WsetR = tf.Variable(tf.random_normal([1, self.NT/2*self.NR/2, self.NT/2, self.NR/2], 0., 1e-4))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.reduce_mean(tf.square(WsetR)))
        Hr = tf.slice(H,[0,0,0],[-1,self.NR/2,self.NT/2])
        nsplit = 1
        Wr = []
        for hr in tf.split(Hr, nsplit):
            wr = tf.reshape(hr, [-1, self.NT/2*self.NR/2, 1, 1]) * WsetR
            Wr.append(tf.reduce_mean(wr, axis=1, keepdims=True))
        Wr = tf.concat(Wr, axis=0)

        Wr = tf.reduce_mean(Wr, axis=1)
        WsetI = tf.Variable(tf.random_normal([1, self.NT/2*self.NR/2, self.NT/2, self.NR/2], 0., 1e-4))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.reduce_mean(tf.square(WsetI)))
        Hi = tf.slice(H,[0,self.NR/2,0],[-1,self.NR/2,self.NT/2])
        Wi = []
        for hi in tf.split(Hi, nsplit):
            wi = tf.reshape(hi, [-1, self.NR/2*self.NT/2, 1, 1]) * WsetI
            Wi.append(tf.reduce_mean(wi, axis=1, keepdims=True)) 
        Wi = tf.concat(Wi, axis=0)
        Wi = tf.reduce_mean(Wi, axis=1)
        Wtotal = self.complex_to_real(tf.complex(Wr,Wi))
        varn = tf.expand_dims(tf.square(features['noise_sigma']), axis=2)
        I = tf.eye(self.NT, batch_shape=[batch_size])
        pH = tf.Variable(1.) * tf.matmul(tf.matrix_inverse(tf.matmul(H, H, transpose_a=True) + (varn + tf.square(tf.Variable(0.01))) * I), H , transpose_b=True)
        print("NO PHHHHHHHHHHH")
        W = Wtotal + tf.tile(tf.Variable(tf.random_normal([1, self.NT, self.NR], 0., 0.01)), [batch_size,1,1]) 
        zt = shatt + batch_matvec_mul(W, rt)
         
        helper = {'W': W, 'WsetR': WsetR, 'WsetI': WsetI, 'H':H}
        return zt, helper

    def fixed_driv(self, shatt, rt, features):
        batch_size = self.batch_size
        H = features['H'] 
        Wset1 = tf.Variable(tf.random_normal([1, self.NT, self.NT, self.NR], 0., 0.01))
        W1 = tf.reshape(tf.reduce_sum(H, axis=1), [-1, self.NT, 1, 1]) * Wset1
        W1 = tf.reduce_mean(W1, axis=1)
        Wset2 = tf.Variable(tf.random_normal([1, self.NR, self.NT, self.NR], 0., 0.01))
        W2 = tf.reshape(tf.reduce_sum(H, axis=2), [-1, self.NR, 1, 1]) * Wset2
        W2 = tf.reduce_mean(W2, axis=1)
        varn = tf.square(features['noise_sigma'])
        I = tf.eye(self.NT, batch_shape=[batch_size])
        pH = tf.matmul(tf.matrix_inverse(tf.Variable(1.) * tf.matmul(H, H, transpose_a=True) + tf.reshape(varn, [-1,1,1]) * I), H , transpose_b=True)
        W = pH + W1 + W2 + tf.tile(tf.Variable(tf.random_normal([1, self.NT, self.NR], 0., 0.01)), [batch_size,1,1]) 
        zt = shatt + batch_matvec_mul(W, rt)
         
        helper = {'W': W}
        return zt, helper

    def semi_fixed2(self, shatt, rt, features):
        batch_size = self.batch_size
        Wr = tf.Variable(tf.random_normal([1, self.NT/2, self.NT/2], 0., 0.01))
        Wi = tf.Variable(tf.random_normal([1, self.NT/2, self.NT/2], 0., 0.01))
        theta = tf.concat([tf.concat([Wr, -Wi], axis=2), tf.concat([Wi, Wr], axis=2)], axis=1)
        theta = tf.tile(theta, [batch_size, 1, 1])
        H = features['H'] 
        H_comp = tf.complex(tf.slice(H,[0,0,0], [-1,self.NR/2,self.NT/2]), tf.slice(H,[0,self.NR/2,0], [-1,self.NR/2,self.NT/2]))
        with tf.device('/cpu:0'):
            Sh, Uh, Vh = tf.svd(H_comp)
        Sw = tf.Variable(1.) * Sh / (tf.square(Sh) + tf.square(tf.Variable(0.01)))
        Sw = tf.matrix_diag(tf.concat([Sw,Sw], axis=1)) 
        Uh = self.complex_to_real(Uh)
        Vh = self.complex_to_real(Vh)
        W = tf.matmul(tf.matmul(Vh, Sw), Uh, transpose_b=True)
        W = tf.matmul(theta, W)
        zt = shatt + batch_matvec_mul(W, rt)
         
        helper = {'W': W, 'theta': theta}
        return zt, helper

    def semi_fixed(self, shatt, rt, features):
        batch_size = self.batch_size
        Wr = tf.Variable(tf.random_normal([1, self.NT/2, self.NR/2], 0., 0.01))
        Wi = tf.Variable(tf.random_normal([1, self.NT/2, self.NR/2], 0., 0.01))
        theta = tf.concat([tf.concat([Wr, -Wi], axis=2), tf.concat([Wi, Wr], axis=2)], axis=1)
        theta = tf.tile(theta, [batch_size, 1, 1])
        H = features['H'] 
        alpha = tf.square(tf.Variable(0.01))
        W = tf.matmul(tf.matrix_inverse(tf.matmul(H,H, transpose_a=True)+alpha*tf.eye(self.NT, batch_shape=[batch_size])), H, transpose_b=True)
        W = tf.Variable(1.) * W + tf.Variable(1.) * theta
        zt = shatt + batch_matvec_mul(W, rt)
         
        helper = {'W': W}
        return zt, helper

    def fixed_W(self, shatt, rt, features):
        batch_size = self.batch_size
        print(batch_size)
        #W = tf.Variable(tf.random_normal([1, self.NT, self.NR], 0., 0.01))
        
        Wr = tf.Variable(tf.random_normal(shape=[1, self.NT//2, self.NR//2], mean=0., stddev=0.01))
        Wi = tf.Variable(tf.random_normal([1, self.NT//2, self.NR//2], 0., 0.01))
        W = tf.concat([tf.concat([Wr, -Wi], axis=2), tf.concat([Wi, Wr], axis=2)], axis=1)
        #tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.reduce_mean(tf.square(W)))
        W = tf.tile(W, [batch_size, 1, 1])
        H = features['H'] 
        #H_comp = tf.complex(tf.slice(H,[0,0,0], [-1,self.NR/2,self.NT/2]), tf.slice(H,[0,self.NR/2,0], [-1,self.NR/2,self.NT/2]))
        #with tf.device('/cpu:0'):
        #    Sh, Uh, Vh = tf.svd(H_comp)
        #W_comp = tf.complex(tf.matrix_transpose(Wr), - tf.matrix_transpose(Wi))
        #with tf.device('/cpu:0'):
        #    Sw, Uw, Vw = tf.svd(W_comp)
        #Uh = self.complex_to_real(Uh)
        #Vh = self.complex_to_real(Vh)
       # H = features['H']
       # with tf.device('/cpu:0'):
       #     Sw, Uw, Vw = tf.svd(tf.matrix_transpose(W))

       # with tf.device('/cpu:0'):
       #     Sh, Uh, Vh = tf.svd(H)
       # W = tf.matmul(tf.matmul(Vw, tf.matrix_diag(Sw)), Uw, transpose_b=True)
       # #S2 = tf.matmul(V, tf.matmul(W, U), transpose_a=True)
        zt = shatt + batch_matvec_mul(W, rt)
         
        helper = {'W': W, 'I_WH': tf.eye(self.NT, batch_shape=[batch_size])-tf.matmul(W, H)}
        #helper = {'W': W,'WH': tf.matmul(W,H), 'svd': [Sw,Uw, Vw, Sh, Uh, Vh]}
        #helper = {'W': W, 'VhVwt': tf.matmul(Vh, Vw, transpose_b=True), 'norm_diff_U': tf.norm(Uh-Uw, axis=[1,2]), 'norm_Uw': tf.norm(Uw, axis=[1,2]), 'norm_diff_V': tf.norm(Vh-Vw, axis=[1,2]), 'norm_V': tf.norm(Vw, axis=[1,2])}
        return zt, helper


    def aHt(self, shatt, rt, features):
        batch_size = self.batch_size
        H = features['H']
        W = tf.matrix_transpose(H)
        helper = {'W': W}
        #zt = shatt + tf.square(tf.Variable(1.))/tf.expand_dims(tf.norm(H, axis=[1,2]), axis=1) * batch_matvec_mul(W, rt) 
        zt = shatt + tf.square(tf.Variable(1.)) * batch_matvec_mul(W, rt) 
        return zt, helper

    def Ht(self, shatt, rt, features):
        batch_size = self.batch_size
        H = features['H']
        W = tf.matrix_transpose(H)
        helper = {'W': W}
        zt = shatt + batch_matvec_mul(W, rt)
        return zt, helper

#    def lin_DetNet(self, shatt, rt, features):
#        batch_size = self.batch_size
#        H = features['H']
#        W = tf.matrix_transpose(H)
#        helper = {'W': W}
#        theta0 = tf.Variable(tf.random_normal([1,self.NT, self.NT], 0., 0.001))
#        beta1 = tf.Variable(1.)
#        beta2 = tf.Variable(0.001)
#        zt = batch_matvec_mul(tf.tile(theta0, [batch_size,1,1]), shatt) + batch_matvec_mul(W, beta1 * rt + beta2 * features['y'])
#
#        return zt, helper
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
    
    def new_cz(self, shatt, rt, features):
        batch_size = self.batch_size
        H = features['H']
        W = tf.Variable(tf.random_normal([1, self.NT, self.NT], 0., 0.0001))
        
        W = tf.tile(W, [batch_size, 1, 1])
        W = tf.matmul(W, H, transpose_b=True)
        helper = {'W': W}
        zt = shatt + batch_matvec_mul(W, rt)
        return zt, helper

    def cz(self, shatt, rt, features):
        batch_size = self.batch_size
        H = features['H']
        Cz = tf.Variable(tf.random_normal([1,self.NT,self.NT], 0., 0.1))
        Cz = tf.tile(Cz,[batch_size, 1, 1])
        
        Cnz = tf.Variable(tf.random_normal([1,self.NR,self.NT], 0., 0.1))
        Cnz = tf.tile(Cnz,[batch_size, 1, 1])
        a = tf.matmul(Cz, H, transpose_a=True, transpose_b=True)
        a -= tf.matrix_transpose(Cnz)

        b = tf.matmul(tf.matmul(H, Cz), H, transpose_b=True)
        c = tf.matmul(H, Cnz, transpose_b=True) + tf.matmul(Cnz, H, transpose_b=True)
        d = tf.expand_dims(tf.square(features['noise_sigma']) + tf.Variable(0.01), axis=2) * tf.eye(self.NR, batch_shape=[batch_size])
        W = tf.matmul(a, tf.matrix_inverse(b-c+d)) 
        helper = {'W': W}
        zt = shatt + batch_matvec_mul(W, rt)
        return zt, helper

    def identity(self, shatt, rt, features):
        helper = {'W': [[0.]]}
        zt = shatt
        return zt, helper

    def mmse(self, shatt, rt, features):
        batch_size = self.batch_size
        H = features['H']
        alpha = tf.square(tf.Variable(0.01))
        W = tf.matmul(tf.matrix_inverse(tf.matmul(H,H, transpose_a=True)+alpha*tf.eye(self.NT, batch_shape=[batch_size])), H, transpose_b=True)
        helper = {'W': W}
        zt = shatt + batch_matvec_mul(W, rt)
        return zt, helper

    #def test_lin3(self, shatt, rt, features):                                               
    #    H = features['H']
    #    pH = tf.matmul(tf.matrix_inverse(tf.matmul(tf.matmul(H, C)) H, transpose_a=True)), H , transpose_b=True)  
    #    Ht = tf.matrix_transpose(H)
    #    helper = {'W': W}
    #    zt = shatt + batch_matvec_mul(W, rt)
    #    return zt, helper

    def test_lin2(self, shatt, rt, features):                                               
        H = features['H']
        pH = tf.matmul(tf.matrix_inverse(tf.matmul(H, H, transpose_a=True)), H , transpose_b=True)  
        feature1 = tf.expand_dims(batch_matvec_mul(pH, rt), axis=2)
        feature2 = tf.reduce_max(tf.abs(pH), axis=2, keep_dims=True)
        feature3 = tf.reduce_mean(tf.square(pH), axis=2, keep_dims=True)
        feature = tf.concat([feature1 / 10., feature2, feature3], axis=2)
        di = tf.nn.relu(fc_layer(tf.reshape(feature,[-1,3]), 10))
        di = tf.nn.relu(fc_layer(di, 3))
        di = tf.square(fc_layer(di, 1))
        di = tf.reshape(di, [-1, self.NT])
        D = tf.matrix_diag(di) * 10.
        W = tf.matmul(D, pH) 
        zt = shatt + batch_matvec_mul(W, rt) 

        helper = {'W': W, 'D': feature}
        zt = shatt + batch_matvec_mul(W, rt)
        return zt, helper

    def test_lin(self, shatt, rt, features):                                               
        H = features['H']
        pH = tf.matmul(tf.matrix_inverse(tf.matmul(H, H, transpose_a=True)), H , transpose_b=True)  
        feature1 = pH / 10.                                                                      
        feature2 = tf.tile(tf.expand_dims(rt, axis=1), [1,self.NT,1])                      
        feature = tf.concat([feature1, feature2], axis=2)
        #trainable = True                                                                   
        f = tf.nn.relu(fc_layer(tf.reshape(feature,[-1,2*self.NR]), 512))#, trainable=trainable))      
        f = tf.nn.relu(fc_layer(f, 200))#, trainable=trainable))                              
        f = fc_layer(f, self.NR)#, trainable=trainable)                                      
        f = tf.reshape(f, [-1, self.NT, self.NR])                                          
        W = pH + 1e-2 * f                                                                         
        zt = shatt + batch_matvec_mul(W, rt)                                               
                                                                                           
        helper = {}                                                                        
    
        return zt, helper 


    def complex_to_real(self, inp):
        Hr = tf.real(inp)
        Hi = tf.imag(inp)
        h1 = tf.concat([Hr, -Hi], axis=-1)
        h2 = tf.concat([Hi,  Hr], axis=-1)
        inp = tf.cast(tf.concat([h1, h2], axis=-2), tf.float32)
        return inp

    def complex_svd(self, shatt, rt, features):
        H = features['H']
        noise_var = tf.expand_dims(tf.square(features['noise_sigma']), axis=2)
        batch_size = self.batch_size
        H_comp = tf.complex(tf.slice(H,[0,0,0], [-1,self.NR/2,self.NT/2]), tf.slice(H,[0,self.NR/2,0], [-1,self.NR/2,self.NT/2]))
        with tf.device('/cpu:0'):
            S, U, V = tf.svd(H_comp)
        U = self.complex_to_real(U)
        V = self.complex_to_real(V)
        #theta_u = tf.Variable(tf.random_normal([1, self.NT/2]))
        ur = tf.Variable(tf.random_normal([1, self.NT/2]))
        ui = tf.Variable(tf.random_normal([1, self.NT/2]))
        #ThUr = tf.matrix_diag(tf.cos(theta_u))
        #ThUi = tf.matrix_diag(tf.sin(theta_u))
        ThUr = tf.matrix_diag(ur)
        ThUi = tf.matrix_diag(ui)
        ThU = tf.concat([tf.concat([ThUr, -ThUi], axis=2), tf.concat([ThUi, ThUr], axis=2)], axis=1)
        ThU = tf.tile(ThU, [batch_size, 1, 1])
        U = tf.matmul(U, ThU)
        #theta_v = tf.Variable(tf.random_normal([1, self.NT/2]))
        vr = tf.Variable(tf.random_normal([1, self.NT/2]))
        vi = tf.Variable(tf.random_normal([1, self.NT/2]))
        #ThVr = tf.matrix_diag(tf.cos(theta_v))
        #ThVi = tf.matrix_diag(tf.sin(theta_v))
        ThVr = tf.matrix_diag(vr)
        ThVi = tf.matrix_diag(vi)
        ThV = tf.concat([tf.concat([ThVr, -ThVi], axis=2), tf.concat([ThVi, ThVr], axis=2)], axis=1)
        ThV = tf.tile(ThV, [batch_size, 1, 1])
        V = tf.matmul(V, ThV)
        #theta_u = tf.Variable(tf.random_normal([1,self.NT, self.NT], 0., 0.01))
        #tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.norm(theta_u, axis=[1,2], ord=1))
        #theta_u = tf.tile(theta_u, [batch_size, 1, 1])
        #theta_v = tf.Variable(tf.random_normal([1,self.NT, self.NT], 0., 0.01))
        #tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.norm(theta_v, axis=[1,2], ord=1))
        #theta_v = tf.tile(theta_v, [batch_size, 1, 1])
        #U = tf.matmul(U, theta_u)
        #V = tf.matmul(V, theta_v)
        #w1 = tf.Variable(tf.random_normal([1,self.NT/2], 0., 0.01))
        #w1 = tf.matrix_diag(tf.square(tf.concat([w1,w1], axis=1)))
        #a = tf.Variable(tf.random_normal([1,self.NT, 1], 0., 0.01))
        #b = tf.Variable(tf.random_normal([1,1, self.NT], 0., 0.01))
        #w1 = tf.matmul(a,b)
        S = tf.matrix_diag(tf.concat([S,S], axis=1))
        S1 = tf.square(tf.Variable(1.)) * S / (tf.square(S) + tf.square(tf.Variable(1.)) * noise_var + tf.square(tf.Variable(0.0001)))
        w1 = tf.Variable(tf.random_normal([1,self.NT, self.NT], 0., 0.01))
        w1 = tf.tile(w1,[batch_size,1 , 1])
        w1 = w1 + S1 
        #w1 = S1
        W = tf.matmul(V, tf.matmul(w1, U, transpose_b=True))
        zt = shatt + batch_matvec_mul(W, rt)
        helper = {'W': W, 'S': S}#, 'w1': w1}
        return zt, helper

    def svd(self, shatt, rt, features):
        H = features['H']
        batch_size = self.batch_size
        with tf.device('/cpu:0'):
            S, U, V = tf.svd(H)
        a = tf.Variable(tf.random_normal([1,self.NT, 1], 0., 0.01))
        b = tf.Variable(tf.random_normal([1,1, self.NT], 0., 0.01))
        w1 = tf.matmul(a, b)
        #w1 = tf.Variable(tf.random_normal([1,self.NT, self.NT], 0., 0.01))
        #w1 = tf.Variable(tf.random_normal([1,self.NT], 0., 0.01))
        #w1 = tf.matrix_diag(tf.square(w1))
        w1 = tf.tile(w1,[batch_size, 1, 1])
        w1 = w1 + tf.Variable(1.) * tf.matrix_diag(S/(tf.square(S) + tf.square(tf.Variable(0.01))))
        W = tf.matmul(V, tf.matmul(w1, U, transpose_b=True))
        zt = shatt + batch_matvec_mul(W, rt)
        helper = {'W': W, 'S': S, 'w1': w1}
        return zt, helper

    def new_svd(self, shatt, rt, features):
        H = features['H']
        batch_size = self.batch_size
        with tf.device('/cpu:0'):
            S, U, V = tf.svd(H)
        theta_u = tf.Variable(tf.random_normal([1,self.NT, self.NT], 0., 0.01))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.norm(theta_u, axis=[1,2], ord=1))
        theta_u = tf.tile(theta_u, [batch_size, 1, 1])
        theta_v = tf.Variable(tf.random_normal([1,self.NT, self.NT], 0., 0.01))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.norm(theta_v, axis=[1,2], ord=1))
        theta_v = tf.tile(theta_v, [batch_size, 1, 1])
        U = tf.matmul(U, theta_u)
        V = tf.matmul(V, theta_v)
        w1 = tf.Variable(tf.random_normal([1,self.NT], 0., 0.01))
        w1 = tf.matrix_diag(tf.square(w1))
        w1 = tf.tile(w1,[batch_size, 1, 1])
        W = tf.matmul(V, tf.matmul(w1, U, transpose_b=True))
        zt = shatt + batch_matvec_mul(W, rt)
        helper = {'W': W, 'S': S, 'w1': w1}
        return zt, helper
    def mmse2(self, shatt, rt, features):                                                   
        batch_size = self.batch_size                                                        
        H = features['H']                                                                   
        noise_sigma = features['noise_sigma']                                               
        alpha = tf.square(tf.Variable(1.))                                                  
        W = tf.matmul(tf.matrix_inverse(tf.matmul(H,H, transpose_a=True)+alpha* tf.square(tf.expand_dims(noise_sigma, axis=2)) / 2. * tf.eye(self.NT, batch_shape=[batch_size])), H, transpose_b=True)
        W = tf.square(tf.Variable(1.)) * W                                                  
        helper = {'W': W, 'alpha':alpha} 
        zt = shatt + batch_matvec_mul(W, rt)                                                
        return zt, helper
    
    def lin_oamp(self, shatt, rt, features):
        print("oamp linear loaded")
        noise_sigma = features['noise_sigma']
        H = features['H']
        #print "noisy estimation of channels activated"
        HTH = tf.matmul(H, H, transpose_a=True)
        HHT = tf.matmul(H, H, transpose_b=True)
        batch_size = self.batch_size
        gamma_t = tf.Variable(1., trainable=True)
        v2_t = tf.divide((tf.reduce_sum(tf.square(rt), axis=1, keep_dims=True) - self.NR * tf.square(noise_sigma) / 2.), tf.expand_dims(tf.trace(HTH), axis=1))
        #v2_t = tf.divide((tf.norm(rt, ord=2, axis=1, keep_dims=True) - self.NR * tf.square(noise_sigma) / 2.), tf.expand_dims(tf.trace(HTH), axis=1))
        v2_t = tf.maximum(v2_t , 1e-9)
        v2_t = tf.expand_dims(v2_t, axis=2)
        lam = tf.square(tf.expand_dims(noise_sigma, axis=2))/2. 
        inv_term = tf.matrix_inverse(v2_t * HHT +  lam * tf.eye(self.NR, batch_shape=[tf.shape(H)[0]]))
        What_t = v2_t * tf.matmul(H, inv_term, transpose_a=True)
        #inv_term = tf.matrix_inverse(v2_t * HTH +  lam * tf.eye(self.NT, batch_shape=[tf.shape(H)[0]]))
        #What_t = v2_t * tf.matmul(inv_term, H, transpose_b=True)
        W_t = self.NT * What_t / tf.reshape(tf.trace(tf.matmul(What_t, H)), [-1,1,1])
        zt = shatt + gamma_t * batch_matvec_mul(W_t, rt)
        #helper = {'W': W_t, 'v2_t': v2_t, 'Hest':H}
        helper = {'W': gamma_t * W_t, 'I_WH': tf.eye(self.NT, batch_shape=[batch_size])-tf.matmul(gamma_t*W_t, H)}
        return zt, helper

    def lin_oamp_chest(self, shatt, rt, features):
        noise_sigma = features['noise_sigma']
        #H = tf.Variable(tf.random_normal([1, self.NR, self.NT]))
        #H = tf.tile(H, [self.batch_size, 1, 1])
        H = features['H']

        HTH = tf.matmul(H, H, transpose_a=True)
        HHT = tf.matmul(H, H, transpose_b=True)
        batch_size = self.batch_size
        gamma_t = tf.Variable(1., trainable=True)
        v2_t = tf.divide((tf.reduce_sum(tf.square(rt), axis=1, keep_dims=True) - self.NR * tf.square(noise_sigma) / 2.), tf.expand_dims(tf.trace(HTH), axis=1))
        #v2_t = tf.divide((tf.norm(rt, ord=2, axis=1, keep_dims=True) - self.NR * tf.square(noise_sigma) / 2.), tf.expand_dims(tf.trace(HTH), axis=1))
        v2_t = tf.maximum(v2_t , 1e-9)
        v2_t = tf.expand_dims(v2_t, axis=2)
        lam = tf.square(tf.expand_dims(noise_sigma, axis=2))/2. 
        inv_term = tf.matrix_inverse(v2_t * HHT +  lam * tf.eye(self.NR, batch_shape=[tf.shape(H)[0]]))
        What_t = v2_t * tf.matmul(H, inv_term, transpose_a=True)
        #inv_term = tf.matrix_inverse(v2_t * HTH +  lam * tf.eye(self.NT, batch_shape=[tf.shape(H)[0]]))
        #What_t = v2_t * tf.matmul(inv_term, H, transpose_b=True)
        W_t = self.NT * What_t / tf.reshape(tf.trace(tf.matmul(What_t, H)), [-1,1,1])
        zt = shatt + gamma_t * batch_matvec_mul(W_t, rt)
        helper = {'W': W_t, 'v2_t': v2_t, 'Hest':H}
        return zt, helper
