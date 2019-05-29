from utils import * 
from denoiser import denoiser 
from linear import linear

def layer(xhat, r, onsager, features, linear_name, denoiser_name, information):
    denoiser_fun = getattr(denoiser(information), denoiser_name)
    linear_fun = getattr(linear(information), linear_name)
    z, linear_helper = linear_fun(xhat, r, features)
    features['onsager'] = onsager
    new_xhat, denoiser_helper, den_output = denoiser_fun(z, xhat, r, features, linear_helper)
    new_r = features['y'] - batch_matvec_mul(features['H'], new_xhat)
    new_onsager = denoiser_helper['onsager']

    W = linear_helper['W'] 
    
    I = tf.eye(information['params']['K']*2, batch_shape=[information['batchsize_placeholder']])
    e10 = batch_matvec_mul(I - tf.matmul(W, features['H']), information['x'] - xhat)
    
    e11 = batch_matvec_mul(W, features['y'] - batch_matvec_mul(features['H'], information['x']))
    helper = {'linear': linear_helper, 'denoiser': denoiser_helper, 'stat': {'e0': xhat-information['x'], 'e1':z-information['x'], 'e2':new_xhat-information['x'], 'e10': e10, 'e11': e11}}
    
    return new_xhat, new_r, new_onsager, helper, den_output
