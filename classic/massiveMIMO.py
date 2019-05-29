# coding: utf-8

# # Tensorflow Massive MIMO Channel Model
# by Jakob Hoydis (jakob.hoydis@nokia-bell-labs.com)
import os
import numpy as np
from scipy.linalg import toeplitz
import tensorflow as tf
tf.set_random_seed(1)

class real_channel(object):

#    def __init__(NT, NR, r_max=250, r_min=35, alpha=3.76, sigma_sf_dB = 10, ASD_deg = 10, antennaSpacing = 0.5, B = 20e6, noise_figure_dB = 7):
#        self.M = NR
#        self.K = NT
#        self.r_max = r_max                  #Maximum distance (m)
#        self.r_min = r_min                  #Minimum distance (m)
#        self.alpha= alpha                   #Pathloss exponent
#        self.sigma_sf_dB = sigma_sf_dB      #Standard deviation of shadow fading (dB)
#        self.ASD_deg = ASD_deg              #Angular spread (degrees), must be smaller than 15deg
#        self.antennaSpacing = antennaSpacing #Antenna spacing (multiples of the wavelength)
#        self.B = B             #Communication bandwidth (Hz)
#        #self.p_dBm = 20           #Transmit power per UE (dBm) 
#        self.noise_figure_dB = noise_figure_dB   #Noise figure at the BS (in dB)
#
    def local_scattering_approximation(self, M, theta, ASDdeg, antennaSpacing=0.5):
        '''Generate the spatial correlation matrix for the local scattering model,
        defined in (2.23) with the Gaussian angular distribution. The small-angle
        approximation described in Section 2.6.2 is used to increase efficiency,
        thus this function should only be used for ASDs below 15 degrees.
    
        INPUT:
        M              = Number of antennas
        theta          = Nominal angle
        ASDdeg         = Angular standard deviation around the nominal angle
                         (measured in degrees)
        antennaSpacing = (Optional) Spacing between antennas (in wavelengths)
    
        OUTPUT:
        R              = M x M spatial correlation matrix
    
    
        This is the Python version of the Matlab function developed to generate simulation results to:
    
        Emil Bjornson, Jakob Hoydis and Luca Sanguinetti (2017), 
        "Massive MIMO Networks: Spectral, Energy, and Hardware Efficiency", 
        Foundations and Trends in Signal Processing: Vol. 11, No. 3-4, 
        pp. 154-655. DOI: 10.1561/2000000093.
    
        For further information, visit: https://www.massivemimobook.com
    
        This is version 1.0 (Last edited: 2017-11-04)
    
        License: This code is licensed under the GPLv2 license. If you in any way
        use this code for research that results in publications, please cite our
        monograph as described above.'''
    
        #Compute the ASD in radians based on input
        ASD = ASDdeg*np.pi/180
        #The correlation matrix has a Toeplitz structure, so we only need to
        #compute the first row of the matrix
        firstRow = np.zeros(shape=[M,1], dtype=np.complex64)
        #Go through all the columns of the first row
        for distance in range(0, M):
            #Compute the approximated integral as in (2.24)
            firstRow[distance] = np.exp(1j*2*np.pi*antennaSpacing*np.sin(theta)*distance)             *np.exp(-ASD**2/2 * ( 2*np.pi*antennaSpacing*np.cos(theta)*distance )**2)
    
        #Compute the spatial correlation matrix by utilizing the Toeplitz structure
        R = toeplitz(firstRow)
        return R

    def covariance_matrices(self, M, theta, ASD_deg, antennaSpacing=0.5):
        '''Implementation of 'local_scattering_approximation' in Tensorflow
         Input:
          M: integer- number of antennas
          theta: [K, 1] Tensor containing nominal angles (radians)
          ASD_deg: float - Angular standard deviation around the nominal angles (degrees)
          antennaSpacing: float - Spacing between antennas (in wavelengths)
         
         Output:
          R: [K, M, M] Tensor containing for each nominal angle the corresponding covariance matrix
        '''
        #Compute the ASD in radians based on input
        ASD = ASD_deg*np.pi/180
        #Compute constants which are independent of the inputs
        indices = tf.constant(toeplitz(np.arange(0, M)), dtype=tf.int32)
        mask_l = tf.expand_dims(tf.constant(np.tril(np.ones(M), k=0), dtype=tf.complex64), axis=0)
        mask_u =  tf.expand_dims(tf.constant(np.triu(np.ones(M), k=1), dtype=tf.complex64), axis=0)
        distance = tf.expand_dims(tf.constant(np.arange(0, M), dtype=tf.complex64), axis=0)
        d = 2*np.pi*antennaSpacing*distance
    
        a = tf.exp(1j*d*tf.cast(tf.sin(theta), dtype=tf.complex64))
        b = tf.exp(-ASD**2/2*(d*tf.cast(tf.cos(theta), dtype=tf.complex64))**2)
        rows = a*b
        
        # This part is a bit tricky. We want to gather the same indices for each of the K-rows of 'rows'
        R = tf.transpose(tf.gather(tf.transpose(rows), indices), perm=[2,0,1])
        
        # Compute the Toeplitz matrix
        R = R*mask_l + tf.conj(R)*mask_u
        return R    

    def channel_gains(self, K, r_max=250, r_min=35, alpha=3.76, sigma_sf_dB=10):
        '''Generate random channel gains assuming that K users are
        randomly uniformly distribyted on a disc of radius r_max with minim distance r_min
         Input:
          K: Integer - Number of users
          r_max: float - Maximum distance in meters
          r_min: float - Minimum distance in meters
          alpha: pathloss exponent
          sigma_sf)dB: float - Shadow fading standard deviation (dB)
          
         Ouput:
          channelGain: [K,1] Tensor containing the pathloss factors in linear scale
          theta: [K,1] Tensor containing the azimuth angles in radians
          xy_pos: [K,2] Tensor containing the positions in cartesian coordinates
        '''
        
        #Generate uniformly randomly distirbuted azimuth angles
        theta = tf.random_uniform(shape=[K, 1], minval=-np.pi, maxval=np.pi)    
        #Generate uniformly randomly distirbuted squared distances
        r2 = tf.random_uniform(shape=[K,1], minval=r_min**2, maxval=r_max**2)
        #Compute distances
        r = tf.sqrt(r2)
        #Compute positions in cartesian coordinates
        xy_pos = r*tf.stack([tf.cos(theta)[:,0], tf.sin(theta)[:,0]], axis=1)
        #Average channel gain in dB at a reference distance of 1 meter. Note that
        #-35.3 dB corresponds to -148.1 dB at 1 km, using pathloss exponent 3.76
        constantTerm_dB = -35.3;
        #Compute channel gains
        channelGain_dB = constantTerm_dB - alpha*10*tf.log(r)/tf.log(10.0)
        #Generate shadow fading factors
        shadowing_dB = tf.random_normal(shape=[K, 1], stddev=sigma_sf_dB)
        #Compute effective channel gain with shadow fading
        channelGainShadowing_dB = channelGain_dB + shadowing_dB
        channelGainShadowing = tf.pow(10.0, channelGainShadowing_dB/10)
    
        return channelGainShadowing, theta, xy_pos
    
    def mmimo_channel(self, M, K, batch_size, p_dBm = 20, r_max=250, r_min=35, alpha=3.76, sigma_sf_dB=10, ASD_deg=10, antenna_spacing=0.5, B=20e6, noise_figure_dB=7):
        
        '''Generate random channel matrices according to the local scattering model,
        defined in (2.23) with the Gaussian angular distribution. The small-angle
        approximation described in Section 2.6.2 is used to increase efficiency,
        thus this function should only be used for ASDs below 15 degrees. Users are assumed 
        uniformly distributed on a disc around the base station. Each user sees a different 
        covariance matrix depeding on its position. The user positions (and hence) covariance 
        matrices are the same for each example in the batch.  
        
         Input:
          M: int - Number of receive antennas
          K: int - Number of users
          batch_size - Number of channel matrices to generate
          p_dm: float -
          r_max: float - Maximum distance (m)
          t_min: float - Minimum distance (m)
          alpha: float - Pathloss exponent
          sigms_sf_dB: float - Standard deviation of shadow fading (dB)
          ASD_deg: float - Angular spread (degrees), must be smaller than 15deg
          antenna_spacing: float - Antenna spacing (multiples of the wavelength)
          B: float - Communication bandwidth (Hz)
          noise_figure_dB: float - Noise figure at the BS (in dB)
          
         Output:
          H: [batch_size, 2M, 2K] tf.float32 Tensor of channel matrices equivalent in real space
          R: [K, M, M] tf.complex64 Tensor of covariance matrices
        '''
        
        # Generate channel gains and azimuth angles
        channelGainShadowing, theta, xy_pos = self.channel_gains(K, r_max, r_min, alpha, sigma_sf_dB)
        # Generate covariance matrices
        R = self.covariance_matrices(M, theta, ASD_deg, antenna_spacing)
        # Compute square-roots Rsqrt of R such that R = Rsqrt*Rsqrt'
        S, U, V = tf.svd(R, full_matrices=True, compute_uv=True)
        S = tf.cast(S, dtype=tf.complex64)
        Rsqrt = tf.Variable(U*tf.expand_dims(tf.sqrt(S), axis=1), trainable=False)
        # Compute iid channel matrices
        H_iid = tf.complex(tf.random_normal(shape=[batch_size, K, M, 1], stddev=1/np.sqrt(2)),                   tf.random_normal(shape=[batch_size, K, M, 1], stddev=1/np.sqrt(2)))
        # Multiply each column of H_iid with the corresponding matrix Rsqrt
        Rsqrt_tiled = tf.tile(tf.expand_dims(Rsqrt, axis=0), [batch_size, 1, 1, 1])
        H_cor = tf.reshape(tf.matmul(Rsqrt_tiled, H_iid), shape=[batch_size, K, M])
        H_cor = tf.transpose(H_cor, perm=[0, 2, 1])
        # Compute effective channel
        channelGainFull = tf.Variable(tf.cast(tf.reshape(channelGainShadowing, shape=[1,1,K]), dtype=tf.complex64), trainable=False)
        H = tf.sqrt(channelGainFull)*H_cor
        Hr = tf.cast(tf.real(H), tf.float32)
        Hi = tf.cast(tf.imag(H), tf.float32)     
        h1 = tf.concat([Hr, -1. * Hi], axis=2)
        h2 = tf.concat([Hi, Hr], axis=2)
        H = tf.concat([h1, h2], axis=1)
        # normalize the variance of H
        Hvar = tf.reduce_mean(tf.pow(H, 2), axis=[1,2])
        normalizer_batch = tf.pow(Hvar, -1.) /(2.*M)
        normalizer_batch = tf.sqrt(normalizer_batch)
        H = tf.multiply(tf.expand_dims(tf.expand_dims(normalizer_batch, dim=[1]), dim=[2]), H)

        return H
