import tensorflow as tf
import numpy as np

def model_eval(test_data, snr_min, snr_max, mmse_accuracy, accuracy, batch_size, snr_db_min, snr_db_max, H, sess, iterations=150):
    SNR_dBs = np.linspace(snr_min, snr_max, snr_max - snr_min + 1)
    accs_mmse = []#np.zeros(shape=SNR_dBs.shape)
    accs_NN = []#np.zeros(shape=SNR_dBs.shape)
    bs = 1000
    for i in range(SNR_dBs.shape[0]):
        noise_ = []
        error_ = []
        mmse = 0.
        nn = 0.
        for j in range(iterations):
            feed_dict = {
                    batch_size: bs,
                    snr_db_max: SNR_dBs[i],
                    snr_db_min: SNR_dBs[i],
                }    
            if not test_data == []:
                sample_ids = np.random.randint(0, np.shape(test_data)[0], bs)
                feed_dict[H] = test_data[sample_ids]
            acc = sess.run([mmse_accuracy, accuracy], feed_dict)
            mmse += acc[0] / iterations
            nn   += acc[1] / iterations
        accs_mmse.append((SNR_dBs[i], 1. - mmse))#+= acc[0]/iterations
        accs_NN.append((SNR_dBs[i], 1. - nn))# += acc[1]/iterations
    return {'mmse':accs_mmse, 'model':accs_NN}

def demodulate(y, constellation):
    shape = tf.shape(y)
    y = tf.reshape(y, shape=[-1,1])
    constellation = tf.reshape(constellation, shape=[1, -1])
    indices = tf.cast(tf.argmin(tf.abs(y - constellation), axis=1), tf.int32)
    indices = tf.reshape(indices, shape=shape)
    return indices  

def accuracy(x, y):
    '''Computes the fraction of elements for which x and y are equal'''
    return tf.reduce_mean(tf.cast(tf.equal(x, y), tf.float32))

def get_weights(in_size, out_size): 
    w = tf.Variable(tf.random_normal([in_size, out_size], stddev=np.sqrt(1./(in_size+out_size)), dtype=tf.float32)) 
    b = tf.Variable(tf.zeros(out_size, dtype=tf.float32)) 
    return w, b 

def fc_layer(x, out_size): 
    w, b = get_weights(int(x.shape[-1]), out_size) 
    #tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(w))
    y = tf.add(tf.matmul(tf.cast(x, tf.float32), w), b) 
    return y 

def mixed_accuracy(x, y, mods):
    acc = tf.cast(tf.equal(x, y), tf.float32)
    bpsk_mask = tf.where(tf.equal(mods,'BPSK'), tf.ones(tf.shape(acc), dtype=tf.float32), tf.zeros(tf.shape(acc), dtype=tf.float32))
    acc_bpsk = tf.reduce_sum(tf.multiply(acc, bpsk_mask))
    acc_bpsk = tf.divide(acc_bpsk, tf.reduce_sum(bpsk_mask))
    pam4_mask = 1 - bpsk_mask
    acc_pam4 = tf.reduce_sum(tf.multiply(acc, pam4_mask))
    acc_pam4 = tf.divide(acc_pam4, tf.reduce_sum(pam4_mask))
    return acc_bpsk, acc_pam4
def batch_matvec_mul(A,b, transpose_a=False):
    '''Multiplies a matrix A of size batch_sizexNxK
       with a vector b of size batch_sizexK
       to produce the output of size batch_sizexN
    '''    
    C = tf.matmul(A, tf.expand_dims(b, axis=2), transpose_a=transpose_a)
    return tf.squeeze(C, -1) 

def emp_opt_layer(train_z, train_x, test_z, test_xid, constellation_):
    '''Emprically optimal minimizing norm2( x - x_est )
    Inputs:
        train_z: nd-array of shape [sample_size, NT], samples of denoiser input at the corresponding layer
        train_x: actual x values correspoding to each z e.g. ground truth
        test_z: the z samples we want to infer the x from
        constellation_: the constellation which x values come from
    Outputs: 
        test_x: Tensor of shape [batch_size, NT]
    '''
    nbins = 1000    
    NT = np.shape(train_z)[1]
    OPT_ = []
    for tx_ in range(NT):
        PROBS_ = []
        for cpoint_ in constellation_:     
            points_ = np.reshape(train_x[:,tx_]==cpoint_, (-1))
            train_n = train_z - train_x
            [vals_, bin_edges_] = np.histogram(train_n[points_, tx_], range=(-2.,2.), bins=nbins, normed=True)
            #[vals_, bin_edges_] = np.histogram(ndump0[:, tx_], range=(-2.,2.), bins=nbins, normed=True)
            bin_no = np.digitize(test_z[:,tx_]-cpoint_, bin_edges_) - 1
            outofrange = np.logical_or(bin_no==-1, bin_no==nbins)
            bin_no[outofrange] = 0
            prob_ = vals_[bin_no]
            prob_ += 1e-14
            prob_[outofrange] = 1e-10
            PROBS_.append(prob_)
        PROBS_ = np.array(PROBS_)
        PROBS_ /= np.sum(PROBS_, axis=0)
        #print np.shape(PROBS)
        #print const_dump
        optimal_x = np.matmul(np.reshape(constellation_, (1,-1)), PROBS_)
        OPT_.append(np.reshape(optimal_x, (1,-1)))
    OPT_ = np.reshape(OPT_, (NT, -1))
    OPT_ = np.transpose(OPT_)
    OPT_distance = np.reshape(OPT_, (-1,1)) - np.reshape(constellation_, (1,-1))
    OPT_id = np.argmin(np.abs(OPT_distance), axis=1)
    OPT_id = np.reshape(OPT_id, (-1, NT))
#    nbins_tf = 1000
#    OPT_ = []
#    for tx_tf in range(NT):
#        P_ = []
#        for cpoint_tf in range(4):
#            points_mask = tf.equal(indices_tf[:,tx_tf], cpoint_tf)
#            noise_tx = tf.boolean_mask(noise_tf[:,tx_tf], points_mask)
#            noise_tx = tf.clip_by_value(noise_tx, clip_value_min=-2., clip_value_max=2.)
#            fn_ = tf.histogram_fixed_width(noise_tx, [-2.,2.],nbins=nbins_tf)
#            fn_ = tf.cast(fn_, tf.float32)
#            fn_ /= tf.reduce_sum(fn_)
#               #bin_ = gen_math_ops.bucketize(tf.expand_dims(zstar[:,tx_tf], axis=1)-tf.reshape(constellation,[1,-1]),np.ndarray.tolist(np.linspace(-10.,10.,nbins_tf)))
#            test_noise = tf.expand_dims(zstar[:,tx_tf], axis=1)-constellation[cpoint_tf]
#            test_noise = tf.clip_by_value(test_noise, clip_value_min = -2., clip_value_max= 2.)
#            test_noise_, gholam_ind_ = sess.run([test_noise, indices], feed_dict)
#               #bin_ = gen_math_ops.bucketize(test_noise, np.ndarray.tolist(np.linspace(-2.,2.,nbins_tf)))
#            bin_ = np.digitize(test_noise_, np.linspace(-2.,2.,nbins_tf))
#            outofrange = np.logical_or(bin_==-1, bin_==nbins_tf)
#            bin_[outofrange] = 0
#               #bin_ = tf.clip_by_value(bin_, clip_value_min=1, clip_value_max=10000)
#               #bin_ -= 1
#            p_ = tf.gather(fn_, bin_)
#            #p_[outofrange] = 0.
#            p_ = tf.cast(p_, tf.float32) + 1e-10
#            P_.append(p_)
#               #p_ /= tf.reduce_sum(p_, axis=1, keepdims=True)
#               #print sess.run(p_, feed_dict)
#        Prob = tf.concat(P_, axis=1)
#        Prob /= tf.reduce_sum(Prob, axis=1, keepdims=True)    
#        emp_opt = tf.matmul(tf.reshape(Prob, [-1, 4]), tf.reshape(constellation, [4,1]))    
#        OPT_.append(emp_opt)
#    OPT_ = tf.concat(OPT_, axis=1)
#    emp_acc = accuracy(gholam_ind_, mimo.demodulate(OPT_, modtypes))
#    #print sess.run(OPT_, feed_dict)    
#    print 1. - sess.run(emp_acc, feed_dict)

    return np.mean(OPT_id == test_xid)

def zf(y, H):
    '''Zero-Forcing Detector
    Inputs:
        y: Tensor of shape [batch_size, N]
        H: Tensor of shape [batch_size, N, K]
        mod: Instance of the modulator class
    Outputs:
        indices: Tensor of shape [batch_size, K]
    '''
    # Projected channel output
    Hty = batch_matvec_mul(tf.transpose(H, perm=[0, 2, 1]), y)

    # Gramian of transposed channel matrix
    HtH = tf.matmul(H, H, transpose_a=True)

    # Inverse Gramian 
    HtHinv = tf.matrix_inverse(HtH)

    # ZF Detector
    x = batch_matvec_mul(HtHinv, Hty)
    
    return x

def mmse(y, H, noise_sigma):
    '''MMSE Detector
    Inputs:
        y: Tensor of shape [batch_size, N]
        H: Tensor of shape [batch_size, N, K]
        noise_sigma: Tensor of shape [batch_size]
    Outputs:
        indices: Tensor of shape [batch_size, K]
    '''
    # Projected channel output
    Hty = batch_matvec_mul(tf.transpose(H, perm=[0, 2, 1]), y)

    # Gramian of transposed channel matrix
    HtH = tf.matmul(H, H, transpose_a=True)
    # Inverse Gramian 
    HtHinv = tf.matrix_inverse(HtH + tf.reshape(tf.square(noise_sigma)/2, [-1, 1, 1]) * tf.eye(tf.shape(H)[-1], batch_shape=[tf.shape(H)[0]]))

    # ZF Detector
    x = batch_matvec_mul(HtHinv, Hty)
    
    return x

