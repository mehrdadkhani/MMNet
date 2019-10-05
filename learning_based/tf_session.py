import tensorflow as tf
from utils import *
from detector import detector
from loss import loss_fun, loss_yhx
from sample_generator import generator

class MMNet_graph():
    def __init__(self, params):
        self.params = params

    def build(self):

        with tf.device('/gpu:0'):
            tf.reset_default_graph()
            tf.set_random_seed(self.params['seed'])
            
            # Placeholders for feed dict
            batch_size = tf.placeholder(tf.int32, shape=(), name='batch_size')
            lr = tf.placeholder(tf.float32, shape=(), name='lr')
            snr_db_max = tf.placeholder(tf.float32, shape=(), name='snr_db_max')
            snr_db_min = tf.placeholder(tf.float32, shape=(), name='snr_db_min')
            train_flag = tf.placeholder(tf.bool, shape=(), name='train_flag')
            
            # MIMO sample generator model
            mimo = generator(self.params, batch_size)

            # Generate transmitt signals
            constellation = mimo.constellation
            indices = mimo.random_indices()
            x = mimo.modulate(indices)
            
            # Send x through the channel
            if self.params['data']:
                H = tf.placeholder(tf.float32, shape=(None, 2*self.params['N'], 2*self.params['K']), name='H')
                y, noise_sigma, actual_snrdB = mimo.channel(x, snr_db_min, snr_db_max, H, self.params['data'], self.params['correlation'])
            else:
                y, H, noise_sigma, actual_snrdB = mimo.channel(x, snr_db_min, snr_db_max, [], self.params['data'], self.params['correlation'])

            # Zero-forcing detection
            #x_mmse = mmse(y, H)
            x_mmse = mmse(y, H, noise_sigma)
            x_mmse_idx = demodulate(x_mmse, constellation)
            x_mmse = tf.gather(constellation, x_mmse_idx)
            acc_mmse = accuracy(indices, demodulate(x_mmse, constellation))
            
            # MMNet detection
            x_NN, helper = detector(self.params, constellation, x, y, H, noise_sigma, indices, batch_size).create_graph()
            loss = loss_fun(x_NN, x)
            #for i in range(10):
            #    print "y-Hx loss instead of xhat-x"
            #loss = loss_yhx(y, x_NN, H)
            for i in range(1):
                print("REPORTING MAX ACCURACY") 
            temp = []
            for i in range(self.params['L']):
                temp.append(accuracy(indices, demodulate(x_NN[i], constellation)))
                #acc_NN = accuracy(indices, mimo.demodulate(x_NN[train_layer_no-1], modtypes))
            acc_NN = tf.reduce_max(temp)

            
            # Training operation
            print("tf_session: Optimizing for the total loss")
            train_step = tf.train.AdamOptimizer(lr).minimize(tf.reduce_mean(loss))
            #train_step = tf.train.AdamOptimizer(lr).minimize(loss[train_layer_no-1])
            #train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss[train_layer_no-1])
            
            # Init operation
            init = tf.global_variables_initializer()
            
            
            # #### Training
            # define saver
            #saver = tf.train.Saver()
            
            # Create summary writer
            #merged = [0]
            #merged = tf.summary.merge_all()
            #print "merged"
            # Create session and initialize all variables
            sess = tf.Session()
            
            #train_writer = tf.summary.FileWriter('./reports/'+self.params['save_name']+'/log/train', sess.graph)
            #test_writer = tf.summary.FileWriter('./reports/'+self.params['save_name']+'/log/test', sess.graph)
            
            if len(self.params['start_from'])>1:
                saver.restore(sess, self.params['start_from'])
            else:
                sess.run(init)
            
            nodes = {'measured_snr':actual_snrdB, 'batch_size':batch_size, 'lr': lr, 'snr_db_min': snr_db_min, 'snr_db_max':snr_db_max, 'x': x, 'x_id': indices, 'H': H, 'y': y, 'sess': sess, 'train': train_step, 'accuracy': acc_NN, 'loss': loss, 'mmse_accuracy': acc_mmse, 'constellation': constellation, 'logs': helper, 'init': init}
        return nodes
