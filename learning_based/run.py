import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_math_ops
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tf_session import *
import argparse
import pickle


parser = argparse.ArgumentParser(description='MIMO signal detection simulator')

parser.add_argument('--x-size', '-xs',
        type = int,
        required=True,
        help = 'Number of senders')

parser.add_argument('--y-size', '-ys',
        type = int,
        required=True,
        help = 'Number of receivers')

parser.add_argument('--layers',
        type = int,
        required=True,
        help = 'Number of neural net blocks')

parser.add_argument('--snr-min',
        type = float,
        required=True,
        help = 'Minimum SNR in dB')

parser.add_argument('--snr-max',
        type = float,
        required=True,
        help = 'Maximum SNR in dB')

parser.add_argument('--start-from',
        type = str,
        required=False,
    default='',
        help = 'Saved model name to start from')

parser.add_argument('--learn-rate', '-lr',
        type = float,
        required=True,
        help = 'Learning rate')

parser.add_argument('--batch-size',
        type = int,
        required=True,
        help = 'Batch size')

parser.add_argument('--test-every',
        type = int,
        required=True,
        help = 'number of training iterations before each test')

parser.add_argument('--train-iterations',
        type = int,
        required=True,
        help = 'Number of training iterations')

parser.add_argument('--modulation', '-mod',
        type = str,
        required=True,
        help = 'Modulation type which can be BPSK, 4PAM, or MIXED')

parser.add_argument('--gpu',
        type = str,
        required=False,
    default="0",
        help = 'Specify the gpu core')

parser.add_argument('--saveas',
        type = str,
        required=True,
        help = 'Path to save the model')

parser.add_argument('--test-batch-size',
        type = int,
        required=True,
        help = 'Size of the test batch')

parser.add_argument('--search',
        type = bool,
        required=False,
    default=False,
        help = 'Apply a distance-1 search on the output and choose the best result')

parser.add_argument('--log',
        action = 'store_true',
        help = 'Log data mode')

parser.add_argument('--correlated',
        type = bool,
        required=False,
    default=False,
        help = 'Use correlated channel')

parser.add_argument('--train-layer',
        type = int,
        required=True,
        default=0,
        help = 'Number of the layer to train')

parser.add_argument('--data',
        action = 'store_true',
        help = 'Use dataset to train/test')

parser.add_argument('--index',
        type = int,
        required=False,
        default=-1,
        help = 'H batch index')

parser.add_argument('--linear',
        type = str,
        required = True,
        help = 'linear transformation step method')

parser.add_argument('--denoiser',
        type = str,
        required = True,
        help = 'denoiser function model')

parser.add_argument('--exp',
        type = str,
        required = False,
        help = 'experiment name')

parser.add_argument('--corr-analysis',
        action = 'store_true',
        help = 'fetch covariance matrices')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu # Ignore if you do not have multiple GPUs

# Simulation parameters
params = {
    'N' : args.y_size, # Number of receive antennas
    'K' : args.x_size, # Number of transmit antennas
    'L' : args.layers, # Number of layers
    'SNR_dB_min' : args.snr_min, # Minimum SNR value in dB for training and evaluation
    'SNR_dB_max' : args.snr_max, # Maximum SNR value in dB for training and evaluation
    'seed' : 1, # Seed for random number generation
    'batch_size': args.batch_size,
    'modulation': args.modulation,
    'TL': args.train_layer,
    'correlation': args.correlated,
    'save_name': args.saveas,
    'start_from': args.start_from,
    'data': args.data,
    'linear_name': args.linear,
    'denoiser_name': args.denoiser
}

def complex_to_real(inp):
    Hr = np.real(inp)
    Hi = np.imag(inp)
    h1 = np.concatenate([Hr, -Hi], axis=2)
    h2 = np.concatenate([Hi,  Hr], axis=2)
    inp = np.concatenate([h1, h2], axis=1)
    return inp

from exp import get_data

#if args.data:
#    H_dataset = np.load('/data/Mehrdad/channel_sequences_normalized.npy')[0,:]
#    H_dataset = np.reshape(H_dataset, (-1, args.y_size, args.x_size))
#    H_dataset = complex_to_real(H_dataset)
#    Hdataset_powerdB = 10. * np.log(np.mean(np.sum(H_dataset ** 2, axis=1))) / np.log(10.) 
#    print Hdataset_powerdB
#
#    train_data = H_dataset[0:int(0.8*np.shape(H_dataset)[0])]
#    test_data = H_dataset[int(0.8*np.shape(H_dataset)[0])+1:-1]
    #print "H is fixed" # dont forget to set the above
    #train_data = np.array([H_dataset[0]])
    #test_data = np.array([H_dataset[0]])
if args.data:
    train_data_ref, test_data_ref, Hdataset_powerdB = get_data(args.exp)
    print(train_data_ref.shape)
    params['Hdataset_powerdB'] = Hdataset_powerdB


# Build the computational graph
mmnet = MMNet_graph(params)
nodes = mmnet.build() 

# Get access to the nodes on the graph
sess = nodes['sess']
x = nodes['x']
H = nodes['H']
x_id = nodes['x_id']
constellation = nodes['constellation']
train = nodes['train']
summary = nodes['summary']
snr_db_min = nodes['snr_db_min']
snr_db_max = nodes['snr_db_max']
lr = nodes['lr']
batch_size = nodes['batch_size']
accuracy = nodes['accuracy']
mmse_accuracy = nodes['mmse_accuracy']
loss = nodes['loss']
test_summary_writer = nodes['test_summary_writer']
train_summary_writer = nodes['train_summary_writer']
saver = nodes['saver']
train_layer_no = nodes['train_layer_no']
logs = nodes['logs']
measured_snr = nodes['measured_snr']
# Training loop
tln_ = args.train_layer

#for t in range(20):
#    print "~ meta learning on H"

record = {'before':[], 'after':[]}
record_flag =False

if args.data:
    train_data = train_data_ref
    test_data  = test_data_ref
else:
    test_data = []
    train_data = [] 
for it in range(args.train_iterations):  
    #pertr = np.random.normal(0.,0.01, (5000, 64, 16))
    #perti = np.random.normal(0.,0.01, (5000, 64, 16))
    #pert = np.concatenate([np.concatenate([pertr,-perti], axis=2), np.concatenate([perti,pertr], axis=2)], axis=1)
    #train_data = train_data_ref + pert
    #test_data  = test_data_ref  + pert
    # Train:
    
#    if it % 10000000 == 0:
#        print "H is fixed"
#        rnd_ = 50#np.random.randint(0, 0.8 * H_dataset.shape[0])
        #train_data = np.array([H_dataset[rnd_]])
        #test_data = np.array([H_dataset[rnd_]])
    
    feed_dict = {
                batch_size: args.batch_size,
                lr: args.learn_rate,
                snr_db_max: params['SNR_dB_max'],
                snr_db_min: params['SNR_dB_min'],
                train_layer_no: tln_,
            }
    if args.data:
        sample_ids = np.random.randint(0, np.shape(train_data)[0], params['batch_size'])
        feed_dict[H] = train_data[sample_ids]
    if record_flag:
        feed_dict_test = {
                    batch_size: args.test_batch_size,
                    lr: args.learn_rate,
                    snr_db_max: params['SNR_dB_max'],
                    snr_db_min: params['SNR_dB_min'],
                    train_layer_no: tln_,
                    #H: train_data[np.tile(sample_ids[0],(args.test_batch_size))],
                }
        if args.data:
            sample_ids = np.random.randint(0, np.shape(test_data)[0], args.test_batch_size)
            feed_dict[H] = test_data[sample_ids]
        before_acc = 1.-sess.run(accuracy, feed_dict_test)
        record['before'].append(before_acc)

    _, train_summary_ = sess.run([train, summary], feed_dict)

    # Test
    if (it % args.test_every==0):
        feed_dict = {
                batch_size: args.test_batch_size,
                snr_db_max: params['SNR_dB_max'],
                snr_db_min: params['SNR_dB_max'],
                train_layer_no: tln_,
            }
        if args.data:
            sample_ids = np.random.randint(0, np.shape(test_data)[0], args.test_batch_size)
            feed_dict[H] = test_data[sample_ids]
        if args.log:
            #test_accuracy_, test_loss_, logs_, x_, H_, test_summary_= sess.run([accuracy, loss, logs, x, H, summary], feed_dict)
            test_accuracy_, test_loss_, logs_, x_, H_= sess.run([accuracy, loss, logs, x, H], feed_dict)
            np.save('log.npy', logs_)
            break
        else:
            #if args.data:
                #result = model_eval(test_data, params['SNR_dB_min'], params['SNR_dB_max'], mmse_accuracy, accuracy, batch_size, snr_db_min, snr_db_max, H, sess)
                #print "iteration", it, result
            test_accuracy_, test_loss_, test_summary_, measured_snr_ = sess.run([accuracy, loss, summary, measured_snr], feed_dict)   
            print((it, 'at layer', tln_, 'SER: {:.2E}'.format(1. - test_accuracy_), test_loss_, measured_snr_))
        if args.corr_analysis:
            log_ = sess.run(logs, feed_dict)
            for l in range(1,int(args.layers)+1):
                c = log_['layer'+str(l)]['linear']['I_WH']
                print((np.linalg.norm(c, axis=(1,2))[0]))
            
            #temp2 = log_['layer'+str(l)]['linear']
            #np.save('W'+str(l)+'.npy', temp2['W'])
            #np.save('H'+str(l)+'.npy', temp2['H'])
            #np.save('WsetR'+str(l)+'.npy', temp2['WsetR'])
            #np.save('WsetI'+str(l)+'.npy', temp2['WsetI'])
            #print "Written"
            
            #np.save('Vw'+str(l)+'.npy', temp2['svd'][2][0])
            #np.save('Uh'+str(l)+'.npy', temp2['svd'][4][0])
            #np.save('Uw'+str(l)+'.npy', temp2['svd'][1][0])
            #np.save('WH'+str(l)+'.npy', temp2['WH'][0])
            #print np.matmul(temp2[1][0], temp2[4][0].conj().T)
            #print temp2['VhVwt'][0]
            #print temp2['norm_diff_U'][0], temp2['norm_Uw'][0], temp2['norm_diff_V'][0], temp2['norm_V'][0]
        saver.save(sess, './reports/'+args.saveas, global_step=it)
        test_summary_writer.add_summary(test_summary_, it)
        train_summary_writer.add_summary(train_summary_, it)

result = model_eval(test_data, params['SNR_dB_min'], params['SNR_dB_max'], mmse_accuracy, accuracy, batch_size, snr_db_min, snr_db_max, H, sess)
print(result)
#SNR_dBs = np.linspace(params['SNR_dB_min'],params['SNR_dB_max'],params['SNR_dB_max']-params['SNR_dB_min']+1)
#accs_mmse = np.zeros(shape=SNR_dBs.shape)
#accs_NN = np.zeros(shape=SNR_dBs.shape)
#iterations = 30
## SER Simulations
##print "PERTURBATIONS ON"
#if args.data:
#    test_data = test_data_ref
#for i in range(SNR_dBs.shape[0]):
#    noise_ = []
#    error_ = []
#    for j in range(iterations):
#        #pertr = np.random.normal(0.,0.01, (5000, 64, 16))
#        #perti = np.random.normal(0.,0.01, (5000, 64, 16))
#        #pert = np.concatenate([np.concatenate([pertr,-perti], axis=2), np.concatenate([perti,pertr], axis=2)], axis=1)
#        #test_data  = test_data_ref  + pert
#        feed_dict = {
#                batch_size: 5000,
#                snr_db_max: SNR_dBs[i],
#                snr_db_min: SNR_dBs[i],
#                train_layer_no: params['TL'],
#            }    
#        if args.data:
#            sample_ids = np.random.randint(0, np.shape(test_data)[0], 5000)
#            feed_dict[H] = test_data[sample_ids]
#        acc = sess.run([mmse_accuracy, accuracy], feed_dict)
#        accs_mmse[i] += acc[0]/iterations
#        accs_NN[i] += acc[1]/iterations
#    print "SER_mmse: ", 1. - accs_mmse 
#    print "SER_NN: ", 1. - accs_NN
