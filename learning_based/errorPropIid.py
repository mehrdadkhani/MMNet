import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_math_ops
import os
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as pl
from tf_session import *
import pickle
from parser import parse

params, args = parse()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def complex_to_real(inp):
    Hr = np.real(inp)
    Hi = np.imag(inp)
    h1 = np.concatenate([Hr, -Hi], axis=2)
    h2 = np.concatenate([Hi,  Hr], axis=2)
    inp = np.concatenate([h1, h2], axis=1)
    return inp

from exp import get_data

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
train_layer_no = nodes['train_layer_no']
logs = nodes['logs']
measured_snr = nodes['measured_snr']
init = nodes['init']
# Training loop
tln_ = args.train_layer


record = {'before':[], 'after':[]}
record_flag =False

if args.data:
    train_data = train_data_ref
    test_data  = test_data_ref
else:
    test_data = []
    train_data = [] 
print('training on:', train_data_ref.shape)    
rndIndx = np.random.randint(0, train_data_ref.shape[0], 100)
print(rndIndx)
train_data_ref = train_data_ref[rndIndx]
test_data_ref = test_data_ref[rndIndx]

# pretrain
sess.run(init)
feed_dict = {
            batch_size: args.batch_size,
            lr: args.learn_rate,
            snr_db_max: params['SNR_dB_max'],
            snr_db_min: params['SNR_dB_min'],
            train_layer_no: tln_,
        }
for i in range(50000):
        sample_ids = np.random.randint(0, np.shape(train_data)[0], params['batch_size'])
        feed_dict[H] = train_data[sample_ids]
        sess.run(train, feed_dict) 

print('done with training')

for r in range(100):
    train_data = np.expand_dims(train_data_ref[r], axis=0)
    test_data = np.expand_dims(test_data_ref[r], axis=0)
    results = {}
    # Test
    for snr_ in range(int(params['SNR_dB_min']), int(params['SNR_dB_max'])+1):
        feed_dict = {
                batch_size: args.test_batch_size,
                snr_db_max: snr_,
                snr_db_min: snr_,
                train_layer_no: tln_,
            }
        if args.data:
            sample_ids = np.random.randint(0, np.shape(test_data)[0], args.test_batch_size)
            feed_dict[H] = test_data[sample_ids]
    
        test_accuracy_, test_loss_, measured_snr_, log_ = sess.run([accuracy, loss, measured_snr, logs], feed_dict)   
        print((r, 'at layer', tln_, 'SER: {:.2E}'.format(1. - test_accuracy_), test_loss_, measured_snr_))
        results[str(snr_)] = {}
        for k in log_:
            results[str(snr_)][k] = log_[k]['stat']
        results[str(snr_)]['accuracy'] = test_accuracy_
    results['cond'] = np.linalg.cond(test_data[sample_ids][0])
    path = '/data3/iid_%s_NT%sNR%s_%s_%s/'%(args.modulation,args.x_size, args.y_size, args.exp,args.linear)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+'reults%d.pkl'%r, 'wb') as f:
        pickle.dump(results, f)
result = model_eval(test_data, params['SNR_dB_min'], params['SNR_dB_max'], mmse_accuracy, accuracy, batch_size, snr_db_min, snr_db_max, H, sess)
print(result)
