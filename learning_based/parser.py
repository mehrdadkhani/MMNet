import argparse

def parse():
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
            required=False,
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
            required=False,
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

    parser.add_argument('--output-dir',
            type = str,
            required = True,
            help = 'Directory for saving the results')
    
    parser.add_argument('--channels-dir',
            type = str,
            required = True,
            help = 'Path for reading the channel dataset from')

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
        'correlation': args.correlated,
        'save_name': args.saveas,
        'start_from': args.start_from,
        'data': args.data,
        'linear_name': args.linear,
        'denoiser_name': args.denoiser
    }
    return params, args
