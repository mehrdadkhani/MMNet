import argparse

def parse():
    parser = argparse.ArgumentParser(description='MIMO signal detection benchmark')
    
    parser.add_argument('--numSamples',
            type = int,
            required=False,
            help = 'Fast benchmark numSamples')
    parser.add_argument('--goTo',
            type = int,
            required=False,
            help = 'Fast benchmark indexing')
    
    parser.add_argument('--x-size', '-xs',
            type = int,
            required=True,
            help = 'Number of senders')
    
    parser.add_argument('--y-size', '-ys',
            type = int,
            required=True,
            help = 'Number of receivers')
    
    parser.add_argument('--snr-min',
            type = float,
            required=True,
            help = 'Minimum SNR in dB')
    
    parser.add_argument('--snr-max',
            type = float,
            required=True,
            help = 'Maximum SNR in dB')
    
    parser.add_argument('--batch-size',
            type = int,
            required=True,
            help = 'Batch size')
    
    parser.add_argument('--modulation', '-mod',
            type = str,
            required=True,
            help = 'Modulation type which can be BPSK, 4PAM, or MIXED')
    
    parser.add_argument('--overwrite',
            action = 'store_true',
            help = 'Overwrite the results into the file')
    
    parser.add_argument('--correlated',
            action = 'store_true',
            help = 'Use correlated channel')
    
    parser.add_argument('--ML',
            action = 'store_true',
            help = 'Include Maximum Likielihood')
    
    parser.add_argument('--AMP',
            action = 'store_true',
            help = 'Include Approximate Message Passing')
    
    parser.add_argument('--SDR',
            action = 'store_true',
            help = 'Include SDR detection algorithm')

    parser.add_argument('--BLAST',
            action = 'store_true',
            help = 'Include BLAST detection algorithm')

    parser.add_argument('--MMSE',
            action = 'store_true',
            help = 'Include Zero Forcing')
    
    parser.add_argument('--data',
            action = 'store_true',
            help = 'Load H from dataset')
    
    parser.add_argument('--data-dir',
            type = str,
            required=True,
            help = 'Channel data directory')

    parser.add_argument('--parallel',
            action = 'store_true',
            help = 'Parallelize the ML solver')
    args = parser.parse_args()
    return args
