import os
import random
import numpy as np
import pickle
import json
from sdrSolver import solvers
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool as ThreadPool 
import itertools
from parser import parse


args = parse()
np.random.seed(123)

NT = args.x_size
NR = args.y_size

if args.modulation.split('_')[0] == 'QAM':
    M = np.sqrt(int(args.modulation.split('_')[1]))
    modulation = args.modulation.split('_')[0]
else:
    raise Exception

sample_size = args.batch_size

#snrMinBit = args.snr_min + 10. * np.log10(np.log2(M**2))
#snrMaxBit = args.snr_max + 10. * np.log10(np.log2(M**2))
#snr_range = np.arange(snrMinBit, snrMaxBit+1)
snr_range = np.arange(args.snr_min, args.snr_max+1)

sigConst = np.linspace(-M+1, M-1, M) 
sigConst /= np.sqrt((sigConst ** 2).mean())
sigConst /= np.sqrt(2.) #Each complex transmitted signal will have two parts

print("==========================================================================================")
print("Calculating the SDR for the SNR range (dB) ****")
print(snr_range)
complex_constel = [np.complex(i,j) for i in sigConst for j in sigConst]
print('For the signal constellation ****************** ')
print(np.around(complex_constel, 2))
print("With average power of *************************") 
print(np.mean(np.abs(complex_constel) ** 2))
print("==========================================================================================")


baseline = {}
baseline['constellation'] = str(sigConst)

def complex_to_real(inp):
    Hr = np.real(inp)
    Hi = np.imag(inp)
    h1 = np.concatenate([Hr, -Hi], axis=2)
    h2 = np.concatenate([Hi,  Hr], axis=2)
    inp = np.concatenate([h1, h2], axis=1)
    return inp

def generate_data_qam(sample_size, repeat_size, sigConst, SNR, NT, NR, correlated):
    if correlated == True:
        gains = np.random.uniform(0., 20., size=[sample_size, 1, NT])
        gains = 10. ** (gains/20.)
        normalize_factor = np.mean(gains ** 2, axis=(0,1,2))
        normalize_factor = np.sqrt(normalize_factor) 
        gains /= normalize_factor
        hBatchr *= gains
        hBatchi *= gains

    if args.data:
        #hBatch = np.load('/data/Mehrdad/channel_sequences_normalized.npy')[:,0] 
        #hBatch = np.load('/data/Mehrdad/Jakobs_channels_normalized_32.npy')[:,0]#[args.index]#[0:int(1e5)]
        hBatch = np.load(args.data_dir)
        hBatch = np.reshape(hBatch, (-1, args.y_size, args.x_size))
        print(hBatch.shape)
        hBatch = complex_to_real(hBatch)
        H_powerdB = 10. * np.log10(np.mean(np.sum(hBatch ** 2, axis=1)))
    else:
        hBatchr = np.random.normal(0., 1./np.sqrt(2.*NR), size=[sample_size, NR, NT])
        hBatchi = np.random.normal(0., 1./np.sqrt(2.*NR), size=[sample_size, NR, NT])
        h1 = np.concatenate([hBatchr, -hBatchi], axis=2)
        h2 = np.concatenate([hBatchi, hBatchr], axis=2)
        hBatch = np.concatenate([h1, h2], axis=1)
        H_powerdB = 0.

    print("Channel powers (dB):", H_powerdB)
    average_H_powerdB = np.mean(H_powerdB)
    if args.data:
        idx_ = np.array([37,235,908,72,767,905,715,645,847,960,144,129,972,583,749,508,390,281    
        ,178,276,254,357,914,468,907,252,490,668,925,398,562,580,215,983,753,503  
        ,478,864,86,141,393,7,319,829,534,313,513,896,316,209,264,728,653,627     
        ,431,633,456,542,71,387,454,917,561,313,515,964,792,497,43,588,26,820     
        ,336,621,883,297,466,15,64,196,25,367,738,471,903,282,665,616,22,777      
        ,707,999,126,279,381,356,155,933,313,595])[args.goTo]                                  
        #hBatch = hBatch[np.random.randint(0,hBatch.shape[0], sample_size)]
        hBatch = np.expand_dims(hBatch[idx_], axis=0)
        print(hBatch.shape)
        hBatch = np.tile(hBatch, (repeat_size, 1, 1))
        print(hBatch.shape)
    xBatch = np.random.randint(0, len(sigConst), size=[sample_size * repeat_size, 2*NT, 1])
    sBatch = np.take(sigConst, xBatch)
    average_s_powerdB = 10. * np.log10(np.mean(np.sum(sBatch ** 2, axis=1)))
    print("Average Transmitter power (dB):", average_s_powerdB) 
    wBatch = np.random.normal(0., 1./np.sqrt(2.), size=[sample_size * repeat_size, 2*NR, 1])
    temp_powwdB = 10. * np.log10(NR)
    wBatch *= (10. ** ((10.*np.log10(NT) + H_powerdB - SNR - 10.*np.log10(NR)) / 20.))
    average_noise_powerdB = 10. * np.log10(np.mean(np.sum(wBatch ** 2, axis=1), axis=0))
    print("Average Noise power(dB):", average_noise_powerdB)
    sigBatch = np.matmul(hBatch, sBatch) 
    average_sig_powerdB = 10. * np.log10(np.mean(np.sum(sigBatch ** 2, axis=1), axis=0))
    print("Average Signal Power(dB):", average_sig_powerdB)
    print("Actual SNR(dB):", average_sig_powerdB - average_noise_powerdB)
    
    yBatch = sigBatch + wBatch
    hthBatch = np.matmul(np.transpose(hBatch, axes=(0,2,1)), hBatch) 
    return xBatch, sBatch, yBatch, hBatch, hthBatch, average_H_powerdB

def generate_data_pam(sample_size, sigConst, SNR, NT, NR):
    hBatch = np.random.normal(0., 1./np.sqrt(NR), size=[sample_size, NR, NT])
#    hBatch /= np.sqrt((sigConst ** 2).mean())
    xBatch = np.random.randint(0, len(sigConst), size=[sample_size, NT, 1])
    #hBatch *= (10. ** (13/20.))
    sBatch = np.take(sigConst, xBatch)
    #sBatch *= (np.sqrt(1.) * 10. ** (SNR/20.))
    wBatch = np.random.normal(0., 1., size=[sample_size, NR, 1])
    wBatch *= (10. ** (-SNR/20.)) 
    yBatch = np.matmul(hBatch, sBatch) + wBatch
    hthBatch = np.matmul(np.transpose(hBatch, axes=(0,2,1)), hBatch) 
    return xBatch, sBatch, yBatch, hBatch, hthBatch

def generate_data(modulation, sample_size, repeat_size, sigConst, snr, NT, NR, correlated):
    if modulation == "QAM":
        return generate_data_qam(sample_size, repeat_size, sigConst, snr, NT, NR, correlated)
    elif modulation == "PAM":
        return generate_data_pam(sample_size, sigConst, snr, NT, NR, correlated)
    elif modulation == "BPSK":
        return generate_data_bpsk(sample_size, sigConst, snr, NT, NR, correlated)
    else:
        raise Exception

def amp_eval(sample_size, repeat_size, sigConst, modulation, snr_range, NT, NR, correlated):
    print("Running AMP evaluation")
    result = {}
    for snr_ in snr_range:
        xBatch, sBatch, yBatch, hBatch, hthBatch, average_H_powerdB = generate_data(modulation, sample_size, repeat_size, sigConst, snr_, NT, NR, correlated)
        complexnoise_sigma = 10. ** ((10.*np.log10(NT) + average_H_powerdB - snr_ - 10.*np.log10(NR))/20.) 
        s = solvers()
        shatBatch = s.ampSolver(hBatch, yBatch, sigConst, complexnoise_sigma)
        xhatBatch = np.argmin(np.abs(np.reshape(shatBatch,[1,-1]) - sigConst.reshape([-1,1])),axis=0)
        xhatBatch = xhatBatch.reshape([sample_size, -1])
        eq = np.equal(xhatBatch, np.reshape(xBatch, [sample_size * repeat_size,-1]))
        acc = eq.mean(axis=1)
        acc = np.mean(np.reshape(acc, (repeat_size,-1)), axis=1) 
        result[str(snr_)] = acc
        h = np.reshape(hBatch,[-1,sample_size, hBatch.shape[1], hBatch.shape[2]])
        result['cond'] = np.linalg.cond(h[0])
                                                         
    return result                             

def ml_proc(hBatch, yBatch):
    s = solvers()
    shatBatch = s.mlSolver(np.array([hBatch]), np.array([yBatch]), sigConst)
    return shatBatch

def ml_proc_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return ml_proc(*a_b)

def ml_eval(sample_size, repeat_size, sigConst, modulation, snr_range, NT, NR, correlated):
    print("Running ML evaluation")
    result = {}
    #num_cores = multiprocessing.cpu_count()
    pool = ThreadPool(40) 
    for snr_ in snr_range:
        xBatch, sBatch, yBatch, hBatch, hthBatch, average_H_powerdB = generate_data(modulation, sample_size, repeat_size, sigConst, snr_, NT, NR, correlated)
        #shatBatch = Parallel(n_jobs=num_cores)(delayed(ml_proc(np.array([hBatch[i]]), np.array([yBatch[i]]), sigConst)) for i in range(hBatch.shape[0]))
        if args.parallel:
            shatBatch = pool.map(ml_proc_star, zip(hBatch, yBatch))
        else:
            s = solvers()
            #print "noisy channel estimations activated"
            #hBatch_est = np.random.normal(1., 0.1, hBatch.shape) * hBatch
            shatBatch = s.mlSolver(hBatch, yBatch, sigConst)
        xhatBatch = np.argmin(np.abs(np.reshape(shatBatch,[1,-1]) - sigConst.reshape([-1,1])),axis=0)
        xhatBatch = xhatBatch.reshape([sample_size * repeat_size, -1])
        eq = np.equal(xhatBatch, np.reshape(xBatch, [sample_size * repeat_size,-1]))
        #acc = eq.mean()  
        acc = eq.mean(axis=1)
        acc = np.mean(np.reshape(acc, (repeat_size,-1)), axis=1) 
        result[str(snr_)] = acc
        h = np.reshape(hBatch,[-1,sample_size, hBatch.shape[1], hBatch.shape[2]])
        result['cond'] = np.linalg.cond(h[0])
    return result                         


def mmse_eval(sample_size, repeat_size, sigConst, modulation, snr_range, NT, NR, correlated):
    print("Running MMSE evaluation")
    result = {}
    for snr_ in snr_range:
        xBatch, sBatch, yBatch, hBatch, hthBatch, average_H_powerdB = generate_data(modulation, sample_size, repeat_size, sigConst, snr_, NT, NR, correlated)
        complexnoise_sigma = 10. ** ((10.*np.log10(NT) + average_H_powerdB - snr_ - 10.*np.log10(NR))/20.) 
        shatBatch = np.zeros([sample_size * repeat_size, len(hBatch[0,0,:])])
        for i, h in enumerate(hBatch):
            shatBatch[i] = np.reshape(np.matmul(np.matmul(np.linalg.pinv(np.matmul(hBatch[i].T, hBatch[i]) + complexnoise_sigma ** 2 / 2 * np.eye(2*NT)), hBatch[i].T), np.matrix(yBatch[i])), len(hBatch[0,0,:]))
        xhatBatch = np.argmin(np.abs(np.reshape(shatBatch,[1,-1]) - sigConst.reshape([-1,1])),axis=0)
        xhatBatch = xhatBatch.reshape([sample_size * repeat_size, -1])
        eq = np.equal(xhatBatch, np.reshape(xBatch, [sample_size * repeat_size,-1]))
        acc = eq.mean(axis=1)
        acc = np.mean(np.reshape(acc, (repeat_size,1)), axis=1) 
        result[str(snr_)] = acc
        h = np.reshape(hBatch,[-1,sample_size, hBatch.shape[1], hBatch.shape[2]])
        result['cond'] = np.linalg.cond(h[0])

    return result                         

def getDir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

if __name__ == "__main__":
    result = {}
    print("repeat is set to 100")
    if args.ML == True:
        result['ML'] = ml_eval(args.batch_size, args.numSamples, sigConst, modulation, snr_range, NT, NR, args.correlated)
        for snr_ in result['ML']:
            if not snr_=='cond':
                print((float(snr_), 1. - np.mean(result['ML'][snr_])))
    if args.AMP == True:
        result['AMP'] = amp_eval(args.batch_size, 1000, sigConst, modulation, snr_range, NT, NR, args.correlated)
        for snr_ in result['AMP']:
            if not snr_=='cond':
                print((float(snr_), 1. - np.mean(result['AMP'][snr_])))
    if args.MMSE == True:
        result['MMSE'] = mmse_eval(args.batch_size, 1000, sigConst, modulation, snr_range, NT, NR, args.correlated)
        for snr_ in result['MMSE']:
            if not snr_=='cond':
                print((float(snr_), 1. - np.mean(result['MMSE'][snr_])))

    
#for snr_ in snrRange:
#    acc_sdr = ampCalc(sample_size, sigConst, snr_, NT, NR)
#    baseline[str(snr_)] = str(acc_sdr)
#    print snr_, acc_sdr
#baseline = json.dumps(result)
    path_ml = "./baseline/ML/%s/%s/NT%d/NR%d/"%(args.modulation, args.data_dir.split('/')[-1].rstrip('.npy'), NT, NR)
    path_mmse = "./baseline/MMSE/%s/%s/NT%d/NR%d/"%(args.modulation, args.data_dir.split('/')[-1].rstrip('.npy'), NT, NR)
    #if not os.path.exists(path_ml):
    #    os.makedirs(path_ml)
    #if not os.path.exists(path_mmse):
    #    os.makedirs(path_mmse)

    if args.overwrite == True:
        if args.ML:
            assert args.snr_min == args.snr_max
            path = getDir(path_ml+'snr%s'%args.snr_max)
            fName = path+'/results%d.pkl'%(args.goTo)
            with open(fName, 'wb') as f:
                pickle.dump(result['ML'], f)
                print("dumped ML successfuly at %s"%(fName))
        if args.MMSE:
            assert args.snr_min == args.snr_max
            path = getDir(path_mmse+'snr%s'%args.snr_max)
            fName = path+'/results%d.pkl'%(args.goTo)
            with open(fName, 'wb') as f:
                pickle.dump(result['MMSE'], f)
                print("dumped MMSE successfuly at %s"%(fName))
            
#    with open(file_name, 'w') as f:
#        f.write(baseline)
#
