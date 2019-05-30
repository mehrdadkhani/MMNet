import os
import random
import numpy as np
import pickle
import json
from genSolver import solvers
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
        if args.data_dir.endswith('H_iid.npy'):
            idx_ = np.array([37,235,908,72,767,905,715,645,847,960,144,129,972,583,749,508,390,281    
        ,178,276,254,357,914,468,907,252,490,668,925,398,562,580,215,983,753,503  
        ,478,864,86,141,393,7,319,829,534,313,513,896,316,209,264,728,653,627     
        ,431,633,456,542,71,387,454,917,561,313,515,964,792,497,43,588,26,820     
        ,336,621,883,297,466,15,64,196,25,367,738,471,903,282,665,616,22,777      
        ,707,999,126,279,381,356,155,933,313,595])[:sample_size]                           
        else:
            idx_ = np.arange(0,sample_size)
        #hBatch = hBatch[np.random.randint(0,hBatch.shape[0], sample_size)]
        hBatch = hBatch[idx_]
        hBatch = np.reshape(np.tile(np.expand_dims(hBatch, axis=0), (repeat_size, 1, 1, 1)), (-1, hBatch.shape[1], hBatch.shape[2]))
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
        xhatBatch = xhatBatch.reshape([sample_size*repeat_size, -1])

        eq = np.equal(xhatBatch, np.reshape(xBatch, [sample_size * repeat_size,-1]))
        acc = eq.mean(axis=1)
        acc = np.mean(np.reshape(acc, (repeat_size,-1)), axis=0) 
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
            shatBatch, status = s.mlSolver(hBatch, yBatch, sigConst)
        xhatBatch = np.argmin(np.abs(np.reshape(shatBatch,[1,-1]) - sigConst.reshape([-1,1])),axis=0)
        xhatBatch = xhatBatch.reshape([sample_size * repeat_size, -1])
        eq = np.equal(xhatBatch, np.reshape(xBatch, [sample_size * repeat_size,-1]))
        
        #introducing bias
        #status = np.reshape(status, (repeat_size,-1))
        #eq = np.reshape(eq, (repeat_size, sample_size, -1))
        #print(status.shape)
        #print(eq.shape)
        #acc = np.zeros((sample_size,1))
        #for i in range(sample_size):
        #    acc[i] = np.mean(eq[status[:,i]==1,i])
                
        acc = eq.mean(axis=1)
        acc = np.mean(np.reshape(acc, (repeat_size,-1)), axis=0) 
        print(acc.shape)
        result[str(snr_)] = acc
        h = np.reshape(hBatch,[-1,sample_size, hBatch.shape[1], hBatch.shape[2]])
        result['cond'] = np.linalg.cond(h[0])
    return result                         

def sdr_eval(sample_size, repeat_size, sigConst, modulation, snr_range, NT, NR, correlated):
    print("Running SDR evaluation")
    result = {}
    for snr_ in snr_range:
        xBatch, sBatch, yBatch, hBatch, hthBatch, average_H_powerdB = generate_data(modulation, sample_size, repeat_size, sigConst, snr_, NT, NR, correlated)
        s = solvers()
        shatBatch = s.sdrSolver(hBatch, yBatch, sigConst, NT)
        xhatBatch = np.argmin(np.abs(np.reshape(shatBatch,[1,-1]) - sigConst.reshape([-1,1])),axis=0)
        xhatBatch = xhatBatch.reshape([sample_size * repeat_size, -1])
        eq = np.equal(xhatBatch, np.reshape(xBatch, [sample_size * repeat_size,-1]))
        #acc = eq.mean()  
        acc = eq.mean(axis=1)
        acc = np.mean(np.reshape(acc, (repeat_size,-1)), axis=0) 
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
            shatBatch[i] = np.reshape(np.matmul(np.matmul(np.linalg.inv(np.matmul(hBatch[i].T, hBatch[i]) + complexnoise_sigma ** 2 / 2 * np.eye(2*NT)), hBatch[i].T), np.matrix(yBatch[i])), len(hBatch[0,0,:]))
        xhatBatch = np.argmin(np.abs(np.reshape(shatBatch,[1,-1]) - sigConst.reshape([-1,1])),axis=0)
        xhatBatch = xhatBatch.reshape([sample_size * repeat_size, -1])
        eq = np.equal(xhatBatch, np.reshape(xBatch, [sample_size * repeat_size,-1]))
        acc = eq.mean(axis=1)
        acc = np.mean(np.reshape(acc, (repeat_size,-1)), axis=0) 
        result[str(snr_)] = acc
        h = np.reshape(hBatch,[-1,sample_size, hBatch.shape[1], hBatch.shape[2]])
        result['cond'] = np.linalg.cond(h[0])

    return result                         

def symbol_detection(y, constellation):
    return np.expand_dims(np.argmin(np.abs(y-np.expand_dims(constellation, 0)), axis=1),1)

def zf_detector(y, H):
    return np.matmul(np.linalg.pinv(H), y)


def pic_detector(y, H, first_stage, constellation):
    #First detection stage
    x_1st = first_stage(y, H)
    x_1st_indices = symbol_detection(x_1st, constellation)
    x_1st = constellation[x_1st_indices]
    
    #PIC detection
    x_pic = np.zeros_like(x_1st)
    for k in range(0,x_pic.shape[0]):
        x_1st_k = np.copy(x_1st)
        x_1st_k[k, 0] = 0
        y_k = y - np.matmul(H, x_1st_k)
        H_k = np.linalg.pinv(np.expand_dims(H[:,k], 1))
        x_pic[k,0] = np.matmul(H_k, y_k)  
    return x_pic

def blast_eval(sample_size, repeat_size, sigConst, modulation, snr_range, NT, NR, correlated):
    print("Running BLAST evaluation")
    result = {}
    for snr_ in snr_range:
        xBatch, sBatch, yBatch, hBatch, hthBatch, average_H_powerdB = generate_data(modulation, sample_size, repeat_size, sigConst, snr_, NT, NR, correlated)
        complexnoise_sigma = 10. ** ((10.*np.log10(NT) + average_H_powerdB - snr_ - 10.*np.log10(NR))/20.) 
        shatBatch = np.zeros([sample_size * repeat_size, len(hBatch[0,0,:])])
        for i, h in enumerate(hBatch):
            shatBatch[i] = np.reshape(pic_detector(yBatch[i], hBatch[i], zf_detector, sigConst), (-1))

        xhatBatch = np.argmin(np.abs(np.reshape(shatBatch,[1,-1]) - sigConst.reshape([-1,1])),axis=0)
        xhatBatch = xhatBatch.reshape([sample_size * repeat_size, -1])
        eq = np.equal(xhatBatch, np.reshape(xBatch, [sample_size * repeat_size,-1]))
        acc = eq.mean(axis=1)
        acc = np.mean(np.reshape(acc, (repeat_size,-1)), axis=0) 
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
    if args.ML == True:
        result['ML'] = ml_eval(args.batch_size, args.numSamples, sigConst, modulation, snr_range, NT, NR, args.correlated)
        for snr_ in result['ML']:
            if not snr_=='cond':
                print((float(snr_), 1. - np.mean(result['ML'][snr_])))
    if args.AMP == True:
        result['AMP'] = amp_eval(args.batch_size, args.numSamples, sigConst, modulation, snr_range, NT, NR, args.correlated)
        for snr_ in result['AMP']:
            if not snr_=='cond':
                print((float(snr_), 1. - np.mean(result['AMP'][snr_])))
    if args.MMSE == True:
        result['MMSE'] = mmse_eval(args.batch_size, args.numSamples, sigConst, modulation, snr_range, NT, NR, args.correlated)
        for snr_ in result['MMSE']:
            if not snr_=='cond':
                print((float(snr_), 1. - np.mean(result['MMSE'][snr_])))
    if args.BLAST == True:
        result['BLAST'] = blast_eval(args.batch_size, args.numSamples, sigConst, modulation, snr_range, NT, NR, args.correlated)
        for snr_ in result['BLAST']:
            if not snr_=='cond':
                print((float(snr_), 1. - np.mean(result['BLAST'][snr_])))
    if args.SDR == True:
        result['SDR'] = sdr_eval(args.batch_size, args.numSamples, sigConst, modulation, snr_range, NT, NR, args.correlated)
        for snr_ in result['SDR']:
            if not snr_=='cond':
                print((float(snr_), 1. - np.mean(result['SDR'][snr_])))

    schemesList = []
    if args.ML:
        schemesList.append('ML')
    if args.MMSE:
        schemesList.append('MMSE')
    if args.AMP:
        schemesList.append('AMP')
    if args.BLAST:
        schemesList.append('BLAST')
    if args.SDR:
        schemesList.append('SDR')

    for sch in schemesList:
        path = "./baseline/%s/%s/%s/NT%d/NR%d/"%(sch, args.modulation, args.data_dir.split('/')[-1].rstrip('.npy'), NT, NR)
        path = getDir(path)
        if args.overwrite == True:
            with open(path+'results.pkl', 'wb') as f:
                pickle.dump(result[sch], f)
