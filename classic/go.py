from subprocess import Popen

snrRange = {'4':[2,7], '16':[9,14], '64':[16,21]}
for M in ['4','16','64']:
    for snr in range(snrRange[M][0],snrRange[M][1]+1):
        pList = []
        for i in range(100):
            cmnd = 'python fastBenchmark.py --x-size 16 --y-size 64 --snr-min %d --snr-max %d --batch-size 1 --modulation QAM_%s --ML --overwrite --data --data-dir /data3/H_iid.npy --goTo %d --numSamples 1000'%(snr, snr, M, i)
            p = Popen(cmnd, shell=True)
            pList.append(p)
    
    
        for p in pList:
            p.wait()
