from subprocess import Popen

pList = []
snr = {'4':[4,9], '16':[11,16], '64':[18,23]}
for M in [4,16,64]:
    for s in range(snr[str(M)][0], snr[str(M)][1]+1):
        cmnd = 'python benchmark.py --x-size 16 --y-size 64 --snr-min %d --snr-max %d --batch-size 10000 --modulation QAM_%d --ML --overwrite --data --data-dir /data3/H_iid.npy'%(s, s, M)
        p = Popen(cmnd, shell=True)
        pList.append(p)

for p in pList:
    p.wait()
pList = []
#pList = []
#for nr in [16, 32, 64, 128, 256]:
#    for nt in [16, 32, 64, 128, 256]:
#        if nr >= nt:
#            cmnd = 'python benchmark.py --x-size %d --y-size %d --snr-min 15 --snr-max 16 --batch-size 100 --modulation QAM_16 --MMSE --overwrite'%(nt, nr)
#            p = Popen(cmnd, shell=True)
#            pList.append(p)
#
#    for p in pList:
#        p.wait()
#    pList = []
