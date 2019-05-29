# Learning-based schemes
## Online training
For online training algorithm run:
```
python onlineTraining.py  --x-size 32 --y-size 64 --snr-min 10 --snr-max 15 --layers 10 -lr 1e-3 --batch-size 500 --train-iterations 1000 --mod QAM_4 --gpu 1  --test-batch-size 5000 --linear MMNet  --denoiser MMNet --data --channels-dir path/to/channels --output-dir path/to/save/results
```
