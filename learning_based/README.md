##Online Training
For online training algorithm run:
```
python onlineTraining.py  --x-size 32 --y-size 64 --snr-min 10 --snr-max 15 --layers 10 -lr 1e-3 --batch-size 500 --train-iterations 1000 --mod QAM_4 --gpu 1  --test-batch-size 5000 --linear fixed_W  --denoiser MMNet --data --channels-dir /data3/Jakobs_channels_normalized_32_small.npy --output-dir .
```
