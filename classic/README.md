# Classic schemes
We provide a baseline for comparison among below classic detection schemes:
- Minimum mean square error (MMSE)
- Approximated message passaing (AMP)
- Semidefinite relaxation (SDR)
- Multistage interference cancelation (BLAST)
- Maximum-likelihood optimal (ML)

We can run all above schemes using:
```
python benchmark.py --x-size 16 --y-size 64 --snr-min 2 --snr-max 7 --batch-size 100  --numSamples 100 --modulation QAM_4 --overwrite --data --channels-dir path/to/channels --MMSE --AMP --SDR --BLAST --ML

```
