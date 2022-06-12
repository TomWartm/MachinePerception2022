#rm -r logs
rm sample_test

bsub -n 6 -W 2:00 -o sample_test -R "rusage[mem=2048, ngpus_excl_p=1]" python scripts/main.py --cfg configs/baseline.yaml

