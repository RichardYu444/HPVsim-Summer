#!/bin/bash
# Batches of 5 seeds at once, one network at a time, for run_equilibrium_1900.py
cd "/c/Users/richa/OneDrive - Nexus365/Documents/HPV sim Project/Summer"
LOG=csvs/equilibrium_1900/logs; mkdir -p $LOG
export PYTHONIOENCODING=utf-8 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for net in default gamma_5 gamma_2 gamma_1 gamma_0.25 gamma_0.05 powerlaw_3; do
  echo "BATCH START $net $(date +%H:%M)"
  for seed in 0 1 2 3 4; do
    /c/Users/richa/miniconda3/envs/summerhpvsim/python.exe run_equilibrium_1900.py $net $seed > $LOG/${net}_seed${seed}.log 2>&1 &
  done
  wait
  echo "BATCH END $net $(date +%H:%M): $(grep -h 'DONE\|Error' $LOG/${net}_seed*.log | tr '\n' ' ')"
done
echo "ALL DONE $(date +%H:%M)"
