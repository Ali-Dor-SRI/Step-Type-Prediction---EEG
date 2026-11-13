source ~/codabench/env040.sh
cd ~/codabench/analysis
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUTF8=1
python sprint1114_p04.py > ~/codabench/logs/sprint_1114/a3_p04.out 2>~/codabench/logs/sprint_1114/a3_p04.err
