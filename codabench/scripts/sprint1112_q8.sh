#!/usr/bin/env bash
# Sprint 11-12 Q8: robust covariance estimators (Huber / Tyler M-estimators) replacing OAS in X4f;
# dump streamed features, self-train at B3's config, compare as a replacement of B3's X4f part.
S=~/codabench/scripts/sprint1112_job.sh
bash $S q8_dump_hub 4 python sprint1106_dump.py --name X4fhub --solver riemann_sealed_proteus_x4f --set estimator=hub --protocols lopo later &
bash $S q8_dump_tyl 4 python sprint1106_dump.py --name X4ftyl --solver riemann_sealed_proteus_x4f --set estimator=tyl --protocols lopo later &
wait
bash $S q5_hub 3 python sprint1112_q5.py --model X4fhub --q 0.85 --lam 5 --start 20 --step 20 --zdtype f16 --C 0.03 &
bash $S q5_tyl 3 python sprint1112_q5.py --model X4ftyl --q 0.85 --lam 5 --start 20 --step 20 --zdtype f16 --C 0.03 &
wait
source ~/codabench/env040.sh >/dev/null 2>&1
cd ~/codabench/analysis
O=~/codabench/logs/sprint_1112; B3=q5_q0.85_l5.0_s20_t20_f16_C0.03
for m in X4fhub X4ftyl; do
  python sprint1112_combo.py --x4f q5_q0.85_l5.0_s20_t20_f16_${m}_C0.03 --x4f-base $B3 --arm $O/adabn/al_p32 --label "Q8 $m replaces X4f (self-trained)" 2>&1 | grep -v Warn
  python sprint1112_combo.py --x4f "$B3+q5_q0.85_l5.0_s20_t20_f16_${m}_C0.03" --x4f-base $B3 --arm $O/adabn/al_p32 --label "Q8 $m added to X4f" 2>&1 | grep -v Warn
done
