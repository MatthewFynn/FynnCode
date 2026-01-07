#!/usr/bin/env bash
set -euo pipefail

split=3
output_dir=saved_results_IIT2
v=1
#__________________
CH_STRING="123456"
CH_STRING="34"

# --- Fixed args that don't change across runs ---
BASE_ARGS="-r cross_fold_splits_ExclusionCriteria_${split}_5.csv \
-o ${output_dir}/savedxx${split} \
-s 31 \
-ind Ind_Table \
-nf 5 \
-bs 64 \
-i Concatenated_heart_data_Ordered_SpRe \
-e 10 \
-tssl trained_SSL_model_pcg2_1D_a10b2m2 \
-m unetssl_cnn \
-fs 2000
-dr 0 \
-oc adam \
-sav 0 \
--notes trained_ssl "

declare -A hs_map
declare -A nl_map
declare -A fl_map
declare -A nfrag_map
declare -A flow_map
declare -A fhigh_map
declare -A ep_map

hs_map[1]=256; nl_map[1]=2; fl_map[1]=4; nfrag_map[1]=31; flow_map[1]=100; fhigh_map[1]=900; ep_map[1]=ep30 
hs_map[2]=512; nl_map[2]=2; fl_map[2]=4; nfrag_map[2]=31; flow_map[2]=10; fhigh_map[2]=700; ep_map[2]=ep60 
hs_map[3]=512; nl_map[3]=4; fl_map[3]=4; nfrag_map[3]=31; flow_map[3]=100; fhigh_map[3]=800; ep_map[3]=ep60 
hs_map[4]=256; nl_map[4]=2; fl_map[4]=4; nfrag_map[4]=31; flow_map[4]=10; fhigh_map[4]=900; ep_map[4]=ep120 
hs_map[5]=256; nl_map[5]=2; fl_map[5]=2; nfrag_map[5]=31; flow_map[5]=10; fhigh_map[5]=700; ep_map[5]=ep120 
hs_map[6]=256; nl_map[6]=3; fl_map[6]=4; nfrag_map[6]=31; flow_map[6]=10; fhigh_map[6]=600; ep_map[6]=ep90 


# -----------------------------
# Build channel array from CH_STRING (e.g., "123456" -> 1 2 3 4 5 6)
# -----------------------------
channels=()
for (( i=0; i<${#CH_STRING}; i++ )); do
  ch_char="${CH_STRING:$i:1}"
  # ignore non-digits, just in case
  if [[ "$ch_char" =~ [0-9] ]]; then
    channels+=("$ch_char")
  fi
done

if [ ${#channels[@]} -eq 0 ]; then
  echo "No valid channels parsed from CH_STRING='$CH_STRING'." >&2
  exit 1
fi

for ch in "${channels[@]}"; do

  hs="${hs_map[$ch]}"
  nl="${nl_map[$ch]}"
  fl="${fl_map[$ch]}"
  nfrag="${nfrag_map[$ch]}"
  flow="${flow_map[$ch]}"
  fhigh="${fhigh_map[$ch]}"
  ep="${ep_map[$ch]}"

  # echo "=== Channel $ch: using -fl $fl -nfrag $nfrag -hs $hs -nl $nl -fs $fs ==="

  args="$BASE_ARGS -ch $ch -fl $fl -nfrag $nfrag -hs $hs -nl $nl -flow $flow -fhigh $fhigh -essl $ep -v $v "
  echo "Running v=$v (ch=$ch) with: $args"
  python3 run_model_trainer_SC_ONLY.py $args

done
