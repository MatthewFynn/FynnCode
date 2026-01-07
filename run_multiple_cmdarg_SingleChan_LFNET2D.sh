#!/usr/bin/env bash
set -euo pipefail

split=3
output_dir=saved_results
#__________________
CH_STRING="123456"
v=1

# --- Fixed args that don't change across runs ---
BASE_ARGS="-r cross_fold_splits_ExclusionCriteria_${split}_5.csv \
-o ${output_dir}/savedxx${split} \
-s 31 \
-ind Ind_Table \
-tssl trained_SSL_model_pcg2_v2_zNorm_T1base \
-nf 5 \
-bs 64 \
-i Concatenated_heart_data_Ordered_SpRe \
-e 10 \
-m unetssl2d_cnn \
-dr 0 \
-oc adam \
-fs 2000 \
--notes trained_ssl_v2 \
-sav 0"

# -----------------------------
# Per-channel hyperparameters
# Fill these with your previously found optima per channel.
# Defaults below mirror your current fixed choices (-fl 5 -nfrag 111 -hs 256 -nl 2)
# -----------------------------
declare -A fl_map
declare -A nfrag_map
declare -A hs_map
declare -A nl_map
declare -A flow_map
declare -A fhigh_map
declare -A ep_map

fl_map[1]=6;   nfrag_map[1]=111; hs_map[1]=256; nl_map[1]=2; flow_map[1]=150; fhigh_map[1]=900; ep_map[1]=ep50;
fl_map[2]=5;   nfrag_map[2]=91; hs_map[2]=512;  nl_map[2]=3; flow_map[2]=50; fhigh_map[2]=900; ep_map[2]=ep10; 
fl_map[3]=5;   nfrag_map[3]=51; hs_map[3]=256;  nl_map[3]=2; flow_map[3]=10; fhigh_map[3]=900; ep_map[3]=ep10; 
fl_map[4]=5;   nfrag_map[4]=111; hs_map[4]=256; nl_map[4]=2; flow_map[4]=150; fhigh_map[4]=700; ep_map[4]=ep90; 
fl_map[5]=5;   nfrag_map[5]=91; hs_map[5]=512;  nl_map[5]=3; flow_map[5]=10; fhigh_map[5]=900; ep_map[5]=ep40; 
fl_map[6]=6;   nfrag_map[6]=111; hs_map[6]=512; nl_map[6]=2; flow_map[6]=50; fhigh_map[6]=900; ep_map[6]=ep20; 


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

# -----------------------------
# Run the sweep per channel, with that channel's own -fl/-nfrag/-hs/-nl
# -----------------------------
for ch in "${channels[@]}"; do

  fl="${fl_map[$ch]}"
  nfrag="${nfrag_map[$ch]}"
  hs="${hs_map[$ch]}"
  nl="${nl_map[$ch]}"
  ep="${ep_map[$ch]}"
  flow="${flow_map[$ch]}"
  fhigh="${fhigh_map[$ch]}"

  
  args="$BASE_ARGS -ch $ch -fl $fl -nfrag $nfrag -hs $hs -nl $nl -essl $ep -flow $flow -fhigh $fhigh -v $v"
  echo "Running v=$v (ch=$ch) with: $args"
  python3 run_model_trainer_SC_ONLY.py $args
  echo "Run v=$v complete"

done

