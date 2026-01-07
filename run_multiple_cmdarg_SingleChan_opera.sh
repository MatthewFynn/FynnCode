#!/usr/bin/env bash
set -euo pipefail

split=3
output_dir=saved_results
v=1
#__________________
CH_STRING="123456"

# --- Fixed args that don't change across runs ---
BASE_ARGS="-r cross_fold_splits_ExclusionCriteria_${split}_5.csv \
-o ${output_dir}/savedxx${split} \
-s 31 \
-ind Ind_Table \
-nf 5 \
-bs 64 \
-i Concatenated_heart_data_Ordered_SpRe \
-e 10 \
-m opera_ce \
-dr 0 \
-oc adam \
-nl 0 \
-sav 0 \
--notes trained_ssl " 


declare -A fl_map
declare -A nfrag_map
declare -A hs_map
declare -A flow_map
declare -A fhigh_map
declare -A fs_map

nfrag_map[1]=111; hs_map[1]=256;  fl_map[1]=7; flow_map[1]=10; fhigh_map[1]=900; fs_map[1]=16000 
nfrag_map[2]=61; hs_map[2]=1024;  fl_map[2]=5; flow_map[2]=10; fhigh_map[2]=900; fs_map[2]=16000 
nfrag_map[3]=91; hs_map[3]=256;   fl_map[3]=5; flow_map[3]=10; fhigh_map[3]=900; fs_map[3]=16000 
nfrag_map[4]=111; hs_map[4]=512;  fl_map[4]=5; flow_map[4]=10; fhigh_map[4]=800; fs_map[4]=16000 
nfrag_map[5]=61; hs_map[5]=512;   fl_map[5]=5; flow_map[5]=10; fhigh_map[5]=900; fs_map[5]=8000 
nfrag_map[6]=91; hs_map[6]=512;   fl_map[6]=5; flow_map[6]=10; fhigh_map[6]=900; fs_map[6]=16000 

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
  fs="${fs_map[$ch]}"
  flow="${flow_map[$ch]}"
  fhigh="${fhigh_map[$ch]}"


args="$BASE_ARGS -ch $ch -fl $fl -nfrag $nfrag -hs $hs -flow $flow -fhigh $fhigh -fs $fs -v $v"
echo "Running v=$v (ch=$ch) with: $args"
python3 run_model_trainer_SC_ONLY.py $args
echo "Run v=$v complete"

done
