#!/usr/bin/env bash
set -euo pipefail

split=3
output_dir=saved_results

CH_STRING="123456"

v=1
# --- Fixed args that don't change across runs ---
BASE_ARGS="-r cross_fold_splits_ExclusionCriteria_${split}_5.csv \
-o ${output_dir}/savedxx${split} \
-s 31 \
-ind Ind_Table \
-nf 5 \
-bs 64 \
-i Concatenated_heart_data_Ordered_SpRe \
-e 10 \
-m encodec_cnn \
-dr 0 \
-oc adamw \
-sav 0
--notes trained_ssl "  

declare -A fl_map
declare -A nfrag_map
declare -A hs_map
declare -A nl_map
declare -A fs_map

nfrag_map[1]=51; hs_map[1]=512;  fs_map[1]=4125; fl_map[1]=6; nl_map[1]=-2; flow_map[1]=100; fhigh_map[1]=700     
nfrag_map[2]=51; hs_map[2]=512;  fs_map[2]=4125; fl_map[2]=6; nl_map[2]=-2; flow_map[2]=10; fhigh_map[2]=600     
nfrag_map[3]=51; hs_map[3]=512;  fs_map[3]=4125; fl_map[3]=4; nl_map[3]=-2; flow_map[3]=100; fhigh_map[3]=800     
nfrag_map[4]=51; hs_map[4]=1024; fs_map[4]=4125; fl_map[4]=6; nl_map[4]=-2; flow_map[4]=100; fhigh_map[4]=800     
nfrag_map[5]=71; hs_map[5]=512;  fs_map[5]=4125; fl_map[5]=6; nl_map[5]=-2; flow_map[5]=100; fhigh_map[5]=700     
nfrag_map[6]=51; hs_map[6]=1024; fs_map[6]=4125; fl_map[6]=6; nl_map[6]=-2; flow_map[6]=10; fhigh_map[6]=600     


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
  fs="${fs_map[$ch]}"
  flow="${flow_map[$ch]}"
  fhigh="${fhigh_map[$ch]}"

  args="$BASE_ARGS -ch $ch -fl $fl -nfrag $nfrag -hs $hs -nl $nl -fs $fs -flow $flow -fhigh $fhigh -v $v"
  echo "Running v=$v (ch=$ch) with: $args"
  python3 run_model_trainer_SC_ONLY.py $args
  echo "Run v=$v complete"

done

