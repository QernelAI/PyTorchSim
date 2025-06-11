#!/bin/bash

ignore_rest=false
gem5_cmd=""
result_path=""
gem5_time=""
togsim_time=""

total_gem5=0
total_togsim=0

while IFS= read -r line; do
  if [[ "$line" == "Wrapper Codegen Path ="* ]]; then
    ignore_rest=true
  fi

  if ! $ignore_rest; then
    continue
  fi

  if [[ "$line" == "[Gem5Simulator] cmd>"* ]]; then
    gem5_cmd=$(echo "$line" | sed 's/^\[Gem5Simulator\] cmd>  *//')
    dir=$(echo "$line" | sed -n 's/.*-d \([^ ]*\).*/\1/p')/sto.log
    echo $dir
    gem5_time=$(grep "Simulation time:" "$dir" | \
                sed -E 's/^Simulation time: ([0-9.]+) seconds$/\1/')
    echo "GEM5: $gem5_time" 
    total_gem5=$(echo "$total_gem5 + $gem5_time" | bc)
  fi

  if [[ "$line" == *"[BackendSimulator] Simulation of"* ]]; then
    result_path=$(echo "$line" | awk -F' ' '{gsub(/"/, "", $8); print $8}')
    togsim_time=$(grep "\[info\] Simulation time:" "$result_path" | \
                  sed -E 's/^\[[^]]+\] \[info\] Simulation time: ([0-9.]+) seconds$/\1/')
    echo "TOGSim: $togsim_time"
    total_togsim=$(echo "$total_togsim + $togsim_time" | bc)
  fi
done

if [[ -n "$total_gem5" && -n "$total_time" ]]; then
  total_time=$(python3 -c "print(round($total_gem5 + $total_togsim, 6))")
  echo "Simulation time: $total_time seconds"
fi