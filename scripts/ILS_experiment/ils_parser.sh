#!/bin/bash

ignore_rest=false
gem5_cmd=""
result_path=""
gem5_time=""
togsim_time=""

total_gem5=0
total_togsim=0

while IFS= read -r line; do
  if [[ "$line" == launch* ]]; then
    tile_path=$(echo "$line" | awk '{print $2}')
    base_dir=$(dirname "$tile_path")
    result_path="$base_dir/m5out/sto.log"
    echo $result_path
    togsim_time=$(grep "Simulation time:" "$result_path" | \
                  sed -E 's/Simulation time: ([0-9.]+) seconds$/\1/')
    echo "GEM5: $togsim_time"
    total_togsim=$(echo "$total_togsim + $togsim_time" | bc)
  fi

  if [[ "$line" == *"[info] Simulation time:"* ]]; then
    togsim_time=$(echo $line | sed -E 's/^\[[^]]+\] \[info\] Simulation time: ([0-9.]+) seconds$/\1/')
    echo "TOGSim: $togsim_time"
  fi
done

if [[ -n "$total_gem5" && -n "$total_togsim" ]]; then
  #total_time=$(python3 -c "print(round($total_gem5 + $total_togsim, 6))")
  echo "Simulation time: $togsim_time seconds"
fi