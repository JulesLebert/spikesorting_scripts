#!/bin/bash

current_script=$(basename "$0")

for file in *; do
  if [ "$file" != "$current_script" ] && [ -f "$file" ]; then
    qsub "$file"
    sleep 1
  fi
done