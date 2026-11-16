#!/bin/bash
# Generate all multi-vehicle sets (CARLA server must be running). Log: logs/multivehicle_gen.log
cd "$(dirname "$0")/.."
PY=../venv-carla/Scripts/python.exe
for spec in "vehicle.dodge.charger sedan_charger" "vehicle.nissan.patrol suv_patrol" "vehicle.ambulance.ford truck_ambulance"; do
  set -- $spec
  for c in red=255,0,0 blue=0,0,255 black=0,0,0; do $PY scripts/gen_multivehicle.py --vehicle $1 --name $2 --color $c --set test || echo "FAILED $2 $c test"; done
  for c in white=255,255,255 gray=187,187,187 black=0,0,0; do $PY scripts/gen_multivehicle.py --vehicle $1 --name $2 --color $c --set calib || echo "FAILED $2 $c calib"; done
done
$PY scripts/gen_multivehicle.py --vehicle vehicle.carlacola.actors --name truck_carlacola --color native=149,0,14 --set test || echo "FAILED carlacola"
echo ALL_DONE
