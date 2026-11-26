#!/usr/bin/env bash
# Thread-scaling benchmark: serial vs OpenMP (1,2,4,8 threads by default).
# Round-robin order: 1 warm-up round (rep 0, discarded in analysis) + REPS measured rounds.
# Settings via environment: EPOCHS (30) REPS (5) BATCH (512) THREADS ("1 2 4 8") DATA_DIR OUT BIND (default|close)
set -u
cd "$(dirname "$0")/.."
EPOCHS=${EPOCHS:-30}
REPS=${REPS:-5}
BATCH=${BATCH:-512}
THREADS=${THREADS:-"1 2 4 8"}
DATA_DIR=${DATA_DIR:-data/processed}
OUT=${OUT:-results/bench_threads.csv}
BIND=${BIND:-default}
ENVFILE=${ENVFILE:-results/bench_env.txt}
mkdir -p results build

echo "Building with the recorded flags..."
g++ -O3 -march=native -std=c++17 src/serial/mlp_serial.cpp -o build/mlp_serial || exit 1
g++ -O3 -march=native -fopenmp -std=c++17 src/openmp/mlp_omp.cpp -o build/mlp_omp || exit 1

if [ "$BIND" = "close" ]; then export OMP_PROC_BIND=close OMP_PLACES=threads; else unset OMP_PROC_BIND OMP_PLACES; fi

{
  echo "date: $(date -Is)"
  echo "kernel: $(uname -sr)"
  lscpu | grep -E "Model name|^CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|^Hypervisor" || true
  free -h | head -2
  echo "gcc: $(gcc --version | head -1)"
  echo "flags serial: -O3 -march=native -std=c++17"
  echo "flags openmp: -O3 -march=native -fopenmp -std=c++17"
  echo "git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "settings: EPOCHS=$EPOCHS REPS=$REPS BATCH=$BATCH THREADS='$THREADS' BIND=$BIND DATA_DIR=$DATA_DIR"
  echo "OMP_PROC_BIND=${OMP_PROC_BIND:-unset} OMP_PLACES=${OMP_PLACES:-unset}"
} > "$ENVFILE"

echo "backend,threads,bind,batch,epochs,rep,train_seconds,sec_per_epoch,train_loss,val_loss,val_acc,val_macro_f1" > "$OUT"

run_one() {  # backend threads rep
  local backend=$1 th=$2 rep=$3 out
  if [ "$backend" = serial ]; then
    out=$(./build/mlp_serial --data-dir "$DATA_DIR" --epochs "$EPOCHS" --batch "$BATCH" --log /dev/null)
  else
    out=$(./build/mlp_omp --threads "$th" --data-dir "$DATA_DIR" --epochs "$EPOCHS" --batch "$BATCH" --log /dev/null)
  fi
  local total per last trl vl va vf
  total=$(echo "$out" | awk '/^TRAINING TIME/{print $7}')
  per=$(echo "$out" | awk '/^TRAINING TIME/{print $10}')
  last=$(echo "$out" | grep '^epoch' | tail -1)
  trl=$(echo "$last" | sed -E 's/.*train loss ([0-9.]+).*/\1/')
  vl=$(echo "$last" | sed -E 's/.*val loss ([0-9.]+).*/\1/')
  va=$(echo "$last" | sed -E 's/.*val acc ([0-9.]+).*/\1/')
  vf=$(echo "$last" | sed -E 's/.*val macro-F1 ([0-9.]+).*/\1/')
  echo "$backend,$th,$BIND,$BATCH,$EPOCHS,$rep,$total,$per,$trl,$vl,$va,$vf" >> "$OUT"
  printf "round %d/%d  %-6s threads=%-2s  %8s s  (%s s/epoch)  val macro-F1 %s\n" "$rep" "$REPS" "$backend" "$th" "$total" "$per" "$vf"
}

for rep in $(seq 0 "$REPS"); do
  [ "$rep" -eq 0 ] && echo "--- warm-up round (not counted) ---"
  run_one serial 1 "$rep"
  for t in $THREADS; do run_one omp "$t" "$rep"; done
done
echo "Done. Results in $OUT, environment in $ENVFILE"
