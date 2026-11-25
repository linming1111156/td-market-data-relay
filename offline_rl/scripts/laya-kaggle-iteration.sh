#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="${LAYA_KAGGLE_BUILD_DIR:-/tmp/laya-kaggle-kernels}"

RELAY_KERNEL="minglinl99/laya-kaggle-relay-increment-sync-main"
CPU_KERNEL="minglinl99/laya-cpu-data-preparation"
GPU_KERNEL="minglinl99/laya-gpu-finetune"
FEEDBACK_KERNEL="minglinl99/laya-model-feedback-gpu-recursive-round-v57"
EXPERIENCE_DATASET="minglinl99/laya-market-paper-trade-experience"
RELAY_DATASET="minglinl99/td-market-vps-captures-increment-20261120"
SOURCE_DATASET_ROUND="v105-relay-v49-15m-edge0-utcday-v1"
BRANCH_PLAN_HOURS="3"

kaggle() {
  uvx --from kaggle --with six kaggle "$@"
}

usage() {
  cat <<'EOF'
Usage:
  laya-kaggle-iteration.sh status
  laya-kaggle-iteration.sh logs [GPU|RELAY|CPU|FEEDBACK]
  laya-kaggle-iteration.sh relay
  laya-kaggle-iteration.sh prepare
  laya-kaggle-iteration.sh train PREVIOUS_FEEDBACK_ROUND RESERVED_GPU_HOURS --web-confirmed
  laya-kaggle-iteration.sh train PREVIOUS_FEEDBACK_ROUND RESERVED_GPU_HOURS --web-confirmed --recurrent-depth-ablation
  laya-kaggle-iteration.sh feedback NEW_FEEDBACK_ROUND --evaluation-confirmed
  laya-kaggle-iteration.sh cycle-help

Commands only update the existing Kaggle projects. They never create a new
round-specific notebook, download Kaggle datasets, or fall back to CPU training.
EOF
}

kernel_status() {
  local kernel="$1"
  kaggle kernels status "$kernel"
}

assert_not_active() {
  local kernel="$1" result
  result="$(kernel_status "$kernel")"
  printf '%s\n' "$result"
  case "$result" in
    *QUEUED*|*RUNNING*|*SUBMITTED*)
      printf '\n%s is still active. Do not push another version.\n' "$kernel" >&2
      printf 'If Kaggle says QUEUED, inspect this notebook in EgoLite: Draft Session and Active Events.\n' >&2
      return 2
      ;;
  esac
}

assert_kernel_complete() {
  local kernel="$1" result
  result="$(kernel_status "$kernel")"
  printf '%s\n' "$result"
  case "$result" in
    *COMPLETE*) return 0 ;;
    *)
      printf '\n%s must complete successfully before continuing.\n' "$kernel" >&2
      if [[ "$result" == *QUEUED* ]]; then
        printf 'Inspect Draft Session and Active Events in EgoLite; QUEUED alone does not mean failure.\n' >&2
      fi
      return 2
      ;;
  esac
}

read_gpu_used_hours() {
  local quota_output used
  quota_output="$(kaggle quota)"
  printf '%s\n' "$quota_output" >&2
  used="$(awk '$1 == "GPU" {gsub(/h/, "", $2); print $2}' <<<"$quota_output")"
  if [[ -z "$used" ]]; then
    printf 'Could not read GPU usage from official Kaggle quota output.\n' >&2
    return 2
  fi
  printf '%s' "$used"
}

check_quota_plan() {
  local used="$1" reserved="$2" branches="$3" plan="$BRANCH_PLAN_HOURS"
  python3 - "$used" "$reserved" "$plan" "$branches" <<'PY'
import sys
used, reserved, per_branch = map(float, sys.argv[1:4])
branches = int(sys.argv[4])
planned = branches * per_branch
total = used + reserved + planned
print(f"GPU plan: used {used:.2f}h + other reservations {reserved:.2f}h + planned {planned:.2f}h ({branches} branches) = {total:.2f}h / 30h")
if total > 30:
    raise SystemExit("Not enough official GPU quota for this fine-tuning plan; shorten the plan or wait for quota refresh.")
PY
}

show_status() {
  printf '%s\n' '=== Official Kaggle quota ==='
  kaggle quota
  printf '\n%s\n' '=== Existing project statuses ==='
  for kernel in "$RELAY_KERNEL" "$CPU_KERNEL" "$GPU_KERNEL" "$FEEDBACK_KERNEL"; do
    printf '\n--- %s ---\n' "$kernel"
    kernel_status "$kernel" || true
  done
  printf '\n%s\n' '=== Cumulative dataset files ==='
  kaggle datasets files "$EXPERIENCE_DATASET" --page-size 20 || true
  kaggle datasets files "$RELAY_DATASET" --page-size 20 || true
  printf '\nDataset version numbers and notes are in each dataset page > Versions.\n'
}

show_logs() {
  local key="${1:-GPU}" kernel
  case "$key" in
    GPU) kernel="$GPU_KERNEL" ;;
    RELAY) kernel="$RELAY_KERNEL" ;;
    CPU) kernel="$CPU_KERNEL" ;;
    FEEDBACK) kernel="$FEEDBACK_KERNEL" ;;
    *) printf 'Choose GPU, RELAY, CPU, or FEEDBACK.\n' >&2; return 2 ;;
  esac
  kernel_status "$kernel" || true
  printf '\n=== Kaggle logs: %s ===\n' "$kernel"
  kaggle kernels logs "$kernel"
}

push_relay() {
  assert_not_active "$RELAY_KERNEL"
  mkdir -p "$OUTPUT_DIR"
  python3 "$REPO_ROOT/offline_rl/scripts/build_laya_kaggle_relay_increment.py" \
    --output-dir "$OUTPUT_DIR" --project-version main
  kaggle kernels push -p "$OUTPUT_DIR/laya-kaggle-relay-increment-sync-main"
}

push_cpu_prepare() {
  assert_not_active "$CPU_KERNEL"
  assert_kernel_complete "$RELAY_KERNEL"
  mkdir -p "$OUTPUT_DIR"
  python3 "$REPO_ROOT/offline_rl/scripts/build_laya_kaggle_cpu_data_preparation.py" \
    --output-dir "$OUTPUT_DIR"
  kaggle kernels push -p "$OUTPUT_DIR/laya-cpu-data-preparation"
}

push_gpu_train() {
  local previous_round="${1:-}" reserved="${2:-}" web_confirmed="${3:-}" ablation="${4:-}"
  if [[ -z "$previous_round" || -z "$reserved" || "$web_confirmed" != "--web-confirmed" || ( -n "$ablation" && "$ablation" != "--recurrent-depth-ablation" ) ]]; then
    printf 'Usage: train PREVIOUS_FEEDBACK_ROUND RESERVED_GPU_HOURS --web-confirmed [--recurrent-depth-ablation]\n' >&2
    printf 'Check the existing notebook session and Active Events in EgoLite first.\n' >&2
    return 2
  fi
  if [[ ! "$previous_round" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    printf 'Invalid feedback round identifier: %s\n' "$previous_round" >&2
    return 2
  fi
  if [[ ! "$reserved" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    printf 'RESERVED_GPU_HOURS must be a number confirmed from other active GPU events.\n' >&2
    return 2
  fi

  assert_not_active "$GPU_KERNEL"
  assert_kernel_complete "$RELAY_KERNEL"
  assert_kernel_complete "$CPU_KERNEL"
  local used
  used="$(read_gpu_used_hours)"
  local branch_count=2
  [[ "$ablation" == "--recurrent-depth-ablation" ]] && branch_count=3
  check_quota_plan "$used" "$reserved" "$branch_count"

  mkdir -p "$OUTPUT_DIR"
  local -a build_args=(
    --output-dir "$OUTPUT_DIR" \
    --project-slug laya-gpu-finetune \
    --dataset-ref "$EXPERIENCE_DATASET" \
    --expected-feedback-round "$previous_round" \
    --capture-increment-ref "$RELAY_DATASET" \
    --quota-used-hours "$used" \
    --reserved-gpu-hours "$reserved" \
    --plan-hours "$BRANCH_PLAN_HOURS"
  )
  [[ "$ablation" == "--recurrent-depth-ablation" ]] && build_args+=(--recurrent-depth-ablation)
  python3 "$REPO_ROOT/offline_rl/scripts/build_laya_kaggle_recursive_iteration.py" "${build_args[@]}"
  kaggle kernels push -p "$OUTPUT_DIR/laya-gpu-finetune"
  printf '\nSubmitted the existing GPU project. Run `status`/`logs GPU` to monitor it.\n'
  printf 'If status is QUEUED, inspect Draft Session and Active Events in EgoLite before deciding what to do.\n'
}

push_feedback() {
  local round="${1:-}" confirmed="${2:-}"
  if [[ -z "$round" || "$confirmed" != "--evaluation-confirmed" ]]; then
    printf 'Usage: feedback NEW_FEEDBACK_ROUND --evaluation-confirmed\n' >&2
    printf 'First confirm the completed GPU logs show CUDA, both branches, and frozen evaluation.\n' >&2
    return 2
  fi
  if [[ ! "$round" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    printf 'Invalid feedback round identifier: %s\n' "$round" >&2
    return 2
  fi
  assert_not_active "$FEEDBACK_KERNEL"
  assert_kernel_complete "$GPU_KERNEL"
  mkdir -p "$OUTPUT_DIR"
  python3 "$REPO_ROOT/offline_rl/scripts/build_laya_kaggle_model_feedback.py" \
    --output-dir "$OUTPUT_DIR" \
    --dataset-ref "$EXPERIENCE_DATASET" \
    --source-kernel "$GPU_KERNEL" \
    --source-dataset-round "$SOURCE_DATASET_ROUND" \
    --feedback-round "$round"
  kaggle kernels push -p "$OUTPUT_DIR/laya-model-feedback-gpu-recursive-round-v57"
  printf '\nSubmitted the existing feedback project. Confirm publication in logs and dataset Versions before the next train command.\n'
}

case "${1:-}" in
  status) show_status ;;
  logs) show_logs "${2:-GPU}" ;;
  relay) push_relay ;;
  prepare) push_cpu_prepare ;;
  train) push_gpu_train "${2:-}" "${3:-}" "${4:-}" "${5:-}" ;;
  feedback) push_feedback "${2:-}" "${3:-}" ;;
  cycle-help)
    cat <<'EOF'
One recursive cycle:
  1. status; check EgoLite Active Events and the existing GPU session.
  2. relay; wait until the existing Relay kernel is COMPLETE; inspect logs for new captures.
  3. prepare; wait until the existing CPU preparation kernel is COMPLETE; inspect row counts (this is not model training).
  4. train PREVIOUS_FEEDBACK_ROUND RESERVED_GPU_HOURS --web-confirmed
  5. If GPU is QUEUED, inspect the web session and Active Events before any decision; never push a duplicate.
  6. Wait for GPU COMPLETE; inspect logs for actual CUDA, both branches, frozen metrics, and GPU hours.
  7. feedback NEW_FEEDBACK_ROUND --evaluation-confirmed; wait for publication and check the new cumulative dataset version/count.
  8. Start the next cycle only when a fresh held-out window is available and its labels are mature.

For unattended periodic observation only, run `while true; do ... status; sleep 1800; done`.
Do not cron the train/feedback commands: those need fresh-data, maturity, quota, and web-session checks.
EOF
    ;;
  -h|--help|"") usage ;;
  *) usage >&2; exit 2 ;;
esac
