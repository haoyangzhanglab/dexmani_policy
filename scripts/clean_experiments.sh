#!/usr/bin/env bash
# 清理 experiments 目录下未正常完成训练的实验及玩具实验。
#
# 三类待清理对象：
#   A. 不完整实验 — 有 checkpoint 但 epoch 没跑到 num_epochs-1，或完全无 checkpoint
#   B. 玩具实验   — num_epochs < 200（测试跑、冒烟跑等），无论是否跑完
#
# 删除策略：
#   --force 时，A 类直接删除；B 类逐项询问 y/n
#
# 用法：
#   bash scripts/clean_experiments.sh              # dry-run
#   bash scripts/clean_experiments.sh --force      # A 类删除 + B 类逐项确认
#   bash scripts/clean_experiments.sh --force --yes # 全部直接删除（跳过 B 类询问）
#   bash scripts/clean_experiments.sh --force --older-than 7
#   bash scripts/clean_experiments.sh --force --include-active
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
EXP_DIR="$ROOT_DIR/experiments"

FORCE=false
YES_ALL=false
OLDER_THAN=0
INCLUDE_ACTIVE=false
SKIP_ACTIVE_MINUTES=60
TOY_MIN_EPOCHS=200

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)            FORCE=true ;;
        --yes|-y)           YES_ALL=true ;;
        --older-than)       OLDER_THAN="$2"; shift ;;
        --include-active)   INCLUDE_ACTIVE=true ;;
        --skip-active-min)  SKIP_ACTIVE_MINUTES="$2"; shift ;;
        --toy-min-epochs)   TOY_MIN_EPOCHS="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

if [[ ! -d "$EXP_DIR" ]]; then
    echo "experiments/ directory not found"
    exit 0
fi

# ── 辅助函数 ──

# 判断实验是否为活跃（最近修改过）
is_active() {
    local exp_dir="$1"
    $INCLUDE_ACTIVE && return 1
    local now newest_ts age_minutes
    now=$(date +%s)
    newest_ts=$(find "$exp_dir" -type f -printf '%T@\n' 2>/dev/null | sort -rn | head -1 | cut -d. -f1)
    [[ -z "$newest_ts" ]] && return 1
    age_minutes=$(( (now - newest_ts) / 60 ))
    [[ $age_minutes -lt $SKIP_ACTIVE_MINUTES ]]
}

# 从 config.yaml 读出 num_epochs；读不到返回空
get_num_epochs() {
    python -c "
import sys
try:
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(sys.argv[1])
    print(cfg.training.loop.num_epochs)
except Exception:
    pass
" "$1/config.yaml" 2>/dev/null || true
}

# 从 checkpoint 文件名提取最大 epoch；无 checkpoint 返回 -1
get_max_epoch() {
    local ckpt_dir="$1" max=-1 epoch pt_file
    for pt_file in "$ckpt_dir"/epoch=*.pt; do
        [[ -f "$pt_file" ]] || continue
        epoch=$(basename "$pt_file" | grep -oP 'epoch=\K\d+')
        [[ -n "$epoch" ]] && { epoch=$((10#$epoch)); [[ $epoch -gt $max ]] && max=$epoch; }
    done
    echo "$max"
}

# ── 扫描实验目录 ──
declare -a INCOMPLETE=()         # A 类：未完成
declare -A INCOMPLETE_REASON=()
declare -a TOY=()                # B 类：玩具实验
declare -A TOY_INFO=()
declare -a SKIPPED=()            # 被保护的活跃实验
TOTAL_INCOMPLETE_SIZE=0
TOTAL_TOY_SIZE=0

while IFS= read -r -d '' checkpoints_dir; do
    exp_dir="$(dirname "$checkpoints_dir")"
    [[ -f "$exp_dir/config.yaml" ]] || continue

    # 时间过滤
    if [[ "$OLDER_THAN" -gt 0 ]]; then
        dir_age_days=$(( ($(date +%s) - $(stat -c %Y "$exp_dir")) / 86400 ))
        [[ $dir_age_days -lt $OLDER_THAN ]] && continue
    fi

    # 活跃保护
    if is_active "$exp_dir"; then
        age_min=$(( ($(date +%s) - $(find "$exp_dir" -type f -printf '%T@\n' 2>/dev/null | sort -rn | head -1 | cut -d. -f1)) / 60 ))
        SKIPPED+=("$exp_dir|SKIP: ${age_min}min 前修改过（用 --include-active 强制纳入）")
        continue
    fi

    num_epochs=$(get_num_epochs "$exp_dir")
    max_epoch=$(get_max_epoch "$checkpoints_dir")
    size=$(du -sb "$exp_dir" 2>/dev/null | cut -f1)

    # --- 分类 ---
    if [[ -n "$num_epochs" ]] && [[ $num_epochs -lt $TOY_MIN_EPOCHS ]]; then
        # B 类：玩具实验
        if [[ $max_epoch -ge 0 ]]; then
            info="num_epochs=$num_epochs, 完成 $((max_epoch + 1))/$num_epochs epochs（最后 checkpoint: epoch=$max_epoch）"
        else
            info="num_epochs=$num_epochs, 无 checkpoint"
        fi
        TOY+=("$exp_dir")
        TOY_INFO["$exp_dir"]="$info"
        TOTAL_TOY_SIZE=$((TOTAL_TOY_SIZE + size))
    elif [[ $max_epoch -lt 0 ]]; then
        # A 类：无 checkpoint
        INCOMPLETE+=("$exp_dir")
        INCOMPLETE_REASON["$exp_dir"]="无 checkpoint（训练崩溃于首个检查点之前）"
        TOTAL_INCOMPLETE_SIZE=$((TOTAL_INCOMPLETE_SIZE + size))
    elif [[ -n "$num_epochs" ]] && [[ $max_epoch -lt $((num_epochs - 1)) ]]; then
        # A 类：未跑完
        INCOMPLETE+=("$exp_dir")
        INCOMPLETE_REASON["$exp_dir"]="仅完成 $((max_epoch + 1))/$num_epochs epochs（最后 checkpoint: epoch=$max_epoch）"
        TOTAL_INCOMPLETE_SIZE=$((TOTAL_INCOMPLETE_SIZE + size))
    fi
    # else: 正常完成且 num_epochs >= TOY_MIN_EPOCHS → 跳过
done < <(find "$EXP_DIR" -type d -name checkpoints -print0)

# ── 输出 ──
if [[ ${#INCOMPLETE[@]} -eq 0 ]] && [[ ${#TOY[@]} -eq 0 ]]; then
    echo "没有发现不完整或玩具实验。"
    [[ ${#SKIPPED[@]} -gt 0 ]] && echo "（${#SKIPPED[@]} 个实验因活跃被保护跳过）"
    exit 0
fi

action_mode="[DRY RUN]"
$FORCE && action_mode="[FORCE]"

# ── A 类：不完整实验 ──
if [[ ${#INCOMPLETE[@]} -gt 0 ]]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    printf "  %s A 类：不完整实验（%d 个）\n" "$action_mode" "${#INCOMPLETE[@]}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    for exp_dir in "${INCOMPLETE[@]}"; do
        size_hr=$(du -sh "$exp_dir" 2>/dev/null | cut -f1)
        rel_path="${exp_dir#$ROOT_DIR/}"
        echo "  $rel_path  ($size_hr)"
        echo "    ↳ ${INCOMPLETE_REASON[$exp_dir]}"
        if $FORCE; then
            rm -rf "$exp_dir"
            parent="$(dirname "$exp_dir")"
            while [[ "$parent" != "$EXP_DIR" ]] && [[ -d "$parent" ]] && [[ -z "$(ls -A "$parent" 2>/dev/null)" ]]; do
                rmdir "$parent"
                parent="$(dirname "$parent")"
            done
        fi
    done
    inc_hr=$(numfmt --to=iec "$TOTAL_INCOMPLETE_SIZE" 2>/dev/null || echo "${TOTAL_INCOMPLETE_SIZE} bytes")
    if $FORCE; then
        echo "  → 已删除 ${#INCOMPLETE[@]} 个不完整实验，释放 $inc_hr。"
    else
        echo "  → 将释放 $inc_hr。"
    fi
fi

# ── B 类：玩具实验 ──
if [[ ${#TOY[@]} -gt 0 ]]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    printf "  %s B 类：玩具实验（num_epochs < %d，%d 个）\n" "$action_mode" "$TOY_MIN_EPOCHS" "${#TOY[@]}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    for exp_dir in "${TOY[@]}"; do
        size_hr=$(du -sh "$exp_dir" 2>/dev/null | cut -f1)
        rel_path="${exp_dir#$ROOT_DIR/}"
        echo "  $rel_path  ($size_hr)"
        echo "    ↳ ${TOY_INFO[$exp_dir]}"
    done
    echo ""

    if $FORCE; then
        if $YES_ALL; then
            for exp_dir in "${TOY[@]}"; do
                rm -rf "$exp_dir"
                parent="$(dirname "$exp_dir")"
                while [[ "$parent" != "$EXP_DIR" ]] && [[ -d "$parent" ]] && [[ -z "$(ls -A "$parent" 2>/dev/null)" ]]; do
                    rmdir "$parent"
                    parent="$(dirname "$parent")"
                done
            done
            toy_hr=$(numfmt --to=iec "$TOTAL_TOY_SIZE" 2>/dev/null || echo "${TOTAL_TOY_SIZE} bytes")
            echo "  → 已删除 ${#TOY[@]} 个玩具实验，释放 $toy_hr。"
        else
            echo "  逐项确认（y=删除 n=跳过 a=全部删除 q=全部跳过）："
            for exp_dir in "${TOY[@]}"; do
                rel_path="${exp_dir#$ROOT_DIR/}"
                read -r -p "    删除 $rel_path? [y/n/a/q] " ans
                case "$ans" in
                    a|A)
                        # 删除当前及后续所有
                        for d in "${TOY[@]}"; do
                            [[ -d "$d" ]] || continue
                            # 找到匹配的 exp_dir 起始位置
                            [[ "$d" < "$exp_dir" ]] && continue
                            rm -rf "$d"
                            parent="$(dirname "$d")"
                            while [[ "$parent" != "$EXP_DIR" ]] && [[ -d "$parent" ]] && [[ -z "$(ls -A "$parent" 2>/dev/null)" ]]; do
                                rmdir "$parent"
                                parent="$(dirname "$parent")"
                            done
                        done
                        echo "     → 已删除当前及后续所有玩具实验。"
                        break
                        ;;
                    q|Q)
                        echo "     → 已跳过剩余玩具实验。"
                        break
                        ;;
                    y|Y)
                        rm -rf "$exp_dir"
                        parent="$(dirname "$exp_dir")"
                        while [[ "$parent" != "$EXP_DIR" ]] && [[ -d "$parent" ]] && [[ -z "$(ls -A "$parent" 2>/dev/null)" ]]; do
                            rmdir "$parent"
                            parent="$(dirname "$parent")"
                        done
                        echo "     → 已删除。"
                        ;;
                    *)
                        echo "     → 已跳过。"
                        ;;
                esac
            done
        fi
    else
        toy_hr=$(numfmt --to=iec "$TOTAL_TOY_SIZE" 2>/dev/null || echo "${TOTAL_TOY_SIZE} bytes")
        echo "  → 加上 --force 可逐项确认删除（释放约 $toy_hr）；加上 --yes 直接删除。"
    fi
fi

# ── 被保护跳过的活跃实验 ──
if [[ ${#SKIPPED[@]} -gt 0 ]]; then
    echo ""
    echo "━━ 以下 ${#SKIPPED[@]} 个实验被保护跳过（最近 ${SKIP_ACTIVE_MINUTES}min 内有修改）："
    for s in "${SKIPPED[@]}"; do
        exp_dir="${s%%|*}"; reason="${s#*|}"
        rel_path="${exp_dir#$ROOT_DIR/}"
        echo "  $rel_path"
        echo "    ↳ $reason"
    done
fi

if ! $FORCE; then
    echo ""
    echo "以上为 dry-run 结果。加上 --force 执行删除。"
fi
