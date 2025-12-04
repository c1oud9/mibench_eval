#!/bin/bash
# ============================================================
# Step 1: MiBench 다운로드 및 빌드
# - MiBench 소스코드 다운로드
# - 각 벤치마크를 LLVM IR로 컴파일
# ============================================================

set -e

# 경로 설정
BASE_DIR="/data/sofusion20/mibench_eval"
MIBENCH_DIR="$BASE_DIR/mibench"
IR_DIR="$BASE_DIR/ir_data"


# LLVM 경로 
LLVM_BIN="/data/sofusion20/anaconda3/envs/llvm-opt/bin"

CLANG="$LLVM_BIN/clang"

echo "=== MiBench 벤치마크 설정 ==="

# 디렉토리 생성
mkdir -p "$BASE_DIR"
mkdir -p "$IR_DIR"

# MiBench 다운로드 (GitHub mirror)
if [ ! -d "$MIBENCH_DIR" ]; then
    echo "[1/3] MiBench 다운로드 중..."
    cd "$BASE_DIR"
    git clone https://github.com/embecosm/mibench.git
    echo "다운로드 완료"
else
    echo "[1/3] MiBench 이미 존재함: $MIBENCH_DIR"
fi

# 벤치마크 목록
BENCHMARKS=(
    "automotive/basicmath"
    "automotive/bitcount"
    "automotive/qsort"
    "automotive/susan"
    "consumer/jpeg"
    "consumer/lame"
    "consumer/tiff"
    "consumer/typeset"
    "network/dijkstra"
    "network/patricia"
    "office/ghostscript"
    "office/ispell"
    "office/rsynth"
    "office/stringsearch"
    "security/blowfish"
    "security/sha"
    "telecomm/adpcm"
    "telecomm/crc32"
    "telecomm/fft"
    "telecomm/gsm"
)

echo "[2/3] 소스 파일에서 LLVM IR 생성 중..."

# 각 벤치마크의 C 파일을 IR로 컴파일
for bench in "${BENCHMARKS[@]}"; do
    bench_name=$(basename "$bench")
    bench_path="$MIBENCH_DIR/$bench"
    ir_out_dir="$IR_DIR/$bench_name"
    
    if [ ! -d "$bench_path" ]; then
        echo "  [SKIP] $bench_name: 디렉토리 없음"
        continue
    fi
    
    mkdir -p "$ir_out_dir"
    
    echo "  Processing: $bench_name"
    
    # C 파일 찾아서 IR로 컴파일
    find "$bench_path" -name "*.c" -type f | while read -r src_file; do
        filename=$(basename "$src_file" .c)
        ir_file="$ir_out_dir/${filename}.ll"
        
        # 이미 존재하면 스킵
        if [ -f "$ir_file" ]; then
            continue
        fi
        
        # Unoptimized IR 생성 (-O0, -emit-llvm)
        # 헤더 경로 포함
        include_dir=$(dirname "$src_file")
        
        $CLANG -O0 -Xclang -disable-O0-optnone \
               -S -emit-llvm \
               -Wno-implicit-function-declaration \
               -I"$include_dir" \
               -I"$MIBENCH_DIR" \
               "$src_file" -o "$ir_file" 2>/dev/null || {
            echo "    [WARN] 컴파일 실패: $filename"
        }
    done
done

# 생성된 IR 파일 수 확인
IR_COUNT=$(find "$IR_DIR" -name "*.ll" | wc -l)
echo "[3/3] IR 생성 완료: $IR_COUNT 개 파일"

echo ""
echo "=== 설정 완료 ==="
echo "IR 디렉토리: $IR_DIR"
echo "다음 단계: python 02_generate_prompts.py"