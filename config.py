"""
MiBench 벤치마크 평가 파이프라인 설정
- LLM Compiler 논문 방법론 기반
- Seraph Aurora 클러스터 환경용
"""

from pathlib import Path

# ============================================================
# 경로 설정
# ============================================================
BASE_DIR = Path("/data/sofusion20/mibench_eval")
MIBENCH_DIR = BASE_DIR / "mibench"
IR_DIR = BASE_DIR / "ir_data"
PROMPT_DIR = BASE_DIR / "prompts"
RESULTS_DIR = BASE_DIR / "results"
MODEL_DIR = Path("/data/sofusion20/models")

# ============================================================
# LLVM 도구 경로 (Seraph에 설치된 LLVM 17.0.6 기준)
# ============================================================
LLVM_BIN = Path("/data/sofusion20/llvm-17/bin")  # 필요시 수정
CLANG = LLVM_BIN / "clang"
OPT = LLVM_BIN / "opt"
LLC = LLVM_BIN / "llc"
LLVM_EXTRACT = LLVM_BIN / "llvm-extract"
LLVM_SIZE = LLVM_BIN / "llvm-size"


# ============================================================
# MiBench 벤치마크 목록 (논문 Table 10 기준)
# ============================================================
MIBENCH_BENCHMARKS = [
    "automotive/basicmath",
    "automotive/bitcount",
    "automotive/qsort",
    "automotive/susan",
    "consumer/jpeg",      # jpeg_c, jpeg_d
    "consumer/lame",
    "consumer/tiff",      # tiff2bw, tiff2rgba, tiffdither, tiffmedian
    "consumer/typeset",
    "network/dijkstra",
    "network/patricia",
    "office/ghostscript",
    "office/ispell",
    "office/rsynth",
    "office/stringsearch",
    "security/blowfish",
    "security/sha",
    "telecomm/adpcm",
    "telecomm/crc32",
    "telecomm/fft",
    "telecomm/gsm",
]

# ============================================================
# 모델 설정
# ============================================================
MODEL_NAME = ""  # 또는 fine-tuned 모델 경로
MAX_TOKENS = 15000  # 논문: 15k token context window
MAX_NEW_TOKENS = 2048

# ============================================================
# 평가 설정
# ============================================================
BASELINE_OPT = "-Oz"  # 기준 최적화 레벨
TIMEOUT_SECONDS = 120  # 컴파일 타임아웃
NUM_PASSES_MIN = 1
NUM_PASSES_MAX = 50

# ============================================================
# SLURM 설정 (Aurora 클러스터)
# ============================================================
SLURM_CONFIG = {
    "partition": "ugrad",
    "account": "ugrad",
    "qos": "ugrad",
    "gres": "gpu:1",
    "cpus_per_gpu": 8,
    "mem_per_gpu": "29G",
    "time": "24:00:00",
}


