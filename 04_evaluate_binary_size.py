#!/usr/bin/env python3
"""
Step 4: 최적화 적용 및 Binary Size 측정
- 모델 생성 pass list로 최적화 적용
- -Oz baseline과 비교
- 논문 방법론: binary size 기준 평가
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import multiprocessing

# ============================================================
# 설정
# ============================================================
from config import (
    IR_DIR, RESULTS_DIR, 
    CLANG, OPT, LLC, LLVM_SIZE,
    BASELINE_OPT, TIMEOUT_SECONDS
)


def get_binary_size(obj_file: Path) -> int:
    """
    Object 파일의 text section 크기 반환
    - llvm-size 또는 size 명령어 사용
    """
    try:
        result = subprocess.run(
            [str(LLVM_SIZE), str(obj_file)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            # llvm-size 출력 파싱
            # text    data     bss     dec     hex filename
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                parts = lines[1].split()
                if len(parts) >= 1:
                    return int(parts[0])  # text section
        
        # fallback: 파일 크기
        return obj_file.stat().st_size
        
    except Exception:
        return -1


def compile_with_passes(ir_file: Path, passes: List[str], 
                       output_obj: Path) -> Tuple[bool, int]:
    """
    주어진 pass list로 IR 최적화 후 object 파일 생성
    Returns: (성공여부, binary_size)
    """
    try:
        with tempfile.NamedTemporaryFile(suffix='.ll', delete=False) as tmp_opt:
            tmp_opt_path = tmp_opt.name
        
        # Pass list를 opt 명령어 형식으로 변환
        # LLVM 17+ 형식: --passes="pass1,pass2,..."
        pass_str = ','.join(passes)
        
        # opt로 최적화 적용
        opt_cmd = [
            str(OPT),
            f'--passes={pass_str}',
            str(ir_file),
            '-S', '-o', tmp_opt_path
        ]
        
        result = subprocess.run(
            opt_cmd,
            capture_output=True,
            timeout=TIMEOUT_SECONDS
        )
        
        if result.returncode != 0:
            return False, -1
        
        # 최적화된 IR을 object로 컴파일
        compile_cmd = [
            str(CLANG),
            '-c', tmp_opt_path,
            '-o', str(output_obj)
        ]
        
        result = subprocess.run(
            compile_cmd,
            capture_output=True,
            timeout=TIMEOUT_SECONDS
        )
        
        # 임시 파일 정리
        os.unlink(tmp_opt_path)
        
        if result.returncode != 0:
            return False, -1
        
        # Binary size 측정
        size = get_binary_size(output_obj)
        return True, size
        
    except subprocess.TimeoutExpired:
        return False, -1
    except Exception as e:
        return False, -1


def compile_baseline(ir_file: Path, output_obj: Path, 
                    opt_level: str = BASELINE_OPT) -> Tuple[bool, int]:
    """
    Baseline (-Oz) 최적화로 컴파일
    """
    try:
        # IR을 직접 clang으로 컴파일 (opt_level 적용)
        compile_cmd = [
            str(CLANG),
            opt_level,
            '-c', str(ir_file),
            '-o', str(output_obj)
        ]
        
        result = subprocess.run(
            compile_cmd,
            capture_output=True,
            timeout=TIMEOUT_SECONDS
        )
        
        if result.returncode != 0:
            return False, -1
        
        size = get_binary_size(output_obj)
        return True, size
        
    except subprocess.TimeoutExpired:
        return False, -1
    except Exception:
        return False, -1


def evaluate_single(item: Dict, work_dir: Path) -> Dict:
    """
    단일 IR 파일에 대해 최적화 평가
    """
    ir_file = Path(item["source_file"])
    
    if not ir_file.exists():
        return {**item, "error": "IR file not found"}
    
    result = {
        "benchmark": item["benchmark"],
        "source_file": str(ir_file),
        "type": item["type"],
    }
    
    # Baseline (-Oz) 컴파일
    baseline_obj = work_dir / f"{ir_file.stem}_baseline.o"
    baseline_ok, baseline_size = compile_baseline(ir_file, baseline_obj)
    
    result["baseline_compiled"] = baseline_ok
    result["baseline_size"] = baseline_size
    
    if not baseline_ok:
        result["error"] = "Baseline compilation failed"
        return result
    
    # 모델 생성 pass list로 컴파일
    passes = item.get("valid_passes", [])
    
    if not passes:
        # Pass list가 없으면 baseline 사용
        result["model_compiled"] = False
        result["model_size"] = -1
        result["used_backup"] = True
        result["final_size"] = baseline_size
        result["improvement"] = 0.0
    else:
        model_obj = work_dir / f"{ir_file.stem}_model.o"
        model_ok, model_size = compile_with_passes(ir_file, passes, model_obj)
        
        result["model_compiled"] = model_ok
        result["model_size"] = model_size
        result["passes_used"] = passes
        
        if model_ok and model_size > 0:
            # 모델 최적화 성공
            if model_size <= baseline_size:
                # 개선됨
                result["used_backup"] = False
                result["final_size"] = model_size
            else:
                # 오히려 나빠짐 -> backup (-Oz) 사용
                result["used_backup"] = True
                result["final_size"] = baseline_size
        else:
            # 모델 최적화 실패 -> backup 사용
            result["used_backup"] = True
            result["final_size"] = baseline_size
        
        # 개선율 계산
        result["improvement"] = (baseline_size - result["final_size"]) / baseline_size * 100
    
    # 임시 파일 정리
    for f in work_dir.glob(f"{ir_file.stem}_*.o"):
        f.unlink()
    
    return result


def run_evaluation():
    """
    전체 평가 실행
    """
    print("=== MiBench Binary Size 평가 ===")
    
    # 추론 결과 로드
    inference_file = RESULTS_DIR / "inference_results.jsonl"
    if not inference_file.exists():
        print(f"추론 결과 파일 없음: {inference_file}")
        print("먼저 03_run_inference.py를 실행하세요.")
        return
    
    items = []
    with open(inference_file, 'r') as f:
        for line in f:
            items.append(json.loads(line))
    
    print(f"평가 대상: {len(items)}개 파일")
    
    # 작업 디렉토리 생성
    work_dir = RESULTS_DIR / "tmp_objects"
    work_dir.mkdir(exist_ok=True)
    
    # 평가 실행
    results = []
    for item in tqdm(items, desc="평가 진행"):
        result = evaluate_single(item, work_dir)
        results.append(result)
    
    # 작업 디렉토리 정리
    import shutil
    shutil.rmtree(work_dir, ignore_errors=True)
    
    # 결과 저장
    output_file = RESULTS_DIR / "evaluation_results.jsonl"
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    # 통계 계산
    compute_statistics(results)
    
    print(f"\n결과 저장: {output_file}")
    return results


def compute_statistics(results: List[Dict]):
    """
    평가 결과 통계 계산 및 출력
    """
    print("\n" + "="*60)
    print("평가 결과 요약")
    print("="*60)
    
    # 전체 통계
    total = len(results)
    baseline_compiled = sum(1 for r in results if r.get("baseline_compiled", False))
    model_compiled = sum(1 for r in results if r.get("model_compiled", False))
    
    # 개선/악화 카운트 (유효한 결과만)
    valid_results = [r for r in results if r.get("baseline_compiled") and "improvement" in r]
    
    improved = sum(1 for r in valid_results if r["improvement"] > 0)
    regressed = sum(1 for r in valid_results if r["improvement"] < 0)
    unchanged = sum(1 for r in valid_results if r["improvement"] == 0)
    
    # 평균 개선율
    improvements = [r["improvement"] for r in valid_results if r["improvement"] != 0]
    avg_improvement = sum(improvements) / len(improvements) if improvements else 0
    
    # Zero-shot 개선율 (backup 미사용)
    zero_shot_results = [r for r in valid_results if not r.get("used_backup", True)]
    zero_shot_improvement = sum(r["improvement"] for r in zero_shot_results) / len(zero_shot_results) if zero_shot_results else 0
    
    # -Oz backup 사용시 개선율 (항상 >= 0)
    backup_results = [r for r in valid_results]
    backup_improvement = sum(max(0, r["improvement"]) for r in backup_results) / len(backup_results) if backup_results else 0
    
    print(f"\n[전체 통계]")
    print(f"  총 테스트: {total}")
    print(f"  Baseline 컴파일 성공: {baseline_compiled}")
    print(f"  모델 최적화 컴파일 성공: {model_compiled}")
    
    print(f"\n[성능 비교] (vs -Oz baseline)")
    print(f"  개선된 파일: {improved} ({100*improved/len(valid_results):.1f}%)")
    print(f"  악화된 파일: {regressed} ({100*regressed/len(valid_results):.1f}%)")
    print(f"  동일: {unchanged}")
    
    print(f"\n[개선율]")
    print(f"  Zero-shot 평균: {zero_shot_improvement:.2f}%")
    print(f"  -Oz backup 적용시: {backup_improvement:.2f}%")
    
    # 벤치마크별 통계
    print(f"\n[벤치마크별 개선율]")
    benchmark_stats = {}
    for r in valid_results:
        bench = r["benchmark"]
        if bench not in benchmark_stats:
            benchmark_stats[bench] = []
        benchmark_stats[bench].append(r["improvement"])
    
    for bench, imps in sorted(benchmark_stats.items()):
        avg = sum(imps) / len(imps)
        print(f"  {bench:20s}: {avg:+.2f}% ({len(imps)} files)")


if __name__ == "__main__":
    run_evaluation()