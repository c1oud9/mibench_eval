#!/usr/bin/env python3
"""
Step 2: LLVM IR에서 Flag Tuning 프롬프트 생성
- LLM Compiler 논문 Listing 4 형식 준수
- 15k 토큰 초과시 함수별 분할 (llvm-extract 사용)
"""

import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from transformers import AutoTokenizer
from tqdm import tqdm

# ============================================================
# 설정 로드
# ============================================================
from config import (
    IR_DIR, PROMPT_DIR, MODEL_NAME, MAX_TOKENS,
    LLVM_EXTRACT, OPT, MIBENCH_BENCHMARKS
)

# ============================================================
# 토크나이저 초기화
# ============================================================
print("토크나이저 로딩...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# ============================================================
# Flag Tuning 프롬프트 템플릿 (논문 Listing 4 기반)
# ============================================================
PROMPT_TEMPLATE = """###Human: I have the following LLVM IR that I would like to optimize for minimal code size:
```
{ir_code}
```

Please provide a list of LLVM opt passes that will optimize this code for the smallest binary size.

###Assistant: I'll analyze the IR and suggest optimization passes.

<PASS_LIST>"""


def count_tokens(text: str) -> int:
    """텍스트의 토큰 수 계산"""
    return len(tokenizer.encode(text, add_special_tokens=False))


def read_ir_file(ir_path: Path) -> str:
    """IR 파일 읽기"""
    with open(ir_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def extract_functions(ir_path: Path, output_dir: Path) -> List[Path]:
    """
    IR 파일에서 개별 함수 추출 (llvm-extract 사용)
    - 큰 모듈을 함수별로 분할
    """
    ir_content = read_ir_file(ir_path)
    
    # 함수 이름 추출 (define 키워드로 시작하는 라인)
    functions = []
    for line in ir_content.split('\n'):
        if line.startswith('define '):
            # @함수이름( 패턴 추출
            start = line.find('@')
            end = line.find('(', start)
            if start != -1 and end != -1:
                func_name = line[start+1:end]
                functions.append(func_name)
    
    if not functions:
        return []
    
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_files = []
    
    for func_name in functions:
        output_file = output_dir / f"{func_name}.ll"
        
        try:
            # llvm-extract로 함수 추출
            result = subprocess.run(
                [str(LLVM_EXTRACT), 
                 f"--func={func_name}",
                 str(ir_path),
                 "-o", str(output_file).replace('.ll', '.bc')],
                capture_output=True,
                timeout=30
            )
            
            # BC를 LL로 변환
            if result.returncode == 0:
                bc_file = str(output_file).replace('.ll', '.bc')
                subprocess.run(
                    ["llvm-dis", bc_file, "-o", str(output_file)],
                    capture_output=True,
                    timeout=30
                )
                if output_file.exists():
                    extracted_files.append(output_file)
                    
        except (subprocess.TimeoutExpired, Exception) as e:
            continue
    
    return extracted_files


def create_prompt(ir_code: str) -> str:
    """IR 코드로 프롬프트 생성"""
    return PROMPT_TEMPLATE.format(ir_code=ir_code.strip())


def process_ir_file(ir_path: Path, benchmark_name: str) -> List[Dict]:
    """
    IR 파일 처리 및 프롬프트 생성
    - 15k 토큰 이하: 전체 모듈 사용
    - 15k 토큰 초과: 함수별 분할
    """
    prompts = []
    ir_code = read_ir_file(ir_path)
    prompt = create_prompt(ir_code)
    token_count = count_tokens(prompt)
    
    if token_count <= MAX_TOKENS:
        # 전체 모듈 프롬프트
        prompts.append({
            "benchmark": benchmark_name,
            "source_file": str(ir_path),
            "type": "module",
            "prompt": prompt,
            "token_count": token_count
        })
    else:
        # 함수별 분할 필요
        split_dir = ir_path.parent / f"{ir_path.stem}_split"
        extracted = extract_functions(ir_path, split_dir)
        
        for func_ir in extracted:
            func_code = read_ir_file(func_ir)
            func_prompt = create_prompt(func_code)
            func_tokens = count_tokens(func_prompt)
            
            if func_tokens <= MAX_TOKENS:
                prompts.append({
                    "benchmark": benchmark_name,
                    "source_file": str(func_ir),
                    "type": "function",
                    "prompt": func_prompt,
                    "token_count": func_tokens
                })
            # 여전히 너무 크면 스킵 (truncated)
    
    return prompts


def main():
    """메인 함수: 모든 IR 파일에서 프롬프트 생성"""
    print("=== MiBench 프롬프트 생성 ===")
    
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_prompts = []
    stats = {
        "total_ir_files": 0,
        "module_prompts": 0,
        "function_prompts": 0,
        "truncated": 0
    }
    
    # IR 디렉토리 순회
    for bench_dir in sorted(IR_DIR.iterdir()):
        if not bench_dir.is_dir():
            continue
            
        bench_name = bench_dir.name
        print(f"\n처리 중: {bench_name}")
        
        ir_files = list(bench_dir.glob("*.ll"))
        stats["total_ir_files"] += len(ir_files)
        
        for ir_path in tqdm(ir_files, desc=f"  {bench_name}"):
            prompts = process_ir_file(ir_path, bench_name)
            
            for p in prompts:
                if p["type"] == "module":
                    stats["module_prompts"] += 1
                else:
                    stats["function_prompts"] += 1
                    
            all_prompts.extend(prompts)
    
    # 프롬프트 저장
    output_file = PROMPT_DIR / "mibench_prompts.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for p in all_prompts:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    
    # 통계 출력
    print("\n=== 생성 완료 ===")
    print(f"총 IR 파일: {stats['total_ir_files']}")
    print(f"모듈 프롬프트: {stats['module_prompts']}")
    print(f"함수 프롬프트: {stats['function_prompts']}")
    print(f"총 프롬프트: {len(all_prompts)}")
    print(f"출력 파일: {output_file}")


if __name__ == "__main__":
    main()