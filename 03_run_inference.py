#!/usr/bin/env python3
"""
Step 3: 모델 추론 - Pass List 생성
- Fine-tuned 모델로 최적화 pass list 예측
- GPU 메모리 효율적 처리
"""

import os
import json
import re
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel  # LoRA 모델용

# ============================================================
# 설정
# ============================================================
from config import (
    PROMPT_DIR, RESULTS_DIR, MODEL_DIR, MODEL_NAME,
    MAX_NEW_TOKENS, BASELINE_OPT
)

# 결과 디렉토리 생성
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 유효한 LLVM Pass 목록 (논문 Table 9 기반, 167개 중 주요 패스)
# ============================================================
VALID_PASSES = {
    # Module passes
    "always-inline", "argpromotion", "attributor", "called-value-propagation",
    "constmerge", "deadargelim", "elim-avail-extern", "extract-blocks",
    "forceattrs", "function-attrs", "globaldce", "globalopt", "globalsplit",
    "hotcoldsplit", "inferattrs", "inline", "internalize", "ipsccp",
    "loop-extract", "loop-extract-single", "mergefunc", "partial-inliner",
    "rewrite-statepoints-for-gc", "strip", "strip-dead-debug-info",
    "strip-dead-prototypes", "strip-debug-declare", "strip-nondebug",
    
    # Function passes
    "adce", "aggressive-instcombine", "alignment-from-assumptions",
    "bdce", "break-crit-edges", "callsite-splitting", "consthoist",
    "constraint-elimination", "correlated-propagation", "dce", "die",
    "dse", "early-cse", "early-cse-memssa", "fix-irreducible",
    "flattencfg", "float2int", "gvn", "gvn-hoist", "gvn-sink",
    "indvars", "infer-address-spaces", "instsimplify", "instcombine",
    "jump-threading", "lcssa", "licm", "loop-data-prefetch",
    "loop-deletion", "loop-distribute", "loop-fusion", "loop-idiom",
    "loop-instsimplify", "loop-interchange", "loop-load-elim",
    "loop-predication", "loop-reduce", "loop-reroll", "loop-rotate",
    "loop-simplifycfg", "loop-sink", "loop-unroll", "loop-unroll-and-jam",
    "loop-unswitch", "loop-vectorize", "lower-constant-intrinsics",
    "lower-expect", "lower-guard-intrinsic", "lower-widenable-condition",
    "loweratomic", "lowerinvoke", "lowerswitch", "mem2reg", "memcpyopt",
    "mergediceh", "mldst-motion", "nary-reassociate", "newgvn",
    "pgo-memop-opt", "reassociate", "reg2mem", "scalarizer", "sccp",
    "simplifycfg", "sink", "slp-vectorizer", "speculative-execution",
    "sroa", "tailcallelim", "unify-loop-exits",
}


def load_model(model_path: str, use_lora: bool = False, lora_path: str = None):
    """
    모델 로드
    - base model 또는 LoRA fine-tuned 모델
    """
    print(f"모델 로딩: {model_path}")
    
    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True,
        padding_side='left'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 모델
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # LoRA 어댑터 로드 (fine-tuned 모델인 경우)
    if use_lora and lora_path:
        print(f"LoRA 어댑터 로딩: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    
    model.eval()
    return model, tokenizer


def parse_pass_list(output: str) -> List[str]:
    """
    모델 출력에서 pass list 추출
    - <PASS_LIST> ... </PASS_LIST> 또는 직접 나열된 패스 파싱
    """
    passes = []
    
    # <PASS_LIST> 태그 내용 추출 시도
    match = re.search(r'<PASS_LIST>(.*?)</PASS_LIST>', output, re.DOTALL)
    if match:
        content = match.group(1)
    else:
        content = output
    
    # 패스 이름 추출 (다양한 형식 지원)
    # 형식 1: -pass-name
    # 형식 2: pass-name
    # 형식 3: module(pass-name) 또는 function(pass-name)
    
    for line in content.split('\n'):
        line = line.strip()
        
        # --passes= 형식
        if '--passes=' in line:
            passes_str = line.split('--passes=')[1].strip('"\'')
            for p in passes_str.split(','):
                p = p.strip()
                # module(xxx) 형식에서 xxx 추출
                inner = re.search(r'\((.*?)\)', p)
                if inner:
                    p = inner.group(1)
                if p in VALID_PASSES:
                    passes.append(p)
        
        # -pass-name 형식
        elif line.startswith('-'):
            pass_name = line.lstrip('-').split()[0]
            if pass_name in VALID_PASSES:
                passes.append(pass_name)
        
        # 단순 패스 이름
        else:
            words = line.replace(',', ' ').split()
            for w in words:
                w = w.strip('- ')
                if w in VALID_PASSES:
                    passes.append(w)
    
    # 중복 제거 (순서 유지)
    seen = set()
    unique_passes = []
    for p in passes:
        if p not in seen:
            seen.add(p)
            unique_passes.append(p)
    
    return unique_passes


def validate_pass_list(passes: List[str]) -> Tuple[bool, List[str]]:
    """
    Pass list 유효성 검증
    - 유효한 패스만 필터링
    """
    valid = [p for p in passes if p in VALID_PASSES]
    is_valid = len(valid) > 0
    return is_valid, valid


def generate_pass_list(model, tokenizer, prompt: str, 
                       max_new_tokens: int = MAX_NEW_TOKENS) -> Dict:
    """
    단일 프롬프트에 대해 pass list 생성
    """
    # 토크나이즈
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        truncation=True,
        max_length=15000
    ).to(model.device)
    
    # 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decoding
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 디코딩
    generated = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    
    # Pass list 파싱
    passes = parse_pass_list(generated)
    is_valid, valid_passes = validate_pass_list(passes)
    
    return {
        "raw_output": generated,
        "parsed_passes": passes,
        "valid_passes": valid_passes,
        "is_valid": is_valid
    }


def run_inference(model_path: str, use_lora: bool = False, 
                  lora_path: str = None, batch_size: int = 1):
    """
    전체 프롬프트에 대해 추론 실행
    """
    # 모델 로드
    model, tokenizer = load_model(model_path, use_lora, lora_path)
    
    # 프롬프트 로드
    prompt_file = PROMPT_DIR / "mibench_prompts.jsonl"
    prompts = []
    with open(prompt_file, 'r') as f:
        for line in f:
            prompts.append(json.loads(line))
    
    print(f"총 {len(prompts)}개 프롬프트 처리")
    
    # 추론 실행
    results = []
    for item in tqdm(prompts, desc="추론 진행"):
        result = generate_pass_list(model, tokenizer, item["prompt"])
        
        results.append({
            "benchmark": item["benchmark"],
            "source_file": item["source_file"],
            "type": item["type"],
            "token_count": item["token_count"],
            **result
        })
        
        # 메모리 정리
        torch.cuda.empty_cache()
    
    # 결과 저장
    output_file = RESULTS_DIR / "inference_results.jsonl"
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    # 통계
    valid_count = sum(1 for r in results if r["is_valid"])
    print(f"\n=== 추론 완료 ===")
    print(f"총 프롬프트: {len(results)}")

    if len(results) > 0:
        print(f"유효한 pass list: {valid_count} ({100*valid_count/len(results):.1f}%)")
    else:
        print(f"경고: 처리된 프롬프트가 없습니다!")
        
    print(f"결과 파일: {output_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MiBench 모델 추론")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                       help="Base 모델 경로")
    parser.add_argument("--lora", type=str, default=None,
                       help="LoRA 어댑터 경로 (fine-tuned 모델용)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="배치 크기")
    
    args = parser.parse_args()
    
    use_lora = args.lora is not None
    run_inference(args.model, use_lora, args.lora, args.batch_size)