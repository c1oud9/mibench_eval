#!/usr/bin/env python3
"""
Step 5: 최종 결과 집계 및 리포트 생성
- LLM Compiler 논문 Table 3 형식 재현
- 벤치마크별 상세 결과
- 시각화 (matplotlib)
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from datetime import datetime

# ============================================================
# 설정
# ============================================================
from config import RESULTS_DIR, MODEL_NAME

# 출력 디렉토리
REPORT_DIR = RESULTS_DIR / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_results() -> List[Dict]:
    """평가 결과 로드"""
    result_file = RESULTS_DIR / "evaluation_results.jsonl"
    results = []
    with open(result_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def create_summary_table(results: List[Dict]) -> pd.DataFrame:
    """
    논문 Table 3 형식의 요약 테이블 생성
    - Size: 모델 크기
    - Improved: 개선된 파일 수
    - Regressed: 악화된 파일 수  
    - Zero-shot: backup 없이 순수 모델 성능
    - -Oz backup: backup 적용시 성능
    """
    valid_results = [r for r in results if r.get("baseline_compiled")]
    
    improved = sum(1 for r in valid_results if r.get("improvement", 0) > 0)
    regressed = sum(1 for r in valid_results if r.get("improvement", 0) < 0 and not r.get("used_backup"))
    
    # Zero-shot: 모든 결과 평균 (악화 포함)
    zero_shot_improvements = [r.get("improvement", 0) for r in valid_results]
    zero_shot_avg = sum(zero_shot_improvements) / len(zero_shot_improvements) if zero_shot_improvements else 0
    
    # -Oz backup: 악화시 0으로 처리
    backup_improvements = [max(0, r.get("improvement", 0)) for r in valid_results]
    backup_avg = sum(backup_improvements) / len(backup_improvements) if backup_improvements else 0
    
    summary = {
        "Model": [MODEL_NAME.split('/')[-1]],
        "Improved": [improved],
        "Regressed": [regressed],
        "Zero-shot (%)": [f"{zero_shot_avg:.2f}"],
        "-Oz backup (%)": [f"{backup_avg:.2f}"]
    }
    
    return pd.DataFrame(summary)


def create_benchmark_table(results: List[Dict]) -> pd.DataFrame:
    """벤치마크별 상세 결과 테이블"""
    valid_results = [r for r in results if r.get("baseline_compiled")]
    
    benchmark_data = defaultdict(lambda: {
        "files": 0,
        "improved": 0,
        "regressed": 0,
        "total_baseline": 0,
        "total_optimized": 0,
        "improvements": []
    })
    
    for r in valid_results:
        bench = r["benchmark"]
        data = benchmark_data[bench]
        
        data["files"] += 1
        data["improvements"].append(r.get("improvement", 0))
        data["total_baseline"] += r.get("baseline_size", 0)
        data["total_optimized"] += r.get("final_size", r.get("baseline_size", 0))
        
        if r.get("improvement", 0) > 0:
            data["improved"] += 1
        elif r.get("improvement", 0) < 0:
            data["regressed"] += 1
    
    rows = []
    for bench, data in sorted(benchmark_data.items()):
        avg_improvement = sum(data["improvements"]) / len(data["improvements"]) if data["improvements"] else 0
        size_reduction = (data["total_baseline"] - data["total_optimized"]) / data["total_baseline"] * 100 if data["total_baseline"] > 0 else 0
        
        rows.append({
            "Benchmark": bench,
            "Files": data["files"],
            "Improved": data["improved"],
            "Regressed": data["regressed"],
            "Avg Improvement (%)": f"{avg_improvement:.2f}",
            "Total Size Reduction (%)": f"{size_reduction:.2f}"
        })
    
    return pd.DataFrame(rows)


def create_improvement_chart(results: List[Dict], output_path: Path):
    """
    논문 Figure 6 형식의 벤치마크별 개선율 차트
    """
    valid_results = [r for r in results if r.get("baseline_compiled")]
    
    # 벤치마크별 평균 개선율
    benchmark_improvements = defaultdict(list)
    for r in valid_results:
        benchmark_improvements[r["benchmark"]].append(r.get("improvement", 0))
    
    benchmarks = sorted(benchmark_improvements.keys())
    avg_improvements = [
        sum(benchmark_improvements[b]) / len(benchmark_improvements[b])
        for b in benchmarks
    ]
    
    # 차트 생성
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in avg_improvements]
    bars = ax.bar(range(len(benchmarks)), avg_improvements, color=colors)
    
    ax.set_xticks(range(len(benchmarks)))
    ax.set_xticklabels(benchmarks, rotation=45, ha='right')
    ax.set_ylabel('Improvement over -Oz (%)')
    ax.set_xlabel('Benchmark')
    ax.set_title('MiBench Optimization Performance')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"차트 저장: {output_path}")


def create_distribution_chart(results: List[Dict], output_path: Path):
    """개선율 분포 히스토그램"""
    valid_results = [r for r in results if r.get("baseline_compiled")]
    improvements = [r.get("improvement", 0) for r in valid_results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(improvements, bins=50, edgecolor='white', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', label='Baseline (-Oz)')
    ax.axvline(x=sum(improvements)/len(improvements), color='green', 
               linestyle='--', label=f'Mean: {sum(improvements)/len(improvements):.2f}%')
    
    ax.set_xlabel('Improvement over -Oz (%)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Optimization Improvements')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"차트 저장: {output_path}")


def generate_report(results: List[Dict]):
    """전체 리포트 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 요약 테이블
    summary_df = create_summary_table(results)
    summary_path = REPORT_DIR / "summary_table.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n요약 테이블 저장: {summary_path}")
    print(summary_df.to_string(index=False))
    
    # 2. 벤치마크별 테이블
    benchmark_df = create_benchmark_table(results)
    benchmark_path = REPORT_DIR / "benchmark_results.csv"
    benchmark_df.to_csv(benchmark_path, index=False)
    print(f"\n벤치마크별 결과 저장: {benchmark_path}")
    print(benchmark_df.to_string(index=False))
    
    # 3. 차트 생성
    create_improvement_chart(results, REPORT_DIR / "improvement_chart.png")
    create_distribution_chart(results, REPORT_DIR / "distribution_chart.png")
    
    # 4. 텍스트 리포트
    report_text = generate_text_report(results, summary_df, benchmark_df)
    report_path = REPORT_DIR / f"evaluation_report_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"\n텍스트 리포트 저장: {report_path}")
    
    return report_path


def generate_text_report(results: List[Dict], summary_df: pd.DataFrame, 
                        benchmark_df: pd.DataFrame) -> str:
    """텍스트 형식 리포트 생성"""
    valid_results = [r for r in results if r.get("baseline_compiled")]
    
    report = []
    report.append("=" * 70)
    report.append("MiBench 벤치마크 최적화 성능 평가 리포트")
    report.append("=" * 70)
    report.append(f"\n평가 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"모델: {MODEL_NAME}")
    report.append(f"Baseline: -Oz")
    
    report.append("\n" + "-" * 70)
    report.append("1. 전체 요약 (논문 Table 3 형식)")
    report.append("-" * 70)
    report.append(summary_df.to_string(index=False))
    
    report.append("\n" + "-" * 70)
    report.append("2. 벤치마크별 상세 결과")
    report.append("-" * 70)
    report.append(benchmark_df.to_string(index=False))
    
    report.append("\n" + "-" * 70)
    report.append("3. 통계 요약")
    report.append("-" * 70)
    
    improvements = [r.get("improvement", 0) for r in valid_results]
    report.append(f"  총 테스트 파일: {len(valid_results)}")
    report.append(f"  평균 개선율: {sum(improvements)/len(improvements):.2f}%")
    report.append(f"  최대 개선: {max(improvements):.2f}%")
    report.append(f"  최대 악화: {min(improvements):.2f}%")
    
    # Pass 사용 통계
    pass_counts = defaultdict(int)
    for r in valid_results:
        for p in r.get("passes_used", []):
            pass_counts[p] += 1
    
    if pass_counts:
        report.append("\n  가장 많이 사용된 Pass (Top 10):")
        for pass_name, count in sorted(pass_counts.items(), key=lambda x: -x[1])[:10]:
            report.append(f"    {pass_name}: {count}")
    
    report.append("\n" + "=" * 70)
    report.append("리포트 끝")
    report.append("=" * 70)
    
    return '\n'.join(report)


def main():
    """메인 함수"""
    print("=== MiBench 평가 리포트 생성 ===")
    
    try:
        results = load_results()
    except FileNotFoundError:
        print("평가 결과 파일이 없습니다.")
        print("먼저 04_evaluate_binary_size.py를 실행하세요.")
        return
    
    print(f"로드된 결과: {len(results)}개")
    
    report_path = generate_report(results)
    
    print("\n=== 완료 ===")
    print(f"리포트 디렉토리: {REPORT_DIR}")


if __name__ == "__main__":
    main()