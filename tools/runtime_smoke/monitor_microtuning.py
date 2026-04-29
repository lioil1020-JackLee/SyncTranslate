#!/usr/bin/env python
"""監控微調測試的進度"""
import json
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
BENCHMARK_RESULTS_BASE = BASE_DIR / 'downloads' / 'benchmark_results'

EXPECTED_TESTS = [
    'microtuning_turbo_dialogue_opt1_20260429_204924',
    # 其他會依次生成
]

def check_progress():
    print("微調測試進度監控")
    print("=" * 80)
    
    while True:
        print(f"\n檢查時間: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 掃描所有 microtuning_* 目錄
        microtuning_dirs = sorted([
            d for d in BENCHMARK_RESULTS_BASE.glob('microtuning_*')
            if d.is_dir()
        ])
        
        for test_dir in microtuning_dirs:
            # 查看 summary.json
            summary_file = test_dir / 'summary.json'
            if summary_file.exists():
                try:
                    data = json.loads(summary_file.read_text(encoding='utf-8'))
                    if isinstance(data, list):
                        samples_count = len([x for x in data if isinstance(x, dict)])
                    elif isinstance(data, dict):
                        samples = data.get('samples', {})
                        if isinstance(samples, dict):
                            samples_count = len(samples)
                        elif isinstance(samples, list):
                            samples_count = len(samples)
                        else:
                            samples_count = 0
                    else:
                        samples_count = 0
                    print(f"✓ {test_dir.name}: {samples_count} 個樣本完成")
                except Exception as e:
                    print(f"✗ {test_dir.name}: JSON 解析失敗 - {e}")
            else:
                # 查看是否有中間結果
                json_files = list(test_dir.glob('zh_*.json'))
                if json_files:
                    print(f"⏳ {test_dir.name}: {len(json_files)} 個樣本正在進行...")
                else:
                    print(f"⏳ {test_dir.name}: 初始化中...")
        
        # 檢查最終的 microtuning_summary 文件
        summary_files = list(BENCHMARK_RESULTS_BASE.glob('microtuning_summary_*.json'))
        if summary_files:
            latest_summary = sorted(summary_files)[-1]
            print(f"\n最新匯總: {latest_summary.name}")
            try:
                data = json.loads(latest_summary.read_text(encoding='utf-8'))
                for test_id, result in data.items():
                    if 'status' not in result:
                        baseline = result.get('baseline_avg', 0)
                        tuned = result.get('tuned_avg', 0)
                        improvement = result.get('improvement', 0)
                        print(f"  {test_id}: {baseline:.1%} → {tuned:.1%} ({improvement:+.1%})")
                break  # 測試全部完成
            except Exception as e:
                print(f"讀取匯總失敗: {e}")
        
        time.sleep(5)  # 每 5 秒檢查一次
        print(".", end="", flush=True)

if __name__ == '__main__':
    check_progress()
