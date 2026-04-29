#!/usr/bin/env python
"""實時監控微調測試進度，並在完成時生成最終報告"""
import json
import time
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent.parent
BENCHMARK_RESULTS_BASE = BASE_DIR / 'downloads' / 'benchmark_results'

def get_test_progress():
    """獲取四組微調測試的進度"""
    tests = {
        'turbo_dialogue_opt1': {
            'samples': ['zh_match_girl', 'zh_three_pigs', 'zh_fruit_cow'],
            'baseline': 0.818,
        },
        'belle_dialogue_opt1': {
            'samples': ['zh_match_girl', 'zh_three_pigs', 'zh_fruit_cow'],
            'baseline': 0.853,
        },
        'turbo_meeting_opt1': {
            'samples': ['zh_match_girl', 'zh_three_pigs', 'zh_fruit_cow'],
            'baseline': 0.857,
        },
        'belle_meeting_opt1': {
            'samples': ['zh_match_girl', 'zh_three_pigs', 'zh_fruit_cow'],
            'baseline': 0.866,
        },
    }
    
    results = {}
    for test_name, test_info in tests.items():
        # 查找最新的測試目錄
        test_dirs = list(BENCHMARK_RESULTS_BASE.glob(f'{test_name}_*'))
        if not test_dirs:
            results[test_name] = {
                'status': 'not_started',
                'progress': 0,
                'completed_samples': 0,
                'total_samples': len(test_info['samples']),
                'avg_accuracy': 0,
                'baseline': test_info['baseline'],
                'improvement': 0,
                'test_dir': None,
            }
            continue
        
        test_dir = sorted(test_dirs)[-1]  # 最新的目錄
        
        # 計算完成的樣本數
        completed_samples = []
        accuracy_values = []
        for sample_name in test_info['samples']:
            sample_file = test_dir / f'{sample_name}.json'
            if sample_file.exists():
                completed_samples.append(sample_name)
                try:
                    data = json.loads(sample_file.read_text(encoding='utf-8'))
                    accuracy = data.get('accuracy', 0)
                    accuracy_values.append(accuracy)
                except:
                    pass
        
        progress_pct = len(completed_samples) / len(test_info['samples']) * 100
        avg_accuracy = sum(accuracy_values) / len(accuracy_values) if accuracy_values else 0
        
        # 檢查是否完成 (有 summary.json)
        summary_file = test_dir / 'summary.json'
        is_complete = summary_file.exists()
        
        if is_complete:
            status = 'complete'
            try:
                data = json.loads(summary_file.read_text(encoding='utf-8'))
                if isinstance(data, list):
                    samples_list = [s for s in data if isinstance(s, dict)]
                elif isinstance(data, dict):
                    raw = data.get('samples', {})
                    if isinstance(raw, dict):
                        samples_list = list(raw.values())
                    elif isinstance(raw, list):
                        samples_list = raw
                    else:
                        samples_list = []
                else:
                    samples_list = []

                accuracy_values = [s.get('accuracy', 0) for s in samples_list if isinstance(s, dict)]
                avg_accuracy = sum(accuracy_values) / len(accuracy_values) if accuracy_values else 0
            except:
                pass
        elif len(completed_samples) > 0:
            status = 'in_progress'
        else:
            status = 'initializing'
        
        results[test_name] = {
            'status': status,
            'progress': progress_pct,
            'completed_samples': len(completed_samples),
            'total_samples': len(test_info['samples']),
            'avg_accuracy': avg_accuracy,
            'baseline': test_info['baseline'],
            'improvement': avg_accuracy - test_info['baseline'] if avg_accuracy > 0 else 0,
            'test_dir': str(test_dir),
        }
    
    return results

def print_progress_table(results):
    """打印進度表"""
    print("\n" + "=" * 100)
    print("四組微調測試進度監控".center(100))
    print("=" * 100)
    print(f"檢查時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 100)
    print(f"{'測試':<25} {'狀態':<15} {'進度':<10} {'平均准確':<12} {'基準':<10} {'改善':<10}")
    print("-" * 100)
    
    for test_name, info in results.items():
        status = info['status']
        
        if status == 'complete':
            status_str = '✓ 完成'
            progress_str = '100%'
            avg_acc_str = f"{info['avg_accuracy']:.1%}"
            baseline_str = f"{info['baseline']:.1%}"
            improvement_str = f"{info['improvement']:+.1%}"
        elif status == 'in_progress':
            status_str = f"⏳ 進行中 {info['completed_samples']}/{info['total_samples']}"
            progress_str = f"{info['progress']:.0f}%"
            avg_acc_str = f"{info['avg_accuracy']:.1%}" if info['avg_accuracy'] > 0 else "---"
            baseline_str = f"{info['baseline']:.1%}"
            improvement_str = f"{info['improvement']:+.1%}" if info['avg_accuracy'] > 0 else "---"
        else:
            status_str = "⏳ 初始化中"
            progress_str = "0%"
            avg_acc_str = "---"
            baseline_str = f"{info['baseline']:.1%}"
            improvement_str = "---"
        
        print(f"{test_name:<25} {status_str:<15} {progress_str:<10} {avg_acc_str:<12} {baseline_str:<10} {improvement_str:<10}")
    
    print("-" * 100)
    
    # 統計完成數量
    complete_count = sum(1 for r in results.values() if r['status'] == 'complete')
    total_count = len(results)
    print(f"總進度: {complete_count}/{total_count} 個測試完成\n")
    
    return complete_count == total_count

def main():
    print("開始監控微調測試...")
    start_time = time.time()
    
    while True:
        try:
            results = get_test_progress()
            all_complete = print_progress_table(results)
            
            if all_complete:
                print("\n✓ 所有測試已完成！")
                elapsed = time.time() - start_time
                print(f"總耗時: {elapsed/60:.1f} 分鐘")
                break
            
            # 每 10 秒檢查一次
            time.sleep(10)
            
        except Exception as e:
            print(f"監控錯誤: {e}")
            time.sleep(5)

if __name__ == '__main__':
    main()
