#!/usr/bin/env python
"""
四組模型×情境的微調候選參數測試
測試 turbo_dialogue, belle_dialogue, turbo_meeting, belle_meeting 四個組合
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent.parent
BENCHMARK_RESULTS_BASE = BASE_DIR / 'downloads' / 'benchmark_results'
PYTHON_EXE = BASE_DIR / 'dist' / 'SyncTranslate-onedir' / 'runtimes' / 'faster_whisper' / 'Scripts' / 'python.exe'

# 四組微調候選參數 (改進版 20260429)
# 基於第一個樣本的教訓，turbo_dialogue_opt1 改為增加 soft_final 而非減少
TUNING_CANDIDATES = {
    'turbo_dialogue_opt1': {
        'model': 'large-v3-turbo',
        'preset': 'turbo_dialogue_opt1',
        'params': {
            'vad': {'min_silence_duration_ms': 280, 'speech_pad_ms': 220},
            'streaming': {'soft_final_audio_ms': 2000, 'final_history_seconds': 10},
            'asr': {'no_speech_threshold': 0.32}
        },
        'tag': 'turbo_dialogue_stable_plus',
        'baseline_avg': 0.818,
    },
    'belle_dialogue_opt1': {
        'model': str(BASE_DIR / 'runtimes' / 'models' / 'belle-zh-ct2'),
        'preset': 'belle_dialogue_opt1',
        'params': {
            'vad': {'min_silence_duration_ms': 310, 'speech_pad_ms': 220},
            'streaming': {'soft_final_audio_ms': 2100, 'final_history_seconds': 16},
            'asr': {'no_speech_threshold': 0.32, 'beam_size': 2, 'final_beam_size': 5}
        },
        'tag': 'belle_dialogue_optimized',
        'baseline_avg': 0.853,
    },
    'turbo_meeting_opt1': {
        'model': 'large-v3-turbo',
        'preset': 'turbo_meeting_opt1',
        'params': {
            'vad': {'min_silence_duration_ms': 600, 'speech_pad_ms': 360},
            'streaming': {'soft_final_audio_ms': 4000, 'final_history_seconds': 20},
            'asr': {'no_speech_threshold': 0.40}
        },
        'tag': 'turbo_meeting_tuned',
        'baseline_avg': 0.857,
    },
    'belle_meeting_opt1': {
        'model': str(BASE_DIR / 'runtimes' / 'models' / 'belle-zh-ct2'),
        'preset': 'belle_meeting_opt1',
        'params': {
            'vad': {'min_silence_duration_ms': 640, 'speech_pad_ms': 360},
            'streaming': {'soft_final_audio_ms': 4100, 'final_history_seconds': 20},
            'asr': {'no_speech_threshold': 0.40}
        },
        'tag': 'belle_meeting_fine_tuned',
        'baseline_avg': 0.866,
    },
}

def run_benchmark_with_params(test_id, candidate_info):
    """為單個候選執行 benchmark"""
    print(f"\n{'='*80}")
    print(f"執行微調測試: {test_id}")
    print(f"{'='*80}")
    print(f"模型: {candidate_info['model']}")
    print(f"標籤: {candidate_info['tag']}")
    print(f"基準平均: {candidate_info['baseline_avg']:.1%}")
    print(f"預設參數:\n{json.dumps(candidate_info['params'], indent=2)}")
    
    # 建立輸出目錄
    output_dir = BENCHMARK_RESULTS_BASE / f'microtuning_{test_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 構建命令
    cmd = [
        str(PYTHON_EXE),
        str(BASE_DIR / 'tools' / 'asr_benchmark' / 'run_multi_benchmark.py'),
        '--model', candidate_info['model'],
        '--preset', candidate_info['preset'],
        '--only-lang', 'zh',
        '--output-dir', str(output_dir),
    ]
    
    print(f"\n執行命令:\n{' '.join(cmd)}\n")
    
    try:
        # 使用 bytes 讀取再手動 decode，避免 Windows cp950 與 UTF-8 混雜造成崩潰。
        result = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=False, timeout=600)

        stdout_text = (result.stdout or b"").decode("utf-8", errors="replace")
        stderr_text = (result.stderr or b"").decode("utf-8", errors="replace")

        print("STDOUT:")
        print(stdout_text[-1000:] if len(stdout_text) > 1000 else stdout_text)

        if result.returncode != 0:
            print("\nSTDERR:")
            print(stderr_text[-1000:] if len(stderr_text) > 1000 else stderr_text)
            return None, output_dir
        
        # 讀取 summary.json 確認結果
        summary_file = output_dir / 'summary.json'
        if summary_file.exists():
            summary = json.loads(summary_file.read_text(encoding='utf-8'))
            return summary, output_dir
        
    except subprocess.TimeoutExpired:
        print(f"超時: {test_id}")
        return None, output_dir
    except Exception as e:
        print(f"錯誤: {e}")
        return None, output_dir
    
    return None, output_dir

def main():
    print("\n" + "="*80)
    print("SyncTranslate 四組模型微調測試套件".center(80))
    print("="*80)
    print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"測試組數: {len(TUNING_CANDIDATES)}")
    
    results_summary = {}
    
    for test_id, candidate_info in TUNING_CANDIDATES.items():
        summary, output_dir = run_benchmark_with_params(test_id, candidate_info)
        
        if summary:
            # run_multi_benchmark 的 summary.json 為 list[dict]；保留 dict 格式相容性。
            if isinstance(summary, list):
                samples_list = [s for s in summary if isinstance(s, dict)]
            elif isinstance(summary, dict):
                raw = summary.get('samples', {})
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
            
            results_summary[test_id] = {
                'output_dir': str(output_dir),
                'baseline_avg': candidate_info['baseline_avg'],
                'tuned_avg': avg_accuracy,
                'improvement': avg_accuracy - candidate_info['baseline_avg'],
                'improvement_pct': ((avg_accuracy - candidate_info['baseline_avg']) / candidate_info['baseline_avg'] * 100) if candidate_info['baseline_avg'] > 0 else 0,
                'tag': candidate_info['tag'],
                'samples': samples_list,
            }
            
            print(f"\n✓ 完成: {test_id}")
            print(f"  基準: {candidate_info['baseline_avg']:.1%} → 微調: {avg_accuracy:.1%}")
            print(f"  改善: {results_summary[test_id]['improvement']:+.1%} ({results_summary[test_id]['improvement_pct']:+.1f}%)")
        else:
            results_summary[test_id] = {
                'output_dir': str(output_dir),
                'status': 'FAILED',
            }
            print(f"\n✗ 失敗: {test_id}")
    
    # 輸出對比表
    print("\n" + "="*80)
    print("四組微調結果對比".center(80))
    print("="*80)
    print(f"{'測試ID':<25} {'基準':>10} {'微調':>10} {'改善':>12} {'改善%':>10}")
    print("-" * 70)
    
    for test_id, result in results_summary.items():
        if 'status' not in result:
            print(f"{test_id:<25} {result['baseline_avg']:>9.1%} {result['tuned_avg']:>9.1%} "
                  f"{result['improvement']:>11.1%} {result['improvement_pct']:>9.1f}%")
        else:
            print(f"{test_id:<25} {'FAILED':>10}")
    
    # 儲存詳細結果
    output_file = BENCHMARK_RESULTS_BASE / f'microtuning_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    output_file.write_text(json.dumps(results_summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\n詳細結果已儲存: {output_file}")
    
    print(f"\n結束時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()
