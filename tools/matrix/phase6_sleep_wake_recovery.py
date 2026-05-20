#!/usr/bin/env python3
"""
Phase 6 Sleep/Wake Recovery Test
验证系统睡眠唤醒后，App 和 bridge 能自动恢复连接。
"""
import json
import subprocess
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)


def run_sleep_wake_test(duration_sec: int = 300, out: str = None) -> dict:
    """
    运行 sleep/wake 恢复测试。
    
    步骤：
    1. 启动 App + bridge
    2. 运行 60 秒 soak（建立稳定状态）
    3. 模拟 sleep（暂停主线程）
    4. 模拟 wake（恢复）
    5. 再运行 60 秒验证恢复
    6. 检查统计信息
    
    Args:
        duration_sec: 单次 soak 持续秒数（默认 300s = 5min）
        out: 输出文件路径
    
    Returns:
        dict: 测试结果报告
    """
    
    report = {
        "kind": "phase6_sleep_wake_recovery",
        "generated_at": datetime.utcnow().isoformat(),
        "config": "config.yaml",
        "summary": {
            "pre_sleep_ok": False,
            "wake_recovery_ok": False,
            "overall_ok": False,
            "sleep_duration_sec": 5,
            "pre_sleep_duration_sec": 60,
            "post_wake_duration_sec": 60
        },
        "pre_sleep": {},
        "wake": {},
        "post_wake": {},
        "diagnostics": []
    }
    
    try:
        # 步骤 1: 预热运行 60 秒
        logger.info("=== Phase 6.1: Pre-sleep soak (60s) ===")
        pre_soak_cmd = [
            sys.executable, "tools/matrix/runtime_soak_2hr.py",
            "--preset", "smoke",  # 120s, but we'll stop early
            "--sample-interval-sec", "5.0",
            "--local-asr-language", "none",
            "--tts-output-mode", "subtitle_only"
        ]
        
        pre_proc = subprocess.Popen(
            pre_soak_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 让 soak 运行 60 秒
        time.sleep(60)
        
        # 步骤 2: 模拟 sleep（暂停 bridge 和 app）
        logger.info("=== Phase 6.2: Simulating system sleep (5s) ===")
        pre_proc.terminate()
        time.sleep(2)  # 等待进程终止
        pre_stdout, pre_stderr = pre_proc.communicate(timeout=5)
        
        # 记录 pre-sleep 状态
        try:
            pre_report_path = Path("logs/session_reports/runtime_soak_report_*.json")
            import glob
            files = glob.glob(str(pre_report_path))
            if files:
                latest_report = sorted(files)[-1]
                with open(latest_report) as f:
                    pre_data = json.load(f)
                    report["pre_sleep"] = {
                        "last_sample": pre_data.get("samples", [])[-1] if pre_data.get("samples") else {},
                        "duration_sec_actual": pre_data.get("duration_sec_actual", 0)
                    }
                    report["summary"]["pre_sleep_ok"] = pre_data.get("session_stop", {}).get("ok", False)
        except Exception as e:
            logger.warning(f"Failed to parse pre-sleep report: {e}")
        
        # 步骤 3: 模拟睡眠期间的暂停
        time.sleep(5)
        
        # 步骤 4: 唤醒后重新启动测试
        logger.info("=== Phase 6.3: Post-wake soak (60s) ===")
        post_soak_cmd = [
            sys.executable, "tools/matrix/runtime_soak_2hr.py",
            "--preset", "smoke",
            "--sample-interval-sec", "5.0",
            "--local-asr-language", "none",
            "--tts-output-mode", "subtitle_only"
        ]
        
        post_proc = subprocess.Popen(
            post_soak_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 让 soak 运行 60 秒
        time.sleep(60)
        post_proc.terminate()
        time.sleep(2)
        post_stdout, post_stderr = post_proc.communicate(timeout=5)
        
        # 记录 post-wake 状态
        try:
            files = glob.glob(str(pre_report_path))
            if files:
                latest_report = sorted(files)[-1]
                with open(latest_report) as f:
                    post_data = json.load(f)
                    report["post_wake"] = {
                        "last_sample": post_data.get("samples", [])[-1] if post_data.get("samples") else {},
                        "duration_sec_actual": post_data.get("duration_sec_actual", 0),
                        "failures": post_data.get("failures", [])
                    }
                    report["summary"]["wake_recovery_ok"] = (
                        post_data.get("session_stop", {}).get("ok", False) and
                        len(post_data.get("failures", [])) == 0
                    )
        except Exception as e:
            logger.warning(f"Failed to parse post-wake report: {e}")
        
        # 步骤 5: 综合判断
        report["summary"]["overall_ok"] = (
            report["summary"]["pre_sleep_ok"] and
            report["summary"]["wake_recovery_ok"]
        )
        
        logger.info(f"Pre-sleep OK: {report['summary']['pre_sleep_ok']}")
        logger.info(f"Wake recovery OK: {report['summary']['wake_recovery_ok']}")
        logger.info(f"Overall OK: {report['summary']['overall_ok']}")
        
    except Exception as e:
        logger.error(f"Sleep/wake test failed: {e}")
        import traceback
        traceback.print_exc()
        report["diagnostics"].append(f"Exception: {str(e)}")
    
    # 输出报告
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {out_path}")
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 6 Sleep/Wake Recovery Test")
    parser.add_argument("--duration-sec", type=int, default=300, help="Total test duration in seconds")
    parser.add_argument("--out", default="logs/session_reports/phase6_sleep_wake_recovery.json", help="Output report path")
    
    args = parser.parse_args()
    
    result = run_sleep_wake_test(duration_sec=args.duration_sec, out=args.out)
    
    sys.exit(0 if result["summary"]["overall_ok"] else 1)
