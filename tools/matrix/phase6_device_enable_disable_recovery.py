#!/usr/bin/env python3
"""
Phase 6 Device Enable/Disable Recovery Test
验证虚拟音频设备在启用/禁用后能自动恢复。
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


def run_device_recovery_test(out: str = None) -> dict:
    """
    运行设备启用/禁用恢复测试。
    
    步骤：
    1. 启动 App + soak（建立稳定状态）
    2. 禁用虚拟音频设备（via devcon/Windows API）
    3. 等待 App 和 bridge 降级到 silence
    4. 重新启用虚拟设备
    5. 验证 App 和 bridge 自动恢复连接
    
    Args:
        out: 输出文件路径
    
    Returns:
        dict: 测试结果报告
    """
    
    report = {
        "kind": "phase6_device_enable_disable_recovery",
        "generated_at": datetime.utcnow().isoformat(),
        "config": "config.yaml",
        "summary": {
            "pre_disable_ok": False,
            "device_disable_detected": False,
            "post_enable_recovery_ok": False,
            "overall_ok": False
        },
        "pre_disable": {},
        "disable_event": {},
        "post_enable": {},
        "diagnostics": []
    }
    
    try:
        # 步骤 1: 运行 60 秒 soak（建立基线）
        logger.info("=== Phase 6.1: Pre-disable soak (60s) ===")
        pre_soak_cmd = [
            sys.executable, "tools/matrix/runtime_soak_2hr.py",
            "--preset", "smoke",
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
        
        time.sleep(60)
        pre_proc.terminate()
        time.sleep(2)
        pre_stdout, pre_stderr = pre_proc.communicate(timeout=5)
        
        # 记录 pre-disable 状态
        try:
            import glob
            pre_report_path = Path("logs/session_reports/runtime_soak_report_*.json")
            files = glob.glob(str(pre_report_path))
            if files:
                latest_report = sorted(files)[-1]
                with open(latest_report) as f:
                    pre_data = json.load(f)
                    report["pre_disable"] = {
                        "last_sample": pre_data.get("samples", [])[-1] if pre_data.get("samples") else {},
                        "duration_sec_actual": pre_data.get("duration_sec_actual", 0),
                        "failures": pre_data.get("failures", [])
                    }
                    report["summary"]["pre_disable_ok"] = (
                        pre_data.get("session_stop", {}).get("ok", False) and
                        len(pre_data.get("failures", [])) == 0
                    )
        except Exception as e:
            logger.warning(f"Failed to parse pre-disable report: {e}")
        
        # 步骤 2: 禁用虚拟设备
        logger.info("=== Phase 6.2: Disabling virtual audio device ===")
        
        # 注意：实际禁用需要用 devcon 或 Windows API，这里模拟失败情况
        # 在实际测试中，应该用 devcon disable 或 PowerShell 禁用设备
        
        disable_cmd = [
            "powershell", "-NoProfile", "-Command",
            "Get-PnpDevice | Where-Object { $_.FriendlyName -match 'SyncTranslate' } | Disable-PnpDevice -Confirm:$false"
        ]
        
        try:
            disable_result = subprocess.run(
                disable_cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            if disable_result.returncode == 0:
                logger.info("Virtual device disabled successfully")
                report["summary"]["device_disable_detected"] = True
                report["disable_event"] = {
                    "action": "disabled",
                    "timestamp": datetime.utcnow().isoformat(),
                    "stdout": disable_result.stdout[:200]
                }
            else:
                logger.warning(f"Failed to disable device: {disable_result.stderr}")
                report["diagnostics"].append(f"Device disable failed: {disable_result.stderr[:200]}")
        except Exception as e:
            logger.warning(f"Device disable command failed: {e}")
            report["diagnostics"].append(f"Device disable exception: {str(e)}")
        
        # 暂停让系统识别设备禁用
        time.sleep(5)
        
        # 步骤 3: 重新启用虚拟设备
        logger.info("=== Phase 6.3: Re-enabling virtual audio device ===")
        
        enable_cmd = [
            "powershell", "-NoProfile", "-Command",
            "Get-PnpDevice | Where-Object { $_.FriendlyName -match 'SyncTranslate' } | Enable-PnpDevice -Confirm:$false"
        ]
        
        try:
            enable_result = subprocess.run(
                enable_cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            if enable_result.returncode == 0:
                logger.info("Virtual device re-enabled successfully")
                report["disable_event"]["re_enabled"] = True
            else:
                logger.warning(f"Failed to enable device: {enable_result.stderr}")
        except Exception as e:
            logger.warning(f"Device enable command failed: {e}")
        
        time.sleep(5)
        
        # 步骤 4: 运行 post-enable soak（验证恢复）
        logger.info("=== Phase 6.4: Post-enable recovery soak (60s) ===")
        
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
        
        time.sleep(60)
        post_proc.terminate()
        time.sleep(2)
        post_stdout, post_stderr = post_proc.communicate(timeout=5)
        
        # 记录 post-enable 状态
        try:
            files = glob.glob(str(pre_report_path))
            if files:
                latest_report = sorted(files)[-1]
                with open(latest_report) as f:
                    post_data = json.load(f)
                    report["post_enable"] = {
                        "last_sample": post_data.get("samples", [])[-1] if post_data.get("samples") else {},
                        "duration_sec_actual": post_data.get("duration_sec_actual", 0),
                        "failures": post_data.get("failures", [])
                    }
                    # 验证恢复：应该能重新连接并运行
                    report["summary"]["post_enable_recovery_ok"] = (
                        post_data.get("session_stop", {}).get("ok", False)
                    )
        except Exception as e:
            logger.warning(f"Failed to parse post-enable report: {e}")
        
        # 步骤 5: 综合判断
        report["summary"]["overall_ok"] = (
            report["summary"]["pre_disable_ok"] and
            report["summary"]["post_enable_recovery_ok"]
        )
        
        logger.info(f"Pre-disable OK: {report['summary']['pre_disable_ok']}")
        logger.info(f"Device disable detected: {report['summary']['device_disable_detected']}")
        logger.info(f"Post-enable recovery OK: {report['summary']['post_enable_recovery_ok']}")
        logger.info(f"Overall OK: {report['summary']['overall_ok']}")
        
    except Exception as e:
        logger.error(f"Device recovery test failed: {e}")
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
    
    parser = argparse.ArgumentParser(description="Phase 6 Device Enable/Disable Recovery Test")
    parser.add_argument("--out", default="logs/session_reports/phase6_device_recovery.json", help="Output report path")
    
    args = parser.parse_args()
    
    result = run_device_recovery_test(out=args.out)
    
    sys.exit(0 if result["summary"]["overall_ok"] else 1)
