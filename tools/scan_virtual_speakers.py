import sounddevice as sd
import soundfile as sf
import re

AUDIO_FILE = r"downloads/asr_regression_en_pool/audio/en_read_001.wav"

# 選出所有 SyncTranslate 虛擬喇叭相關的輸出裝置
pattern = re.compile(r"synctranslate|virtual", re.IGNORECASE)

print("=== 掃描所有 SyncTranslate 虛擬喇叭裝置並嘗試播放 ===")
devices = sd.query_devices()
found = False
for idx, dev in enumerate(devices):
    if dev['max_output_channels'] > 0 and pattern.search(dev['name']):
        print(f"嘗試 index={idx}: {dev['name']}")
        try:
            data, samplerate = sf.read(AUDIO_FILE, dtype='float32')
            sd.play(data, samplerate, device=idx)
            sd.wait()
            print(f"[成功] index={idx}: {dev['name']} 可正常播放！")
            found = True
        except Exception as e:
            print(f"[失敗] index={idx}: {dev['name']} -> {e}")

if not found:
    print("沒有任何 SyncTranslate 虛擬喇叭裝置可被 sounddevice 成功播放。")
else:
    print("測試結束。請確認有聲音輸出或 ASR 有反應。")
