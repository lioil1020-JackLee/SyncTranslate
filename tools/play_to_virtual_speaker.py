import sounddevice as sd
import soundfile as sf
import sys

# 預設音檔路徑，可改為其他 en_read_*.wav
AUDIO_FILE = r"downloads/asr_regression_en_pool/audio/en_read_001.wav"

VIRTUAL_SPEAKER_INDEX = 14  # 喇叭 (SyncTranslate Virtual Audio)




def main():
    data, samplerate = sf.read(AUDIO_FILE, dtype='float32')
    print(f"播放 {AUDIO_FILE} 到 index={VIRTUAL_SPEAKER_INDEX} (喇叭 (SyncTranslate Virtual Audio))，取樣率 {samplerate}")
    sd.play(data, samplerate, device=VIRTUAL_SPEAKER_INDEX)
    sd.wait()
    print("播放完成。請檢查 SyncTranslate 遠端 ASR 是否有字幕輸出。")

if __name__ == "__main__":
    main()
