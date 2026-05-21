import sounddevice as sd

print("可用輸出裝置列表：")
for i, d in enumerate(sd.query_devices()):
    if d['max_output_channels'] > 0:
        print(f"{i}: {d['name']}")
