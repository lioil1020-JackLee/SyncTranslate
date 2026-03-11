# VoiceMeeter Banana 設定說明
## SyncTranslate 個人雙向口譯機

## 1. 目的

這份文件只處理一件事：

把 `VoiceMeeter Banana` 設定成適合本專案的 `Banana-only` 音訊拓樸。

這份設定完成後，系統會符合以下目標：

1. 視訊會議軟體的對方聲音可以被翻譯程式抓到。
2. 翻譯程式產生的英文 TTS 可以送回視訊會議軟體。
3. 中文譯音只播放到你的耳機，不回到會議。
4. 不同音訊路徑彼此隔離，避免回授循環。

---

## 2. 先講結論

你目前不需要額外硬體。

也不需要真的把電腦音源分接成兩條實體線。

目前最簡單可行的方案是：

1. 用 `VoiceMeeter Banana` 內建的 `VoiceMeeter Input / Output` 與 `VoiceMeeter AUX Input / Aux Output`。
2. 用 `A1` 接你的實體耳機。
3. 讓翻譯程式直接抓實體麥克風。
4. 讓中文譯音直接播到耳機，不先經過 VoiceMeeter。

> 注意（對應目前程式行為）：
> - `載入 Banana 預設` 只會自動填入 `remote_in` 與 `meeting_tts_out`。
> - `local_mic_in`、`local_tts_out` 仍需在 `音訊路由` 分頁手動確認。
> - 若 `config.yaml` 的四個音訊欄位是空白，請先完成路由設定再按 `開始`。

---

## 3. 本方案的音訊路由圖

### 3.1 對方講話路徑

`會議軟體 Speaker -> VoiceMeeter Input -> B2 -> 翻譯程式 remote_in -> 英文 ASR -> 中文翻譯 -> 中文 TTS -> 實體耳機`

### 3.2 你講話路徑

`實體麥克風 -> 翻譯程式 local_mic_in -> 中文 ASR -> 英文翻譯 -> 英文 TTS -> VoiceMeeter AUX Input -> B1 -> 會議軟體 Microphone`

### 3.3 這個拓樸的重點

1. `B2` 只負責把會議軟體的對方聲音送給翻譯程式。
2. `B1` 只負責把翻譯程式的英文 TTS 送給會議軟體。
3. `A1` 只負責你的耳機監聽。
4. 你不需要聽對方原音，所以 `VoiceMeeter Input` strip 不要送到 `A1`。

---

## 4. 需要用到的裝置名稱

Windows 裡通常會看到類似以下名稱：

| 類型 | 常見名稱 | 用途 |
| --- | --- | --- |
| 虛擬播放裝置 | `VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)` | 會議軟體 Speaker |
| 虛擬錄音裝置 | `VoiceMeeter Output (VB-Audio VoiceMeeter VAIO)` | 會議軟體 Microphone |
| 虛擬播放裝置 | `VoiceMeeter AUX Input (VB-Audio VoiceMeeter AUX VAIO)` | 翻譯程式英文 TTS 的輸出目標 |
| 虛擬錄音裝置 | `VoiceMeeter Aux Output (VB-Audio VoiceMeeter AUX VAIO)` | 翻譯程式 `remote_in` |
| 硬體輸出 | 你的耳機 / 耳麥耳機 | `A1` |
| 硬體輸入 | 你的實體麥克風 | `local_mic_in` |

不同電腦上的名稱可能有些微差異，但關鍵字通常是：

1. `VoiceMeeter Input`
2. `VoiceMeeter Output`
3. `VoiceMeeter AUX Input`
4. `VoiceMeeter Aux Output`

---

## 5. VoiceMeeter Banana 具體設定步驟

### 5.1 A1 設定

1. 打開 VoiceMeeter Banana。
2. 在右上角 `A1` 下拉選單選你的耳機。
3. 優先選 `WDM` 或 `KS`，若不穩再改 `MME`。
4. 不建議先用外放喇叭，因為中文譯音很容易被麥克風收回去。

### 5.2 A2 / A3 設定

1. 第一版先留空。
2. 若你未來要額外監聽其他聲音，再啟用。

### 5.3 Hardware Input strips

Banana 左邊有 3 條 Hardware Input strips。

這版建議：

1. 不把實體麥克風先進 Banana。
2. 實體麥克風由翻譯程式直接抓。
3. 所以 `Hardware Input 1/2/3` 都可以先空著。

如果你暫時在 `Hardware Input 1` 放了麥克風，也請先把這條 strip 的：

1. `A1`
2. `A2`
3. `A3`
4. `B1`
5. `B2`

全部關掉，避免誤送。

### 5.4 Virtual Input: VoiceMeeter Input

這條是本方案最重要的第一條虛擬路徑。

用途：

1. 讓會議軟體把對方聲音送進來。
2. 再由 Banana 轉送到 `B2`。
3. 讓翻譯程式從 `VoiceMeeter Aux Output` 抓到。

這條 strip 的設定：

1. `B2` 打開。
2. `A1` 關閉。
3. `A2` 關閉。
4. `A3` 關閉。
5. `B1` 關閉。

這樣設定的原因：

1. 你不需要直接聽對方原音。
2. 對方聲音只該進翻譯程式。
3. 不要讓它誤送回會議。

### 5.5 Virtual Input: VoiceMeeter AUX Input

這條是本方案最重要的第二條虛擬路徑。

用途：

1. 翻譯程式把英文 TTS 輸出到 `VoiceMeeter AUX Input`。
2. Banana 再把這條聲音送到 `B1`。
3. 視訊會議軟體從 `VoiceMeeter Output` 把它當成麥克風收到。

這條 strip 的設定：

1. `B1` 打開。
2. `A1` 關閉。
3. `A2` 關閉。
4. `A3` 關閉。
5. `B2` 關閉。

這樣設定的原因：

1. 英文 TTS 只該送回會議。
2. 不應回送到翻譯程式的 `remote_in`。
3. 也不一定需要播給你自己聽。

### 5.6 Master 區不用特別改的地方

第一版先保持預設即可：

1. `Normal mode`
2. 不開 `mono`
3. 不開 `EQ`
4. 不額外開壓縮或效果

因為第一階段重點是路由正確，不是音色美化。

---

## 6. 視訊會議軟體要怎麼設

以 Google Meet、Teams、Zoom 為例，都只要處理兩個欄位：

| 欄位 | 裝置 |
| --- | --- |
| Speaker | `VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)` |
| Microphone | `VoiceMeeter Output (VB-Audio VoiceMeeter VAIO)` |

請注意：

1. Speaker 不是設耳機。
2. Microphone 不是設你的實體麥克風。
3. 讓會議軟體只看到 Banana 的虛擬端點。

---

## 7. 翻譯程式要怎麼設

翻譯程式建議使用以下映射：

| 邏輯通道 | 實際裝置 |
| --- | --- |
| `remote_in` | `VoiceMeeter Aux Output (VB-Audio VoiceMeeter AUX VAIO)` |
| `local_mic_in` | 你的實體麥克風 |
| `local_tts_out` | 你的實體耳機 |
| `meeting_tts_out` | `VoiceMeeter AUX Input (VB-Audio VoiceMeeter AUX VAIO)` |

重點：

1. `remote_in` 是錄音裝置。
2. `meeting_tts_out` 是播放裝置。
3. `local_tts_out` 直接指向耳機，不經 Banana。
4. `載入 Banana 預設` 不會覆蓋 `local_mic_in`、`local_tts_out`。

---

## 8. 第一次 bring-up 的實際操作順序

建議照這個順序做，不要跳步：

### Step 1

先確認 Banana 能正常開啟，`A1` 能選到耳機。

### Step 2

設定 `VoiceMeeter Input` strip：

1. 只開 `B2`
2. 其他全關

### Step 3

設定 `VoiceMeeter AUX Input` strip：

1. 只開 `B1`
2. 其他全關

### Step 4

在會議軟體裡設定：

1. Speaker = `VoiceMeeter Input`
2. Microphone = `VoiceMeeter Output`

### Step 5

先開翻譯程式到 `音訊診斷` 分頁，測 `remote_in` 擷取：

1. remote 裝置選 `VoiceMeeter Aux Output`
2. 讓會議中有人講話或播放測試音
3. 按 `開始 remote 擷取`，確認音量條有跳動
4. 測完按 `停止 remote 擷取`

### Step 6

在同一頁測回送路徑：

1. 先確認 `meeting_tts_out = VoiceMeeter AUX Input`
2. 按 `測試英文送出`
3. 在會議軟體查看 Mic test meter
4. 確認麥克風電平有跳動

### Step 7

最後才接上你的翻譯程式：

1. `remote_in`
2. `local_mic_in`
3. `local_tts_out`
4. `meeting_tts_out`

全部配好再做雙向測試。
建議先用 `只收聽翻譯` 驗證單向，再切到 `雙向模式`。

---

## 9. 檢查清單

### 9.1 正常狀態應該看到什麼

| 項目 | 正常狀態 |
| --- | --- |
| `VoiceMeeter Input` strip | 有對方會議音時會跳表 |
| `B2` | 有對方音時會有電平 |
| `VoiceMeeter AUX Input` strip | 只有英文 TTS 輸入時會跳表 |
| `B1` | 英文 TTS 送出時會有電平 |
| A1 耳機 | 當 `local_tts_out` 指到耳機時，會聽到中文譯音；若 `VoiceMeeter Input` 未開 `A1`，不會直接聽到對方原音 |

### 9.2 若你聽到對方原音

表示以下其中一個地方錯了：

1. `VoiceMeeter Input` strip 開了 `A1`
2. Windows 或會議軟體還直接把聲音送到耳機
3. 會議軟體 Speaker 沒有設到 `VoiceMeeter Input`

### 9.3 若對方聽不到英文 TTS

檢查順序：

1. 翻譯程式 `meeting_tts_out` 是否設到 `VoiceMeeter AUX Input`
2. `VoiceMeeter AUX Input` strip 是否有開 `B1`
3. `VoiceMeeter AUX Input` strip 是否真的有收到音
4. 會議軟體 Microphone 是否設到 `VoiceMeeter Output`

### 9.4 若翻譯程式抓不到對方聲音

檢查順序：

1. 會議軟體 Speaker 是否設到 `VoiceMeeter Input`
2. `VoiceMeeter Input` strip 是否有開 `B2`
3. 翻譯程式 `remote_in` 是否設到 `VoiceMeeter Aux Output`
4. 是否誤選成 `VoiceMeeter Output`

### 9.5 若出現回授或鬼叫

最常見原因：

1. `VoiceMeeter AUX Input` 開了 `B2`
2. `VoiceMeeter Input` 開了 `B1`
3. 中文 TTS 被送回 Banana 的送話路徑
4. 你用外放喇叭，中文譯音被實體麥克風收到

第一個要做的事：

1. 先全部停止播放
2. 檢查兩條 Virtual Input strip 的按鈕狀態
3. 確認 `VoiceMeeter Input = B2 only`
4. 確認 `VoiceMeeter AUX Input = B1 only`

---

## 10. 建議的 Windows 使用習慣

1. 會議時盡量用耳機，不要用喇叭。
2. Banana 設定固定後，不要讓 Windows 自動切換預設裝置。
3. 同一場會議中，不要隨意切 Speaker / Microphone。
4. 若會議軟體更新後把裝置重設，先檢查會議設定，不要先懷疑程式壞掉。

---

## 11. 何時再考慮加 VB-CABLE

第一版不需要。

之後若你想做以下事情，再考慮：

1. 中文 TTS 也想先進 Banana 控制。
2. 想保留對方原音做低音量監聽。
3. 想把系統音、會議音、譯音完全拆開。
4. 想建立更多預設拓樸。

---

## 12. 本文件對應的最終標準

這份設定完成後，應該達成以下狀態：

1. 會議軟體能把對方音訊送進 Banana。
2. Banana 能把對方音訊轉到 `B2`。
3. 翻譯程式能從 `VoiceMeeter Aux Output` 抓到 `remote_in`。
4. 翻譯程式能把英文 TTS 打到 `VoiceMeeter AUX Input`。
5. Banana 能把這段英文 TTS 轉到 `B1`。
6. 會議軟體能從 `VoiceMeeter Output` 收到這段英文 TTS。
7. 你自己只從耳機聽到中文譯音。
