$ErrorActionPreference = "Stop"

# =============================
# SyncTranslate 半自動安裝腳本
# =============================

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$Root = "C:\Tools\SyncTranslate"
$DownloadDir = Join-Path $Root "Downloads"
$AppDir = Join-Path $Root "SyncTranslate"
$TempDir = Join-Path $Root "Temp"
$VoiceMeeterExtractDir = Join-Path $TempDir "VoiceMeeter"

# 維護者如需啟用自動下載，可在這裡填入正式網址。
# 一般使用者不需要修改腳本，只要把安裝檔放在本腳本同一資料夾即可。
$LmStudioUrl = ""
$VoiceMeeterUrl = ""
$SyncTranslateUrl = ""

$LmStudioInstaller = Join-Path $DownloadDir "LM-Studio-Setup.exe"
$VoiceMeeterInstaller = Join-Path $DownloadDir "VoiceMeeterSetup.zip"
$SyncTranslateZip = Join-Path $DownloadDir "SyncTranslate-onedir.zip"

function Write-Step($message) {
    Write-Host ""
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host $message -ForegroundColor Cyan
    Write-Host "==================================================" -ForegroundColor Cyan
}

function Ensure-Directory($path) {
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Force -Path $path | Out-Null
    }
}

function Resolve-Package($label, $expectedNames, $configuredUrl, $downloadTarget) {
    $localCandidates = @()
    foreach ($name in $expectedNames) {
        $localCandidates += (Join-Path $ScriptDir $name)
    }
    $localCandidates += $downloadTarget

    foreach ($candidate in $localCandidates) {
        if (Test-Path $candidate) {
            if ($candidate -ne $downloadTarget) {
                Copy-Item -Path $candidate -Destination $downloadTarget -Force
                Write-Host "已使用同資料夾中的 $label：$candidate" -ForegroundColor Green
            }
            else {
                Write-Host "已找到現成的 $label：$candidate" -ForegroundColor DarkYellow
            }

            return $downloadTarget
        }
    }

    if ([string]::IsNullOrWhiteSpace($configuredUrl)) {
        Write-Host "找不到 $label。請把檔案放在腳本同一資料夾，檔名可為：$($expectedNames -join ' / ')" -ForegroundColor Yellow
        return $null
    }

    Write-Host "下載 $label 中：$configuredUrl"
    Invoke-WebRequest -Uri $configuredUrl -OutFile $downloadTarget
    Write-Host "下載完成：$downloadTarget" -ForegroundColor Green
    return $downloadTarget
}

function Start-InstallerIfExists($label, $path) {
    if ($path -and (Test-Path $path)) {
        Start-Process -FilePath $path -Verb RunAs
    }
    else {
        Write-Host "找不到 $label 安裝檔，請確認檔案是否已放到腳本同一資料夾。" -ForegroundColor Yellow
    }
}

function Start-VoiceMeeterInstaller($packagePath) {
    if (-not $packagePath -or -not (Test-Path $packagePath)) {
        Write-Host "找不到 VoiceMeeter 安裝檔，請確認檔案是否已放到腳本同一資料夾。" -ForegroundColor Yellow
        return
    }

    $extension = [System.IO.Path]::GetExtension($packagePath).ToLowerInvariant()
    if ($extension -eq ".exe") {
        Start-Process -FilePath $packagePath -Verb RunAs
        return
    }

    if ($extension -ne ".zip") {
        Write-Host "VoiceMeeter 安裝檔格式不支援：$packagePath" -ForegroundColor Yellow
        return
    }

    if (Test-Path $VoiceMeeterExtractDir) {
        Remove-Item $VoiceMeeterExtractDir -Recurse -Force
    }

    Expand-Archive -Path $packagePath -DestinationPath $VoiceMeeterExtractDir -Force
    $setupExe = Get-ChildItem -Path $VoiceMeeterExtractDir -Recurse -Filter "*Setup*.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($setupExe) {
        Start-Process -FilePath $setupExe.FullName -Verb RunAs
    }
    else {
        Write-Host "已解壓 VoiceMeeter 套件，但找不到安裝程式。" -ForegroundColor Yellow
    }
}

Write-Step "建立安裝資料夾"
Ensure-Directory $Root
Ensure-Directory $DownloadDir
Ensure-Directory $AppDir
Ensure-Directory $TempDir

Write-Step "準備安裝檔"
$ResolvedLmStudioInstaller = Resolve-Package "LM Studio 安裝程式" @("LM-Studio-Setup.exe") $LmStudioUrl $LmStudioInstaller
$ResolvedVoiceMeeterInstaller = Resolve-Package "VoiceMeeter 安裝程式" @("VoiceMeeterSetup.exe", "VoiceMeeterSetup.zip", "VoicemeeterSetup_v2122.zip") $VoiceMeeterUrl $VoiceMeeterInstaller
$ResolvedSyncTranslateZip = Resolve-Package "SyncTranslate onedir 壓縮檔" @("SyncTranslate-onedir.zip", "SyncTranslate-onedir-windows.zip") $SyncTranslateUrl $SyncTranslateZip

Write-Step "啟動 LM Studio 安裝程式"
Start-InstallerIfExists "LM Studio" $ResolvedLmStudioInstaller

Write-Step "啟動 VoiceMeeter 安裝程式"
Start-VoiceMeeterInstaller $ResolvedVoiceMeeterInstaller

Write-Step "解壓 SyncTranslate onedir"
if ($ResolvedSyncTranslateZip -and (Test-Path $ResolvedSyncTranslateZip)) {
    try {
        Expand-Archive -Path $ResolvedSyncTranslateZip -DestinationPath $AppDir -Force
        Write-Host "SyncTranslate 解壓完成：$AppDir" -ForegroundColor Green
    }
    catch {
        Write-Host "解壓失敗，請確認壓縮檔格式是否為 zip。" -ForegroundColor Red
        throw
    }
}
else {
    Write-Host "找不到 SyncTranslate 壓縮檔，請確認檔案是否已放到腳本同一資料夾。" -ForegroundColor Yellow
}

Write-Step "嘗試開啟 LM Studio"
$LmStudioExeCandidates = @(
    "$env:LOCALAPPDATA\Programs\LM Studio\LM Studio.exe",
    "$env:ProgramFiles\LM Studio\LM Studio.exe"
)

$LmStudioOpened = $false
foreach ($exe in $LmStudioExeCandidates) {
    if (Test-Path $exe) {
        Start-Process $exe
        $LmStudioOpened = $true
        break
    }
}

if (-not $LmStudioOpened) {
    Write-Host "尚未找到 LM Studio.exe。這可能是因為安裝尚未完成。請你手動完成安裝後再打開 LM Studio。" -ForegroundColor Yellow
}

Write-Step "確認 SyncTranslate.exe 位置"
$SyncTranslateExe = Get-ChildItem -Path $AppDir -Recurse -Filter "SyncTranslate.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($SyncTranslateExe) {
    Write-Host "已找到 SyncTranslate.exe：" -ForegroundColor Green
    Write-Host $SyncTranslateExe.FullName
}
else {
    Write-Host "尚未找到 SyncTranslate.exe。請檢查 onedir 壓縮檔內容是否正確。" -ForegroundColor Yellow
}

Write-Step "後續你必須手動完成的事情"
Write-Host "1. 完成 LM Studio 安裝"
Write-Host "2. 在 LM Studio 的 Discover 搜尋：hy-mt1.5-7b"
Write-Host "3. 選 Tencent 版本，量化選 Q4_K_M"
Write-Host "4. 下載並載入模型"
Write-Host "5. 在 LM Studio 啟用 Local Server"
Write-Host "6. 確認 API 位址是 http://127.0.0.1:1234"
Write-Host "7. 完成 VoiceMeeter 安裝，如有要求請重新開機"
Write-Host "8. 打開 SyncTranslate.exe"
Write-Host "9. 在 SyncTranslate 中設定模型與音訊裝置"
Write-Host "10. 執行系統檢查"

Write-Host ""
Write-Host "安裝腳本已執行完畢。" -ForegroundColor Green
Write-Host "請回到說明文件，依序完成後續手動步驟。" -ForegroundColor Green
