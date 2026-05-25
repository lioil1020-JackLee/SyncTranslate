param(
    [string]$PackageDir = "artifacts/driver/synctranslate_virtual_audio/package",
    [string]$CertificatePath = "artifacts/driver/synctranslate_virtual_audio/SyncTranslateVirtualAudioTest.cer",
    [string]$OutputMsi = "artifacts/driver/synctranslate_virtual_audio/SyncTranslateVirtualAudioDriver.msi",
    [string]$ProductVersion = "2.1.7",
    [string]$WixToolPath = "artifacts/tools/wix5",
    [string]$WixVersion = "5.0.2"
)

$ErrorActionPreference = "Stop"

$dotnetRootCandidates = @(
    "C:\dotnet",
    (Join-Path $env:ProgramFiles "dotnet"),
    (Join-Path ${env:ProgramFiles(x86)} "dotnet")
)
$dotnetRoot = $dotnetRootCandidates | Where-Object { $_ -and (Test-Path (Join-Path $_ "dotnet.exe")) } | Select-Object -First 1
if ($dotnetRoot) {
    $env:DOTNET_ROOT = (Resolve-Path $dotnetRoot).Path
    if ($env:PATH -notlike "*$env:DOTNET_ROOT*") {
        $env:PATH = "$env:DOTNET_ROOT;$env:PATH"
    }
}
$dotnetTools = Join-Path $env:USERPROFILE ".dotnet\tools"
if ((Test-Path $dotnetTools) -and $env:PATH -notlike "*$dotnetTools*") {
    $env:PATH = "$dotnetTools;$env:PATH"
}

if (!(Test-Path $PackageDir)) {
    throw "Missing signed driver package directory: $PackageDir. Build and sign the driver package first."
}

$infFiles = Get-ChildItem -Path $PackageDir -Filter "*.inf" -File
if (!$infFiles) {
    throw "No INF files found in driver package directory: $PackageDir"
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$driverRoot = Split-Path -Parent $scriptRoot
$wxs = Join-Path $driverRoot "installer/Package.wxs"
if (!(Test-Path $wxs)) {
    throw "Missing WiX source: $wxs"
}

$stagingDir = Join-Path (Split-Path -Parent $OutputMsi) "msi_payload"
if (Test-Path $stagingDir) {
    Remove-Item -LiteralPath $stagingDir -Recurse -Force
}
New-Item -ItemType Directory -Path $stagingDir -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stagingDir "package") -Force | Out-Null

Copy-Item -Path (Join-Path $PackageDir "*") -Destination (Join-Path $stagingDir "package") -Recurse -Force
Copy-Item -Path (Join-Path $scriptRoot "install_driver_package.ps1") -Destination $stagingDir -Force
Copy-Item -Path (Join-Path $scriptRoot "uninstall_driver_package.ps1") -Destination $stagingDir -Force
Copy-Item -Path (Join-Path $scriptRoot "preflight_driver_install.ps1") -Destination $stagingDir -Force
Copy-Item -Path (Join-Path $scriptRoot "verify_driver_install.ps1") -Destination $stagingDir -Force
Copy-Item -Path (Join-Path $driverRoot "README.md") -Destination $stagingDir -Force
Copy-Item -Path (Join-Path $driverRoot "driver_contract.md") -Destination $stagingDir -Force

# Copy create_root_device.ps1 for the devcon-free MSI path. The MSI one-click
# action runs install_driver_package.ps1; that script uses this SetupAPI helper
# when devcon.exe is unavailable, then binds the staged driver with pnputil.
$createRootDeviceScript = Join-Path $scriptRoot "create_root_device.ps1"
if (Test-Path $createRootDeviceScript) {
    Copy-Item -Path $createRootDeviceScript -Destination (Join-Path $stagingDir "create_root_device.ps1") -Force
} else {
    Write-Warning "create_root_device.ps1 not found at $createRootDeviceScript. Manual fallback root-device creation will not be included."
}

# Copy devcon.exe if available from WDK (optional, no longer required by MSI)
$devconCandidates = @(
    "${env:ProgramFiles(x86)}\Windows Kits\10\Tools\10.0.26100.0\x64\devcon.exe",
    "${env:ProgramFiles}\Windows Kits\10\Tools\10.0.26100.0\x64\devcon.exe"
)
$devcon = $devconCandidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if ($devcon) {
    Copy-Item -Path $devcon -Destination (Join-Path $stagingDir "devcon.exe") -Force
}
if (Test-Path $CertificatePath) {
    Copy-Item -Path $CertificatePath -Destination (Join-Path $stagingDir "SyncTranslateVirtualAudioTest.cer") -Force
}
else {
    Write-Warning "Certificate not found: $CertificatePath. MSI will still be built, but install may fail until a trusted certificate is provided."
    New-Item -ItemType File -Path (Join-Path $stagingDir "SyncTranslateVirtualAudioTest.cer") -Force | Out-Null
}

$repoWix = Join-Path $WixToolPath "wix.exe"
if (Test-Path $repoWix) {
    $wixExe = (Resolve-Path $repoWix).Path
}
else {
    $pathWix = Get-Command wix.exe -ErrorAction SilentlyContinue
    if ($pathWix) {
        $wixExe = $pathWix.Source
    }
    else {
        $wixCandidates = @(
            (Join-Path $env:USERPROFILE ".dotnet\tools\wix.exe"),
            (Join-Path $env:USERPROFILE ".dotnet\tools\.store\wix\$WixVersion\wix\$WixVersion\tools\net8.0\any\wix.exe")
        )
        $existingWix = $wixCandidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
        if ($existingWix) {
            $wixExe = (Resolve-Path $existingWix).Path
        }
        else {
            if (!(Test-Path $WixToolPath)) {
                New-Item -ItemType Directory -Path $WixToolPath -Force | Out-Null
            }
            $wixExe = Join-Path $WixToolPath "wix.exe"
        }
    }
}

if (!(Test-Path $wixExe)) {
    $dotnet = Get-Command dotnet -ErrorAction SilentlyContinue
    if (!$dotnet) {
        $dotnetCandidates = @(
            "C:\dotnet\dotnet.exe",
            (Join-Path $env:ProgramFiles "dotnet\dotnet.exe"),
            (Join-Path ${env:ProgramFiles(x86)} "dotnet\dotnet.exe")
        )
        $existingDotnet = $dotnetCandidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
        if ($existingDotnet) {
            $dotnet = [pscustomobject]@{ Source = (Resolve-Path $existingDotnet).Path }
        }
    }
    if (!$dotnet) {
        throw "WiX is required to build the MSI. Install wix.exe or install the .NET SDK so this script can run 'dotnet tool install wix'."
    }
    $sdks = & $dotnet.Source --list-sdks 2>$null
    if (!$sdks -or $sdks.Count -eq 0) {
        throw "WiX is required to build the MSI. This machine has dotnet runtime but no .NET SDK. Install the .NET SDK or install wix.exe and add it to PATH."
    }
    Write-Host "[driver-msi] installing WiX command-line tool $WixVersion into $WixToolPath"
    & $dotnet.Source tool install wix --version $WixVersion --tool-path $WixToolPath
    if ($LASTEXITCODE -ne 0) {
        throw "dotnet tool install wix failed with exit code $LASTEXITCODE"
    }
}

$outputParent = Split-Path -Parent $OutputMsi
if ($outputParent) {
    New-Item -ItemType Directory -Path $outputParent -Force | Out-Null
}
if (Test-Path $OutputMsi) {
    Remove-Item -LiteralPath $OutputMsi -Force
}

$payloadFull = (Resolve-Path $stagingDir).Path
$wxsFull = (Resolve-Path $wxs).Path
$outputFull = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputMsi)

Write-Host "[driver-msi] building MSI"
& $wixExe build `
    $wxsFull `
    -arch x64 `
    -d "PayloadDir=$payloadFull" `
    -d "ProductVersion=$ProductVersion" `
    -out $outputFull
if ($LASTEXITCODE -ne 0) {
    throw "wix build failed with exit code $LASTEXITCODE"
}

Get-Item $OutputMsi | Select-Object FullName,Length | Format-Table -AutoSize
