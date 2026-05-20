param(
    [switch]$All,
    [switch]$InstallDotNetSdk,
    [switch]$InstallWix,
    [switch]$InstallWdkBuildTools,
    [switch]$LaunchElevatedWdkInstaller,
    [switch]$InstallSpectreLibraries,
    [switch]$InstallWdk,
    [switch]$PrintWdkInstructions,
    [string]$DotNetVersion = "8.0.416",
    [string]$DotNetDir = "artifacts/tools/dotnet",
    [string]$WixToolPath = "artifacts/tools/wix5",
    [string]$WixVersion = "5.0.2",
    [string]$VsInstallPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools",
    [string]$WdkSetupUrl = "https://go.microsoft.com/fwlink/?linkid=2335869",
    [string]$WdkSetupPath = "artifacts/tools/wdk/wdksetup.exe"
)

$ErrorActionPreference = "Stop"

if ($All) {
    $InstallDotNetSdk = $true
    $InstallWix = $true
    $InstallWdkBuildTools = $true
    $InstallSpectreLibraries = $true
    $InstallWdk = $true
    $PrintWdkInstructions = $true
}

function Get-LocalDotnet {
    param([string]$Root)
    $candidate = Join-Path $Root "dotnet.exe"
    if (Test-Path $candidate) {
        return (Resolve-Path $candidate).Path
    }
    $pathDotnet = Get-Command dotnet.exe -ErrorAction SilentlyContinue
    if ($pathDotnet) {
        $sdks = & $pathDotnet.Source --list-sdks 2>$null
        if ($sdks -and $sdks.Count -gt 0) {
            return $pathDotnet.Source
        }
    }
    return ""
}

function Install-LocalDotnetSdk {
    param(
        [string]$Version,
        [string]$Root
    )
    New-Item -ItemType Directory -Path $Root -Force | Out-Null
    $scriptPath = Join-Path $Root "dotnet-install.ps1"
    if (!(Test-Path $scriptPath)) {
        Write-Host "[prereq] downloading dotnet-install.ps1"
        Invoke-WebRequest `
            -Uri "https://dot.net/v1/dotnet-install.ps1" `
            -OutFile $scriptPath `
            -UseBasicParsing
    }
    Write-Host "[prereq] installing local .NET SDK $Version into $Root"
    & $scriptPath -Version $Version -InstallDir $Root -NoPath
    if ($LASTEXITCODE -ne 0) {
        throw "dotnet-install.ps1 failed with exit code $LASTEXITCODE"
    }
    return (Resolve-Path (Join-Path $Root "dotnet.exe")).Path
}

function Install-LocalWix {
    param(
        [string]$DotnetExe,
        [string]$ToolPath,
        [string]$Version
    )
    New-Item -ItemType Directory -Path $ToolPath -Force | Out-Null
    $wixExe = Join-Path $ToolPath "wix.exe"
    if (Test-Path $wixExe) {
        Write-Host "[prereq] WiX already installed: $wixExe"
        return
    }
    Write-Host "[prereq] installing WiX Toolset $Version into $ToolPath"
    & $DotnetExe tool install wix --version $Version --tool-path $ToolPath
    if ($LASTEXITCODE -ne 0) {
        throw "dotnet tool install wix failed with exit code $LASTEXITCODE"
    }
}

function Show-WdkInstructions {
    param([string]$InstallPath)
    $vsInstaller = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vs_installer.exe"
    Write-Host ""
    Write-Host "[prereq] WDK/inf2cat still requires a system-level Visual Studio/WDK install."
    Write-Host "[prereq] Open an Administrator PowerShell and run Visual Studio Installer, then add:"
    Write-Host "  Component.Microsoft.Windows.DriverKit.BuildTools"
    Write-Host ""
    if (Test-Path $vsInstaller) {
        Write-Host "Example command:"
        Write-Host "`"$vsInstaller`" modify --installPath `"$InstallPath`" --add Component.Microsoft.Windows.DriverKit.BuildTools --passive --norestart"
    }
    else {
        Write-Host "Visual Studio Installer was not found at:"
        Write-Host "  $vsInstaller"
        Write-Host "Install Visual Studio Build Tools 2022 and Windows Driver Kit from Microsoft first."
    }
    Write-Host ""
}

function Test-IsAdmin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Install-WdkBuildTools {
    param(
        [string]$InstallPath,
        [switch]$LaunchElevated
    )
    $vsInstaller = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vs_installer.exe"
    if (!(Test-Path $vsInstaller)) {
        throw "Visual Studio Installer was not found: $vsInstaller"
    }
    if (!(Test-Path $InstallPath)) {
        throw "Visual Studio Build Tools install path was not found: $InstallPath"
    }
    if (!(Test-IsAdmin)) {
        if ($LaunchElevated) {
            $argLine = 'modify --installPath "{0}" --add Component.Microsoft.Windows.DriverKit.BuildTools --passive --norestart' -f $InstallPath
            Write-Host "[prereq] launching elevated Visual Studio Installer"
            Write-Host "`"$vsInstaller`" $argLine"
            Start-Process -FilePath $vsInstaller -ArgumentList $argLine -Verb RunAs -Wait
            return
        }
        Show-WdkInstructions -InstallPath $InstallPath
        throw "Installing WDK DriverKit build tools requires an elevated Administrator PowerShell."
    }
    Write-Host "[prereq] installing WDK DriverKit build tools into $InstallPath"
    & $vsInstaller modify `
        --installPath $InstallPath `
        --add Component.Microsoft.Windows.DriverKit.BuildTools `
        --passive `
        --norestart
    if ($LASTEXITCODE -ne 0) {
        throw "Visual Studio Installer failed with exit code $LASTEXITCODE"
    }
}

function Install-VcSpectreLibraries {
    param(
        [string]$InstallPath,
        [switch]$LaunchElevated
    )
    $vsInstaller = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vs_installer.exe"
    if (!(Test-Path $vsInstaller)) {
        throw "Visual Studio Installer was not found: $vsInstaller"
    }
    if (!(Test-Path $InstallPath)) {
        throw "Visual Studio Build Tools install path was not found: $InstallPath"
    }
    $argLine = 'modify --installPath "{0}" --add Microsoft.VisualStudio.Component.VC.14.44.17.14.ATL --add Microsoft.VisualStudio.Component.VC.14.44.17.14.x86.x64.Spectre --add Microsoft.VisualStudio.Component.VC.14.44.17.14.ATL.Spectre --add Microsoft.VisualStudio.Component.VC.14.44.17.14.MFC.Spectre --passive --norestart' -f $InstallPath
    if (!(Test-IsAdmin)) {
        if ($LaunchElevated) {
            Write-Host "[prereq] launching elevated Visual Studio Installer for VC Spectre libraries"
            Write-Host "`"$vsInstaller`" $argLine"
            Start-Process -FilePath $vsInstaller -ArgumentList $argLine -Verb RunAs -Wait
            return
        }
        throw "Installing VC Spectre libraries requires an elevated Administrator PowerShell. Re-run with -LaunchElevatedWdkInstaller."
    }
    Write-Host "[prereq] installing VC Spectre libraries into $InstallPath"
    & $vsInstaller modify `
        --installPath $InstallPath `
        --add Microsoft.VisualStudio.Component.VC.14.44.17.14.ATL `
        --add Microsoft.VisualStudio.Component.VC.14.44.17.14.x86.x64.Spectre `
        --add Microsoft.VisualStudio.Component.VC.14.44.17.14.ATL.Spectre `
        --add Microsoft.VisualStudio.Component.VC.14.44.17.14.MFC.Spectre `
        --passive `
        --norestart
    if ($LASTEXITCODE -ne 0) {
        throw "Visual Studio Installer failed with exit code $LASTEXITCODE"
    }
}

function Install-Wdk {
    param(
        [string]$Url,
        [string]$SetupPath
    )
    $parent = Split-Path -Parent $SetupPath
    if ($parent) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    if (!(Test-Path $SetupPath)) {
        Write-Host "[prereq] downloading WDK setup from Microsoft"
        Invoke-WebRequest -Uri $Url -OutFile $SetupPath -UseBasicParsing
    }
    if (!(Test-IsAdmin)) {
        Write-Host "[prereq] launching elevated WDK setup"
        Start-Process -FilePath (Resolve-Path $SetupPath).Path -ArgumentList "/quiet /norestart" -Verb RunAs -Wait
        return
    }
    Write-Host "[prereq] running WDK setup"
    & (Resolve-Path $SetupPath).Path /quiet /norestart
    if ($LASTEXITCODE -ne 0) {
        throw "WDK setup failed with exit code $LASTEXITCODE"
    }
}

if (!$InstallDotNetSdk -and !$InstallWix -and !$InstallWdkBuildTools -and !$LaunchElevatedWdkInstaller -and !$InstallSpectreLibraries -and !$InstallWdk -and !$PrintWdkInstructions) {
    Write-Host "No action selected. Use -All, -InstallDotNetSdk, -InstallWix, -InstallWdkBuildTools, -LaunchElevatedWdkInstaller, -InstallSpectreLibraries, -InstallWdk, or -PrintWdkInstructions."
    exit 0
}

$dotnetExe = Get-LocalDotnet -Root $DotNetDir
if ($InstallDotNetSdk -and !$dotnetExe) {
    $dotnetExe = Install-LocalDotnetSdk -Version $DotNetVersion -Root $DotNetDir
}
elseif ($InstallDotNetSdk) {
    Write-Host "[prereq] .NET SDK already available: $dotnetExe"
}

if ($InstallWix) {
    if (!$dotnetExe) {
        $dotnetExe = Install-LocalDotnetSdk -Version $DotNetVersion -Root $DotNetDir
    }
    Install-LocalWix -DotnetExe $dotnetExe -ToolPath $WixToolPath -Version $WixVersion
}

if ($InstallWdkBuildTools -or ($LaunchElevatedWdkInstaller -and !$InstallSpectreLibraries)) {
    Install-WdkBuildTools -InstallPath $VsInstallPath -LaunchElevated:$LaunchElevatedWdkInstaller
}

if ($InstallSpectreLibraries) {
    Install-VcSpectreLibraries -InstallPath $VsInstallPath -LaunchElevated:$LaunchElevatedWdkInstaller
}

if ($InstallWdk) {
    Install-Wdk -Url $WdkSetupUrl -SetupPath $WdkSetupPath
}

if ($PrintWdkInstructions) {
    Show-WdkInstructions -InstallPath $VsInstallPath
}

Write-Host "[prereq] done"
