param (
    [string]$exp = "HashGrid",
    [switch]$trainable_cameras,
    [int]$scan_id = 114,
    [switch]$eval_rendering,
    [switch]$h
)

function Show-Usage {
    Write-Output "Usage: ./run_eval.ps1 [-exp <EXPERIMENT>] [-trainable_cameras] [-scan_id <SCAN_ID>] [-eval_rendering] [-h]"
    Write-Output "Options:"
    Write-Output "  -exp <EXPERIMENT>          Specify the experiment name (default: HashGrid)"
    Write-Output "  -trainable_cameras         Use trainable cameras"
    Write-Output "  -scan_id <SCAN_ID>         Specify the scan ID (default: 114)"
    Write-Output "  -eval_rendering            Enable rendering evaluation"
    Write-Output "  -h                         Display this help message"
    exit 0
}

if ($h) {
    Show-Usage
}

# Resolve script directory and move one level up
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location (Join-Path $ScriptDir "..")

# Determine config directory
switch ($exp) {
    "HashGrid"       { $configBase = "./confs/embedder_conf_var/MultiResHashPointsAndViewDirs" }
    "Posenc"         { $configBase = "./confs/embedder_conf_var/PosEnc" }
    "FourierNTK"     { $configBase = "./confs/embedder_conf_var/FourierFeatures" }
    "HashGridCUDA"   { $configBase = "./confs/embedder_conf_var/CUDA_HashGrid" }
    "NFFB"           { $configBase = "./confs/embedder_conf_var/FFB" }
    "StylemodNFFB"   { $configBase = "./confs/embedder_conf_var/FFB_StyleMod" }
    "HashGridTCNN"   { $configBase = "./confs/embedder_conf_var/HashGrid_TCNN_PointsAndViewDirs" }
    "HashNerf"       { $configBase = "./confs/embedder_conf_var/MultiResHashPointsPosencViews" }
    "NFFB_TCNN"      { $configBase = "./confs/embedder_conf_var/FFB_TCNN" }
    default {
        Write-Error "Invalid experiment name: $exp"
        exit 1
    }
}

# Append camera configuration
if ($trainable_cameras) {
    $configPath = Join-Path $configBase "dtu_trained_cameras.conf"
} else {
    $configPath = Join-Path $configBase "dtu_fixed_cameras.conf"
}

# Main evaluation loop
while ($true) {
    Write-Output "Working directory: $(Get-Location)"
    Write-Output "Starting Neural Surface Reconstruction Evaluation... $exp"
    Write-Output "Config directory: $configPath"
    Write-Output "Scan ID: $scan_id"

    if ($eval_rendering) {
        Write-Output "Rendering evaluation enabled"
        python -u ./evaluation/eval.py --expname $exp --conf $configPath --scan_id $scan_id --checkpoint latest --eval_rendering
    } else {
        Write-Output "Rendering evaluation disabled"
        python -u ./evaluation/eval.py --expname $exp --conf $configPath --scan_id $scan_id --checkpoint latest
    }

    if ($LASTEXITCODE -eq 0) {
        Write-Output "Python script finished successfully, exiting loop."
        break
    } else {
        Write-Output "Python script failed with exit code $LASTEXITCODE, restarting..."
    }
}
