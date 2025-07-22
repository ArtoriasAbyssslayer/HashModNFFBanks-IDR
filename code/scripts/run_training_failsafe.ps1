# Set alias for python
Set-Alias python3 python

# Get total memory in KB
$MemTotal = (Get-CimInstance -ClassName Win32_ComputerSystem).TotalPhysicalMemory / 1KB
$MemoryLimit = [int]($MemTotal * 0.9)

# Set memory limit (Windows does not support ulimit, so this is informational)
Write-Host "Total available memory: {0:N2} GB" -f ($MemTotal / 1024 / 1024)

# Default values
$EXPERIMENT = "HashGrid"
$TRAINABLE_CAMERAS = $false
$SCAN_ID = 114
$INCLUDE_IS_CONTINUE = $false

# Display usage instructions
function Show-Usage {
    Write-Host "Usage: .\run_experiment.ps1 [--exp <EXPERIMENT>] [--trainable_cameras] [--scan_id <SCAN_ID>] [--is_continue]"
    exit 1
}

# Parse command line arguments
for ($i = 0; $i -lt $args.Length; $i++) {
    switch ($args[$i]) {
        "--exp" {
            $i++; $EXPERIMENT = $args[$i]
        }
        "--trainable_cameras" {
            $TRAINABLE_CAMERAS = $true
        }
        "--scan_id" {
            $i++; $SCAN_ID = [int]$args[$i]
        }
        "--is_continue" {
            $INCLUDE_IS_CONTINUE = $true
        }
        "-h" {
            Show-Usage
        }
        default {
            Write-Error "Unknown option: $($args[$i])"
            exit 1
        }
    }
}

# Set config directory
switch ($EXPERIMENT) {
    "HashGrid"        { $CONFIG_DIR = "./confs/embedder_conf_var/MultiResHashPointsAndViewDirs" }
    "Posenc"          { $CONFIG_DIR = "./confs/embedder_conf_var/PosEnc" }
    "FourierNTK"      { $CONFIG_DIR = "./confs/embedder_conf_var/FourierFeatures" }
    "HashGridCUDA"    { $CONFIG_DIR = "./confs/embedder_conf_var/CUDA_HashGrid" }
    "NFFB"            { $CONFIG_DIR = "./confs/embedder_conf_var/FFB" }
    "StylemodNFFB"    { $CONFIG_DIR = "./confs/embedder_conf_var/FFB_StyleMod" }
    "HashGridTCNN"    { $CONFIG_DIR = "./confs/embedder_conf_var/HashGrid_TCNN_PointsAndViewDirs" }
    "HashNerf"        { $CONFIG_DIR = "./confs/embedder_conf_var/MultiResHashPointsPosencViews" }
    "NFFB_TCNN"       { $CONFIG_DIR = "./confs/embedder_conf_var/FFB_TCNN" }
    default {
        Write-Error "Invalid experiment name: $EXPERIMENT"
        exit 1
    }
}

# Append the config file
if ($TRAINABLE_CAMERAS) {
    $CONFIG_DIR = "$CONFIG_DIR/dtu_trained_cameras.conf"
} else {
    $CONFIG_DIR = "$CONFIG_DIR/dtu_fixed_cameras.conf"
}

# Change to script parent directory
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location (Join-Path $SCRIPT_DIR "..")

Write-Host "Is continue: $INCLUDE_IS_CONTINUE"

# Retry loop
while ($true) {
    Write-Host "Working directory: $(Get-Location)"
    Write-Host "Starting Neural Surface Reconstruction Experiment...$EXPERIMENT"
    Write-Host "Config directory: $CONFIG_DIR"
    Write-Host "Scan ID: $SCAN_ID"

    if ($INCLUDE_IS_CONTINUE) {
        Write-Host "Continue training from the latest checkpoint"
        python3 -u ./training/exp_runner.py --conf $CONFIG_DIR --expname $EXPERIMENT --scan_id $SCAN_ID --checkpoint latest --validation_slope_print --is_continue
    } else {
        Write-Host "Start training from scratch"
        python3 -u ./training/exp_runner.py --conf $CONFIG_DIR --expname $EXPERIMENT --scan_id $SCAN_ID --checkpoint latest --validation_slope_print
    }

    if ($LASTEXITCODE -eq 0) {
        Write-Host "Python script finished successfully, exiting loop."
        break
    } else {
        Write-Warning "Python script failed with exit code $LASTEXITCODE, restarting..."
        $INCLUDE_IS_CONTINUE = $true
    }
}
