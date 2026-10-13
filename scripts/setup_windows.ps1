<#
.SYNOPSIS
  Download LibTorch (if needed), build go9 with Visual Studio + CMake, and run the tests.

.DESCRIPTION
  Requirements: Visual Studio 2022 (Desktop development with C++ workload) and CMake >= 3.18
  (CMake ships with Visual Studio; run this from a "Developer PowerShell for VS 2022" if cmake
  is not on PATH). An NVIDIA driver is all you need for GPU training - the CUDA toolkit is NOT
  required because go9 links the LibTorch libraries directly.

.EXAMPLE
  .\scripts\setup_windows.ps1                       # NVIDIA GPU (CUDA 12.8 LibTorch, RTX 20xx-50xx)
.EXAMPLE
  .\scripts\setup_windows.ps1 -Variant cu118        # older GPUs (GTX 10xx and up)
.EXAMPLE
  .\scripts\setup_windows.ps1 -Variant cpu          # no GPU
.EXAMPLE
  .\scripts\setup_windows.ps1 -UsePipTorch          # reuse a torch installed with pip
.EXAMPLE
  .\scripts\setup_windows.ps1 -TorchRoot D:\libtorch
#>
param(
  [ValidateSet("cu128", "cu126", "cu118", "cpu")] [string]$Variant = "cu128",
  [string]$TorchVersion = "2.7.1",
  [string]$TorchRoot = "",
  [switch]$UsePipTorch,
  [string]$BuildDir = "build",
  [string]$Generator = "",
  [switch]$NoTests
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is 10x faster without the progress bar
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if ($UsePipTorch) {
  $TorchRoot = (python -c "import torch, os; print(os.path.dirname(torch.__file__))").Trim()
  Write-Host "Using pip torch at $TorchRoot"
}
if (-not $TorchRoot) {
  $TorchRoot = Join-Path $Root "libtorch"
  if (-not (Test-Path (Join-Path $TorchRoot "lib\torch.lib"))) {
    $url = "https://download.pytorch.org/libtorch/$Variant/libtorch-win-shared-with-deps-$TorchVersion%2B$Variant.zip"
    $zip = Join-Path $env:TEMP "libtorch-$TorchVersion-$Variant.zip"
    if (-not (Test-Path $zip)) {
      Write-Host "Downloading LibTorch $TorchVersion ($Variant) - this is a large file (several GB for CUDA)..."
      Write-Host "  $url"
      Invoke-WebRequest -Uri $url -OutFile $zip
    }
    Write-Host "Extracting to $Root\libtorch ..."
    if (Get-Command tar -ErrorAction SilentlyContinue) { tar -xf $zip -C $Root } else { Expand-Archive $zip -DestinationPath $Root -Force }
  }
}
if (-not (Test-Path (Join-Path $TorchRoot "lib\torch.lib"))) {
  throw "LibTorch not found at $TorchRoot (expected lib\torch.lib)"
}

$cmakeArgs = @("-S", ".", "-B", $BuildDir, "-DTORCH_ROOT=$TorchRoot", "-DGO9_BUILD_TESTS=ON")
if ($Generator) { $cmakeArgs += @("-G", $Generator) }
if (-not $Generator -or $Generator -like "Visual Studio*") { $cmakeArgs += @("-A", "x64") }
Write-Host "cmake $($cmakeArgs -join ' ')"
cmake @cmakeArgs
if ($LASTEXITCODE) { throw "CMake configure failed" }
cmake --build $BuildDir --config Release --parallel
if ($LASTEXITCODE) { throw "Build failed" }

$exeDir = Join-Path $BuildDir "Release"
if (-not (Test-Path (Join-Path $exeDir "go9.exe"))) { $exeDir = $BuildDir }
if (-not $NoTests) {
  & (Join-Path $exeDir "go9_tests.exe")
  if ($LASTEXITCODE) { throw "Unit tests failed" }
  & (Join-Path $exeDir "go9_crossval.exe") tests\data\openspiel_games.txt
  if ($LASTEXITCODE) { throw "OpenSpiel cross-validation failed" }
  & (Join-Path $exeDir "go9.exe") selftest
  & (Join-Path $exeDir "go9.exe") bench
}
Write-Host ""
Write-Host "Build complete. Next steps:" -ForegroundColor Green
Write-Host "  $exeDir\go9.exe train --config configs\gpu.ini --run_dir runs\main     # start / resume training"
Write-Host "  $exeDir\go9.exe gui --run runs\main --open                             # play + watch training"
