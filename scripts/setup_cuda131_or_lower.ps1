param(
  [string]$Python = ".venv\Scripts\python.exe"
)

Write-Host "Installing the official PyTorch CUDA 12.8 wheels."
Write-Host "Use this for NVIDIA driver/CUDA environments up to CUDA 13.1 when the driver supports CUDA 12.x runtime wheels."
& "$PSScriptRoot\setup_cuda128.ps1" -Python $Python
