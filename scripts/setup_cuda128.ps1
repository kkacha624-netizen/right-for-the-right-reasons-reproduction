param(
  [string]$Python = ".venv\Scripts\python.exe"
)

if (!(Test-Path $Python)) {
  Write-Error "Python executable not found: $Python. Run 'uv venv' and 'uv sync' first."
  exit 1
}

uv pip install --python $Python --reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}
& $Python -c "import torch; print('cuda_available=', torch.cuda.is_available(), 'torch_cuda=', torch.version.cuda)"
