# Creates (or recreates) repo\.venv on D: and installs dependencies. Run from repo\ in PowerShell.
# RULE 0: nothing may land on C:. Python itself is the uv-managed copy under D:\cache\uv\python (not the C: Python).
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)   # always work in repo\, whatever folder the script was started from
if ((Get-Location).Path -notlike "D:\*") { throw "Run this from D:\Masters_Thesis_Sachchit\repo" }
if (-not $env:HF_HOME)       { $env:HF_HOME="D:\cache\hf" }
if (-not $env:TORCH_HOME)    { $env:TORCH_HOME="D:\cache\torch" }
if (-not $env:PIP_CACHE_DIR) { $env:PIP_CACHE_DIR="D:\cache\pip" }
if (-not $env:UV_CACHE_DIR)  { $env:UV_CACHE_DIR="D:\cache\uv" }
$env:UV_PYTHON_INSTALL_DIR="D:\cache\uv\python"
$env:TEMP="D:\cache\tmp"; $env:TMP="D:\cache\tmp"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"
New-Item -ItemType Directory -Force D:\cache\tmp, D:\cache\uv\python | Out-Null

# uv bootstrap inside a venv on D: (no --user install into C:)
$boot = "D:\cache\uv-bootstrap"
if (-not (Test-Path "$boot\Scripts\uv.exe")) {
    python -m venv $boot
    & "$boot\Scripts\python.exe" -m pip install uv
}
$uv = "$boot\Scripts\uv.exe"

# --python-preference only-managed: always use uv's own Python 3.11 in D:\cache\uv\python, never a system Python on C:
# --clear: recreate .venv if it already exists (its Python was the C: one)
& $uv venv .venv --python 3.11 --python-preference only-managed --clear
if ($LASTEXITCODE -ne 0) { throw "uv venv failed" }
$cfg = Get-Content .\.venv\pyvenv.cfg | Select-String "^home"
Write-Host $cfg
if ($cfg -notmatch "= D:\\") { throw "venv Python is not on D: ($cfg)" }
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path

if (Test-Path requirements.lock.txt) {
    # reproducible rebuild from the frozen versions (torch+cu126 comes from the PyTorch index)
    & $uv pip install --python $py -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
    if ($LASTEXITCODE -ne 0) { throw "Install from requirements.lock.txt failed" }
} else {
    & $uv pip install --python $py torch --index-url https://download.pytorch.org/whl/cu126
    if ($LASTEXITCODE -ne 0) { throw "torch install failed" }
    & $uv pip install --python $py -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "requirements install failed" }
    & $py -m spacy download en_core_web_sm
    & $uv pip freeze --python $py | Set-Content -Encoding ascii requirements.lock.txt
}
Write-Host "Done. Next: .\.venv\Scripts\python scripts\smoke_test.py"
