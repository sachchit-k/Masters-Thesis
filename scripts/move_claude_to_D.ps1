# Moves Claude Code (CLI binary + config/history) from C: to D:. Run in PowerShell. Does NOT delete anything on C:.
# Close all Claude Code terminal sessions first (the Cursor chat panel may stay open).
$ErrorActionPreference = "Stop"
$srcExe = "$env:USERPROFILE\.local\bin\claude.exe"
$srcCfg = "$env:USERPROFILE\.claude"
$srcJson = "$env:USERPROFILE\.claude.json"
$dstBin = "D:\tools\claude\bin"
$dstCfg = "D:\tools\claude\config"
if (-not (Test-Path $srcExe)) { throw "Not found: $srcExe" }
New-Item -ItemType Directory -Force $dstBin, $dstCfg | Out-Null

# 1. copy (not move) so C: stays intact until D: is verified
Copy-Item $srcExe "$dstBin\claude.exe" -Force
if (Test-Path $srcCfg)  { Copy-Item "$srcCfg\*" $dstCfg -Recurse -Force }
if (Test-Path $srcJson) { Copy-Item $srcJson "$dstCfg\.claude.json" -Force }

# 2. user-level env vars (this account only): config on D:, no background self-update back onto C:
[Environment]::SetEnvironmentVariable("CLAUDE_CONFIG_DIR", $dstCfg, "User")
[Environment]::SetEnvironmentVariable("DISABLE_AUTOUPDATER", "1", "User")

# 3. user PATH: D: bin first, drop the C: bin entry
$parts = [Environment]::GetEnvironmentVariable("Path","User") -split ";" | Where-Object { $_ -and $_ -notlike "*\.local\bin*" }
$new = (@($dstBin) + $parts) -join ";"
[Environment]::SetEnvironmentVariable("Path", $new, "User")

# 4. verify from D: in this window
$env:CLAUDE_CONFIG_DIR = $dstCfg
& "$dstBin\claude.exe" --version
Write-Host "`nOpen a NEW terminal and run:  claude --version ; claude doctor ; where.exe claude"
Write-Host "`nOnly after both work and 'where.exe claude' shows D:\tools\claude\bin, remove the C: copies yourself:"
Write-Host '  Remove-Item "$env:USERPROFILE\.local\bin\claude.exe" -Force'
Write-Host '  Remove-Item "$env:USERPROFILE\.local\share\claude" -Recurse -Force'
Write-Host '  Remove-Item "$env:USERPROFILE\.claude" -Recurse -Force ; Remove-Item "$env:USERPROFILE\.claude.json" -Force'
Write-Host "(I did not delete them: the project rule is never to delete on system drives without you doing it.)"
