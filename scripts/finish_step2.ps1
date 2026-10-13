# Run after setup_env.ps1 and a passing smoke test. Commits the first commit and verifies ignores.
$ErrorActionPreference = "Stop"
git add -A
git status --short
$bad = git ls-files | Select-String -Pattern '^(data|models|outputs|\.venv)/'
if ($bad) { throw "Ignored folders are tracked: $bad" }
git commit -m "Initial commit: thesis skeleton, project rules, env setup, smoke test" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`nClaude-Session: https://claude.ai/code/session_01Hm8Xanyyd61ZhTCaurNwM1"
git ls-files
