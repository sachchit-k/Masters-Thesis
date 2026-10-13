# READ-ONLY audit: finds every Python environment (site-packages) that contains our libraries. Deletes nothing.
# Writes the report to D:\Masters_Thesis_Sachchit\outputs\env_audit.txt
$libs  = "torch","transformers","captum","sentence_transformers","spacy","scipy","pandas"
$roots = @("C:\Users\acer","C:\Program Files","C:\Program Files (x86)","C:\ProgramData","D:\") | Where-Object { Test-Path $_ }
$rows = foreach ($sp in Get-ChildItem $roots -Recurse -Directory -Filter site-packages -ErrorAction SilentlyContinue) {
    $found = $libs | Where-Object { Test-Path (Join-Path $sp.FullName $_) }
    if ($found) {
        $t = Join-Path $sp.FullName "torch"
        $gb = if (Test-Path $t) { [math]::Round((Get-ChildItem $t -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum/1GB,2) } else { 0 }
        [pscustomobject]@{ Drive=$sp.FullName.Substring(0,2); Path=$sp.FullName; Libs=($found -join ","); TorchGB=$gb }
    }
}
$out = "D:\Masters_Thesis_Sachchit\outputs\env_audit.txt"
$rows | Sort-Object Drive,Path | Format-List | Out-File $out -Encoding utf8
$rows | Sort-Object Drive,Path | Format-Table Drive,TorchGB,Libs,Path -AutoSize -Wrap
Write-Host "`nEnvironments with our libraries: $(@($rows).Count). Report saved to $out"
Write-Host "Nothing was deleted."
