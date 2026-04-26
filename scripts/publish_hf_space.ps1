param(
    [string]$RemoteName = "hf",
    [string]$Branch = "main",
    [string]$SpaceBranch = "main"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$remoteUrl = git -C $repoRoot remote get-url $RemoteName
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($remoteUrl)) {
    throw "Could not read git remote '$RemoteName'. Add it with: git remote add hf https://huggingface.co/spaces/<user>/<space>"
}

$tempRoot = [System.IO.Path]::GetTempPath().TrimEnd("\")
$stamp = Get-Date -Format "yyyyMMddHHmmss"
$publishDir = Join-Path $tempRoot "sentinel-space-publish-$stamp"
New-Item -ItemType Directory -Path $publishDir | Out-Null

Write-Host "Preparing sanitized HF Space snapshot..."
Write-Host "Source: $repoRoot"
Write-Host "Temp:   $publishDir"

robocopy $repoRoot $publishDir /E /NFL /NDL /NJH /NJS /NP `
    /XD .git .github __pycache__ .pytest_cache .qodo .claude winner_analysis outputs notebooks tests wandb dist build .eggs `
    /XF *.pdf *.png *.jpg *.jpeg *.gif *.safetensors tokenizer.json uv.lock SENTINEL_MASTER_PLAN.md SENTINEL_ARCHITECTURE.md practice_reward_template.py tests_output.txt tests_output_fast.txt | Out-Null

if ($LASTEXITCODE -gt 7) {
    throw "robocopy failed with code $LASTEXITCODE"
}

$requirements = Join-Path $publishDir "requirements.txt"
if (-not (Test-Path -LiteralPath $requirements)) {
    throw "requirements.txt missing from publish snapshot"
}

$largeFiles = Get-ChildItem -Path $publishDir -Recurse -File |
    Where-Object { $_.Length -gt 10MB } |
    Select-Object FullName, Length
if ($largeFiles) {
    $largeFiles | Format-Table -AutoSize
    throw "Publish snapshot contains files over 10 MB. Refusing to push to HF Space."
}

Set-Location $publishDir
git init -b $SpaceBranch | Out-Null
git config user.email "sentinel-space@users.noreply.github.com"
git config user.name "sentinel-space-publisher"

git add .
git add -f requirements.txt requirements-train.txt 2>$null

$trackedRequirements = git ls-files requirements.txt
if ($trackedRequirements -ne "requirements.txt") {
    throw "requirements.txt is not tracked in the publish commit. Check .gitignore rules."
}

git commit -m "space: publish latest Sentinel app snapshot" | Out-Null
git remote add $RemoteName $remoteUrl

$head = git rev-parse HEAD
Write-Host "Publishing sanitized Space commit $head..."
git push --force $RemoteName "${SpaceBranch}:$Branch"

if ($LASTEXITCODE -ne 0) {
    throw "HF Space push failed"
}

Write-Host ""
Write-Host "HF Space publish complete."
Write-Host "Commit: $head"
Write-Host "Dashboard: https://srikrishna2005-openenv.hf.space/sentinel/dashboard"
Write-Host "Health:    https://srikrishna2005-openenv.hf.space/health"
