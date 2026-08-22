[CmdletBinding()]
param(
    [int]$MaximumPages = 100,
    [string]$Distro = 'Ubuntu-24.04',
    [string]$Python = 'python',
    [switch]$DownloadOnly,
    [switch]$Background
)

$ErrorActionPreference = 'Stop'
$repo = [System.IO.Path]::GetFullPath($PSScriptRoot)
if ($repo -notmatch '^(?<drive>[A-Za-z]):\\(?<tail>.*)$') {
    throw "This launcher expects the repository on a Windows drive: $repo"
}
$drive = $Matches.drive.ToLowerInvariant()
$tail = $Matches.tail.Replace('\', '/')
$linuxRepo = "/mnt/$drive/$tail"
$script = 'scripts/loc_live/run_best_churro_on_confirmed_untranscribed_loc.py'
$arguments = @('-d', $Distro, '--cd', $linuxRepo, $Python, $script, '--maximum-pages', "$MaximumPages")
if ($DownloadOnly) { $arguments += '--download-only' }

if ($Background) {
    $logRoot = Join-Path $repo 'outputs\confirmed_untranscribed_loc'
    New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
    $stdout = Join-Path $logRoot 'runner.stdout.log'
    $stderr = Join-Path $logRoot 'runner.stderr.log'
    $process = Start-Process -WindowStyle Hidden -FilePath 'wsl.exe' `
        -ArgumentList $arguments -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr -PassThru
    Write-Host "Started host PID $($process.Id)"
    Write-Host "Output: $logRoot"
    exit 0
}

& wsl.exe @arguments
exit $LASTEXITCODE
