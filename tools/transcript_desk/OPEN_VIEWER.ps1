[CmdletBinding()]
param(
    [int]$Port = 8813,
    [string]$Predictions = '',
    [string]$Python = '',
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$root = [System.IO.Path]::GetFullPath($PSScriptRoot)
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $root '..\..'))
$server = Join-Path $root 'server.py'
$url = "http://127.0.0.1:$Port/"
$health = "http://127.0.0.1:$Port/api/health"

if ([string]::IsNullOrWhiteSpace($Predictions)) {
    $Predictions = Join-Path $repoRoot 'outputs\confirmed_untranscribed_loc\predictions_grounded\predictions.jsonl'
}
$Predictions = [System.IO.Path]::GetFullPath($Predictions)
if (-not (Test-Path -LiteralPath $Predictions -PathType Leaf)) {
    throw "Predictions file not found yet: $Predictions"
}

$pythonExecutable = $null
$pythonCandidates = @()
if (-not [string]::IsNullOrWhiteSpace($Python)) {
    $pythonCandidates += $Python
} else {
    $pythonCandidates += (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe')
    $pythonCandidates += (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\python.exe')
    $pythonCandidates += (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe')
    $pythonCandidates += (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe')
    $pythonCandidates += 'py.exe'
    $pythonCandidates += 'python.exe'
}
foreach ($candidate in $pythonCandidates) {
    if ([string]::IsNullOrWhiteSpace($candidate)) { continue }
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        $pythonExecutable = [System.IO.Path]::GetFullPath($candidate)
        break
    }
    $pythonCommand = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($null -ne $pythonCommand -and $pythonCommand.Source -notmatch '\\WindowsApps\\') {
        $pythonExecutable = $pythonCommand.Source
        break
    }
}
if ($null -eq $pythonExecutable) {
    throw 'Python 3 was not found. Install Python or pass -Python C:\path\to\python.exe.'
}

$alreadyRunning = $false
try {
    $response = Invoke-RestMethod -Uri $health -TimeoutSec 2
    $alreadyRunning = [bool]$response.ok
} catch {
    $alreadyRunning = $false
}

if (-not $alreadyRunning) {
    $stdout = Join-Path $root 'viewer.stdout.log'
    $stderr = Join-Path $root 'viewer.stderr.log'
    $process = Start-Process -WindowStyle Hidden -FilePath $pythonExecutable -WorkingDirectory $root `
        -ArgumentList @($server, '--port', "$Port", '--predictions', $Predictions) `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    Set-Content -LiteralPath (Join-Path $root 'viewer.pid') -Value $process.Id
    $ready = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Milliseconds 250
        if ($process.HasExited) {
            $detail = if (Test-Path -LiteralPath $stderr) { Get-Content -LiteralPath $stderr -Raw } else { '' }
            throw "Transcript viewer exited during startup.`n$detail"
        }
        try {
            $response = Invoke-RestMethod -Uri $health -TimeoutSec 2
            if ($response.ok) { $ready = $true; break }
        } catch { }
    }
    if (-not $ready) { throw "Transcript viewer did not become ready at $url" }
}

Write-Host "LOC Transcript Desk: $url"
Write-Host "Predictions: $Predictions"
if (-not $NoBrowser) { Start-Process $url }
