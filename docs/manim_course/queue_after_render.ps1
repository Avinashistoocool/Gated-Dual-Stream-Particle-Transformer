param(
    [Parameter(Mandatory=$true)][int[]]$After,
    [Parameter(Mandatory=$true)][int[]]$Chapters
)
$ErrorActionPreference = 'Stop'
$sourceText = $After -join ' '
$pattern = '*render_course.py --chapter ' + $sourceText + ' --jobs 1*'
$processes = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
               Where-Object { $_.CommandLine -like $pattern })
if (-not $processes.Count) { throw "No active predecessor render found: $sourceText" }
$suffix = ($After | ForEach-Object { '{0:00}' -f $_ }) -join '_'
$targetSuffix = ($Chapters | ForEach-Object { '{0:00}' -f $_ }) -join '_'
$predecessorStatus = Join-Path $PSScriptRoot "output\render_status_$suffix.json"
$statusPath = Join-Path $PSScriptRoot "output\queue_$targetSuffix.json"
$record = [ordered]@{
    after = $After
    chapters = $Chapters
    predecessor_processes = @($processes.ProcessId)
    queued_at = (Get-Date).ToString('o')
    status = 'waiting for predecessor render'
}
[System.IO.File]::WriteAllText($statusPath, ($record | ConvertTo-Json -Depth 5))
Write-Output "Queued chapters $($Chapters -join ',') after $sourceText; waiting on PIDs $($processes.ProcessId -join ',')."
Wait-Process -Id @($processes.ProcessId) -ErrorAction SilentlyContinue
if (-not (Test-Path -LiteralPath $predecessorStatus)) { throw 'Predecessor did not persist a completed render status.' }
$status = Get-Content -LiteralPath $predecessorStatus -Raw | ConvertFrom-Json
if ($status.failures.Count -or $status.chapters.Count -ne $After.Count) {
    throw 'Predecessor has failed or incomplete renders; downstream rendering was not started.'
}
$record['status'] = 'rendering'
$record['started_at'] = (Get-Date).ToString('o')
[System.IO.File]::WriteAllText($statusPath, ($record | ConvertTo-Json -Depth 5))
& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\render_course.py" --chapter $Chapters --jobs 1
$result = $LASTEXITCODE
$record['status'] = if ($result -eq 0) { 'rendered; media QA pending' } else { 'render failed' }
$record['completed_at'] = (Get-Date).ToString('o')
[System.IO.File]::WriteAllText($statusPath, ($record | ConvertTo-Json -Depth 5))
exit $result
