param(
    [string]$Manifest = "$PSScriptRoot\output\delivery_manifest.json",
    [int]$BrowserProcessId = 141468,
    [switch]$VerifyOnly,
    [string[]]$SkipUpload = @()
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Add-Type -AssemblyName System.Windows.Forms
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class CourseDesktop {
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] public static extern IntPtr GetWindow(IntPtr hWnd, uint command);
    [DllImport("user32.dll")] public static extern bool IsWindow(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int cmd);
    [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
    [DllImport("user32.dll")] public static extern void mouse_event(uint flags, uint dx, uint dy, uint data, UIntPtr extra);
    [StructLayout(LayoutKind.Sequential)] public struct LASTINPUTINFO { public uint cbSize; public uint dwTime; }
    [DllImport("user32.dll")] public static extern bool GetLastInputInfo(ref LASTINPUTINFO info);
    public static double IdleSeconds() {
        LASTINPUTINFO info = new LASTINPUTINFO();
        info.cbSize = (uint)Marshal.SizeOf(info);
        GetLastInputInfo(ref info);
        return (unchecked((uint)Environment.TickCount) - info.dwTime) / 1000.0;
    }
}
'@
$folder = '1BiAyQ9qpmuspf_isz8S5o-xjsSFthccK'
$entries = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
$directory = Split-Path -Parent $Manifest
foreach ($entry in $entries) {
    if ($entry.filename -match '[\\/:]' -or [System.IO.Path]::GetFileName($entry.filename) -cne $entry.filename) {
        throw 'Delivery entries must name files directly inside the output directory.'
    }
}
$paths = @($entries | ForEach-Object { Join-Path $directory $_.filename })
for ($index = 0; $index -lt $paths.Count; $index++) {
    $path = $paths[$index]
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing deliverable: $path" }
    if ((Get-Item -LiteralPath $path).Length -ne $entries[$index].bytes -or
        (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $entries[$index].sha256) {
        throw "Deliverable changed after verification: $path"
    }
}
$browser = Get-Process -Id $BrowserProcessId
$handle = $browser.MainWindowHandle
if ($handle -eq 0) { throw 'The authorized Edge window is unavailable.' }

function Get-BrowserRoot {
    return [System.Windows.Automation.AutomationElement]::FromHandle($handle)
}
function Get-Elements {
    param($Root)
    return $Root.FindAll([System.Windows.Automation.TreeScope]::Descendants,
                        [System.Windows.Automation.Condition]::TrueCondition)
}
function Assert-Folder {
    $matched = $false
    foreach ($element in (Get-Elements (Get-BrowserRoot))) {
        if ($element.Current.ControlType -eq [System.Windows.Automation.ControlType]::Edit -and
            $element.Current.Name -eq 'Address and search bar') {
            $pattern = $null
            if ($element.TryGetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern, [ref]$pattern)) {
                $url = $pattern.Current.Value
                if ($url.ToLowerInvariant().StartsWith('drive.google.com/')) { $url = 'https://' + $url }
                $uri = $null
                if ([Uri]::TryCreate($url, [UriKind]::Absolute, [ref]$uri)) {
                    $pathPattern = '^/drive/(?:u/\d+/)?folders/' + [regex]::Escape($folder) + '/?$'
                    $matched = ($uri.Scheme -eq 'https' -and $uri.Host -eq 'drive.google.com' -and
                                $uri.AbsolutePath -cmatch $pathPattern)
                }
            }
            break
        }
    }
    if (-not $matched) { throw 'The active tab is not the authorized Drive folder; no navigation or clicks performed.' }
}
function Click-Element {
    param($Element)
    if ([CourseDesktop]::GetForegroundWindow() -ne $handle) {
        throw 'Foreground changed before a Drive click; no mouse event was sent.'
    }
    $point = $Element.GetClickablePoint()
    [void][CourseDesktop]::SetCursorPos([int]$point.X, [int]$point.Y)
    [CourseDesktop]::mouse_event(2, 0, 0, 0, [UIntPtr]::Zero)
    [CourseDesktop]::mouse_event(4, 0, 0, 0, [UIntPtr]::Zero)
}
function Find-Element {
    param([string]$Expression)
    $deadline = (Get-Date).AddSeconds(30)
    while ((Get-Date) -lt $deadline) {
        Assert-Folder
        foreach ($element in (Get-Elements (Get-BrowserRoot))) {
            if ($element.Current.IsEnabled -and -not $element.Current.IsOffscreen -and
                $element.Current.Name -match $Expression) { return $element }
        }
        Start-Sleep -Milliseconds 500
    }
    throw "Drive control unavailable: $Expression"
}
function Wait-Idle {
    $deadline = (Get-Date).AddMinutes(10)
    while ([CourseDesktop]::IdleSeconds() -lt 20) {
        if ((Get-Date) -gt $deadline) { throw 'Desktop remained active; upload deferred to avoid interfering.' }
        Start-Sleep -Seconds 20
    }
}
Assert-Folder
Wait-Idle
Assert-Folder
[void][CourseDesktop]::ShowWindow($handle, 9)
[void][CourseDesktop]::SetForegroundWindow($handle)
Start-Sleep -Seconds 1
if ([CourseDesktop]::GetForegroundWindow() -ne $handle) { throw 'Could not verify Edge foreground.' }

function Submit-FileUpload {
    param([string]$UploadPath)
    Assert-Folder
    if ([CourseDesktop]::GetForegroundWindow() -ne $handle) {
        throw 'Foreground changed before the next file; no click was sent.'
    }
    Click-Element (Find-Element '^New(?: Folder menu)?$')
    Start-Sleep -Seconds 1
    Click-Element (Find-Element '^File upload')
    Start-Sleep -Seconds 2
    $dialog = $null
    $foreground = [CourseDesktop]::GetForegroundWindow()
    $candidate = [System.Windows.Automation.AutomationElement]::FromHandle($foreground)
    if ($candidate.Current.Name -eq 'Open' -and [CourseDesktop]::GetWindow($foreground, 4) -eq $handle) {
        $dialog = $candidate
    }
    foreach ($window in [System.Windows.Automation.AutomationElement]::RootElement.FindAll(
        [System.Windows.Automation.TreeScope]::Children, [System.Windows.Automation.Condition]::TrueCondition)) {
        if ($null -ne $dialog) { break }
        $windowHandle = [IntPtr]$window.Current.NativeWindowHandle
        $owned = [CourseDesktop]::GetWindow($windowHandle, 4) -eq $handle
        if ($window.Current.Name -eq 'Open' -and $owned) { $dialog = $window; break }
    }
    if ($null -eq $dialog) { throw 'Native file chooser did not open; no filename was typed.' }
    $dialogHandle = [IntPtr]$dialog.Current.NativeWindowHandle
    if ([CourseDesktop]::GetForegroundWindow() -ne $dialogHandle) {
        throw 'File chooser is not foreground; no filename was typed.'
    }
    # A single absolute path avoids native-dialog multi-file text truncation.
    $selection = '"' + $UploadPath + '"'
    $escaped = $selection -replace '([+^%~(){}\[\]])', '{$1}'
    [System.Windows.Forms.SendKeys]::SendWait('%n')
    [System.Windows.Forms.SendKeys]::SendWait($escaped)
    if ([CourseDesktop]::GetForegroundWindow() -ne $dialogHandle) {
        throw 'Foreground changed during filename entry; upload confirmation was not sent.'
    }
    [System.Windows.Forms.SendKeys]::SendWait('{ENTER}')
    Start-Sleep -Seconds 5
    if ([CourseDesktop]::IsWindow($dialogHandle) -and [CourseDesktop]::IsWindowVisible($dialogHandle)) {
        throw 'The file chooser did not close; file selection may have been rejected.'
    }
    Assert-Folder
}

$completionEvidence = $null
if (-not $VerifyOnly) {
    $uploadPaths = @($paths | Where-Object { $SkipUpload -notcontains [System.IO.Path]::GetFileName($_) })
    if (-not $uploadPaths.Count) { throw 'No upload files remain; use VerifyOnly instead.' }
    $submitted = 0
    foreach ($uploadPath in $uploadPaths) {
        Submit-FileUpload $uploadPath
        $submitted++
        Write-Output "Submitted $submitted of $($uploadPaths.Count): $([System.IO.Path]::GetFileName($uploadPath))"
    }
    Write-Output 'Waiting for Drive upload completion; filenames will then be verified after reload.'
    $deadline = (Get-Date).AddHours(3)
    $completed = $false
    while ((Get-Date) -lt $deadline) {
        Assert-Folder
        $names = @((Get-Elements (Get-BrowserRoot)) | ForEach-Object { $_.Current.Name })
        if ($names -match '(?i)upload failed|could not upload|upload error|storage full') {
            throw 'Drive reported an upload failure; inspect the folder manually.'
        }
        if ($names -match '(?i)replace existing|keep both|upload as a new version') {
            throw 'Drive requested an overwrite decision; no existing file was replaced.'
        }
        $completionEvidence = $null
        foreach ($name in $names) {
            if ($name -match '(?i)\b(\d+)\s+(?:uploads?\s+complete(?:d)?|files?\s+uploaded)\b') {
                if ([int]$matches[1] -ge $uploadPaths.Count) { $completionEvidence = $matches[0] }
            }
        }
        if ($completionEvidence -and -not ($names -match '(?i)\buploading\b')) {
            $completed = $true
            break
        }
        Start-Sleep -Seconds 30
    }
    if (-not $completed) { throw 'Batch completion was not confirmed; transfers may still be running.' }
}

# Reloading removes transient upload UI. Verify persisted folder entries afterward.
Wait-Idle
Assert-Folder
[void][CourseDesktop]::SetForegroundWindow($handle)
if ([CourseDesktop]::GetForegroundWindow() -ne $handle) { throw 'Edge foreground changed before verification.' }
[System.Windows.Forms.SendKeys]::SendWait('^r')
Start-Sleep -Seconds 12
Assert-Folder
$observed = @{}
for ($page = 0; $page -lt 12; $page++) {
    $elements = Get-Elements (Get-BrowserRoot)
    foreach ($entry in $entries) {
        foreach ($element in $elements) {
            $name = $element.Current.Name
            $escapedName = [regex]::Escape($entry.filename)
            if ($name -eq $entry.filename -or $name -match ("^(?:Name: )?" + $escapedName + " (?:Video|Compressed archive|ZIP|More info)")) {
                $observed[$entry.filename] = $true
            }
        }
    }
    if ($observed.Count -eq $entries.Count) { break }
    Assert-Folder
    if ([CourseDesktop]::GetForegroundWindow() -ne $handle) {
        throw 'Foreground changed during folder verification; no scroll was sent.'
    }
    $rootBounds = (Get-BrowserRoot).Current.BoundingRectangle
    [void][CourseDesktop]::SetCursorPos([int]($rootBounds.Left+$rootBounds.Width*.7),
                                      [int]($rootBounds.Top+$rootBounds.Height*.7))
    [CourseDesktop]::mouse_event(0x0800, 0, 0, [uint32]4294966576, [UIntPtr]::Zero)
    Start-Sleep -Seconds 2
}
$missing = @($entries | Where-Object { -not $observed.ContainsKey($_.filename) } | ForEach-Object { $_.filename })
$report = [ordered]@{
    verified_at = (Get-Date).ToString('o')
    folder = "https://drive.google.com/drive/folders/$folder"
    method = 'Authorized signed-in Edge UI; exact filenames observed after folder reload.'
    upload_completion = $completionEvidence
    verified = @($observed.Keys | Sort-Object)
    missing = $missing
    success = ($missing.Count -eq 0)
}
[System.IO.File]::WriteAllText((Join-Path $directory 'drive_upload_verification.json'),
                              ($report | ConvertTo-Json -Depth 5))
if ($missing.Count) { throw "Not verified after reload: $($missing -join ', ')" }
Write-Output "Verified $($entries.Count) exact filenames after Drive folder reload."
