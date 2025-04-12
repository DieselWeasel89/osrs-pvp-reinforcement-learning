# --- CONFIGURATION ---
$ibPath = "C:\Program Files (x86)\IncrediBuild\ib_console.exe"
$projectDir = "projects"
$config = "Release"
$logDir = "build_logs"

# Create logs folder
if (-Not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

# Find projects: .sln, .bat, .py
$projectFiles = Get-ChildItem -Path $projectDir -Recurse -Include *.sln, *.bat, *.py

if ($projectFiles.Count -eq 0) {
    Write-Error "❌ No buildable project files (.sln, .bat, .py) found in $projectDir"
    exit 1
}

# Loop through each and run
foreach ($file in $projectFiles) {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $logFile = "$logDir\build-$($file.BaseName)-$timestamp.log"
    $ext = $file.Extension.ToLower()

    Write-Host "🚀 Building $($file.FullName) ..."
    
    if ($ext -eq ".sln") {
        & "$ibPath" BuildConsole $file.FullName "/cfg=$config" /rebuild /openmonitor *>> $logFile
    }
    elseif ($ext -eq ".bat") {
        & "$ibPath" BuildConsole $file.FullName /openmonitor *>> $logFile
    }
    elseif ($ext -eq ".py") {
        & "$ibPath" BuildConsole "python $($file.FullName)" /openmonitor *>> $logFile
    }

    Write-Host "✅ Log: $logFile"
}