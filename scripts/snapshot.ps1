$ProjectRoot = Get-Location
$OutputZip = Join-Path $ProjectRoot "atlas_snapshot.zip"

$Exclude = @(
    ".git",
    ".venv",
    "__pycache__",
    "data",
    "logs",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "build",
    "dist",
    ".idea",
    ".vscode"
)

if (Test-Path $OutputZip) {
    Remove-Item $OutputZip -Force
}

$temp = Join-Path $env:TEMP "atlas_snapshot"

if (Test-Path $temp) {
    Remove-Item $temp -Recurse -Force
}

New-Item -ItemType Directory -Path $temp | Out-Null

Get-ChildItem -Recurse | ForEach-Object {

    $relative = $_.FullName.Substring($ProjectRoot.Path.Length + 1)

    foreach ($folder in $Exclude) {
        if ($relative -like "$folder*" -or $relative -like "*\$folder\*") {
            return
        }
    }

    $destination = Join-Path $temp $relative

    if ($_.PSIsContainer) {

        New-Item -ItemType Directory -Force -Path $destination | Out-Null

    } else {

        $parent = Split-Path $destination

        if (!(Test-Path $parent)) {
            New-Item -ItemType Directory -Force -Path $parent | Out-Null
        }

        Copy-Item $_.FullName $destination
    }

}

tree $temp /F /A > (Join-Path $temp "project_structure.txt")

Compress-Archive `
    -Path (Join-Path $temp "*") `
    -DestinationPath $OutputZip `
    -CompressionLevel Optimal

Remove-Item $temp -Recurse -Force

Write-Host ""
Write-Host "Atlas snapshot created:"
Write-Host $OutputZip -ForegroundColor Green