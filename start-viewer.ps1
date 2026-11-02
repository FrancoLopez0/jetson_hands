param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
& uv run --no-project --python 3.14 python -m http.server $Port --bind 127.0.0.1 --directory "$PSScriptRoot\web"
