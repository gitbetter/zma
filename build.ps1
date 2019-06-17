param(
    [switch]$development=$false,
    [switch]$double_precision=$false
)

if (-not (Test-Path build -PathType Container)) {
    $exec = 'powershell .\rebuild.ps1'
    if ($development) {
	$exec = "$exec -development"
    }
    if ($double_precision) {
        $exec = "$exec -double_precision"
    }
    Invoke-Expression $exec
} else {
    cd build;
    $exec = 'cmake -G "Visual Studio 15 2017 Win64"'
    if ($development) {
        $exec = "$exec -DDEVELOPMENT=ON"
    }
    if ($double_precision) {
        $exec = "$exec -double_precision"
    }
    $exec = "$exec ..;"
    Invoke-Expression $exec
    cd ..;
}
