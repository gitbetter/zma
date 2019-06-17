param(
    [switch]$development=$false,
    [switch]$double_precision=$false
)

if (Test-Path build -PathType Container) {
    rm -r -force build/;
}

mkdir build; cd build;

$exec = 'cmake -G "Visual Studio 15 2017 Win64"'

if ($development) {
    $exec = "$exec -DDEVELOPMENT=ON"
}

if ($double_precision) {
    $exec = "$exec -DDOUBLE_PRECISION=ON"
}

$exec = "$exec ..;"

Invoke-Expression $exec

explorer .; cd ..;
