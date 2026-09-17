# Runs run_equilibrium_1900.py (1900-2070) in batches of 5 seeds at once, one network at a time,
# then summarises with analyse_equilibrium_1900.py.
#
#   powershell -ExecutionPolicy Bypass -File run_equilibrium_batches.ps1
#   powershell -ExecutionPolicy Bypass -File run_equilibrium_batches.ps1 -Networks gamma_2,powerlaw_3
#
# Each run needs up to ~2 GB RAM (5 at once ~10 GB); close other heavy apps first.
# Logs: csvs\equilibrium_1900\logs\<network>_seed<seed>.log

param(
    [string[]]$Networks = @('gamma_5', 'gamma_2', 'gamma_1', 'gamma_0.25', 'gamma_0.05', 'powerlaw_3'),
    [int[]]$Seeds = @(0, 1, 2, 3, 4),
    [string]$Python = 'C:\Users\richa\miniconda3\envs\summerhpvsim\python.exe'
)

$root = $PSScriptRoot
$logDir = Join-Path $root 'csvs\equilibrium_1900\logs'
New-Item -ItemType Directory -Force $logDir | Out-Null

$env:PYTHONPATH = $root
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '1'      # one numerical thread per process, so 5 runs don't oversubscribe 6 cores
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'

foreach ($net in $Networks) {
    Write-Host "$(Get-Date -Format HH:mm) starting $net"
    $procs = foreach ($seed in $Seeds) {
        $log = Join-Path $logDir "$($net)_seed$seed"
        Start-Process -FilePath $Python -ArgumentList @('run_equilibrium_1900.py', $net, $seed) `
            -WorkingDirectory $root -NoNewWindow -PassThru `
            -RedirectStandardOutput "$log.log" -RedirectStandardError "$log.err"
    }
    $procs | Wait-Process
    foreach ($seed in $Seeds) {
        $log = Join-Path $logDir "$($net)_seed$seed"
        $done = Select-String -Path "$log.log" -Pattern '^DONE' | Select-Object -Last 1
        if ($done) { Write-Host "  $($done.Line)" } else { Write-Host "  seed $seed FAILED - see $log.err" }
    }
}

Write-Host "$(Get-Date -Format HH:mm) all batches finished; summarising"
& $Python (Join-Path $root 'analyse_equilibrium_1900.py')
