# Runs the Task 3/4 measurement sequence of WEIGHT_PROVIDER.md unattended on Windows.
# Default paths are for the owner machine (RTX 2060 Super 8 GB, 32 GB RAM, Qwen3-30B-A3B Q4_K_M).
# Every step logs to its own file in the results folder; a failed step is recorded and the script goes on.
# At the end summary.md lists every step with status, time and the main numbers.
#
#   powershell -ExecutionPolicy Bypass -File tools\expert-trace\measure-windows.ps1
#   powershell -ExecutionPolicy Bypass -File tools\expert-trace\measure-windows.ps1 -Repeats 1 -SkipKld

param(
    [string] $Model      = "D:\AInode2\models\Qwen3-30B-A3B-Q4_K_M.gguf",
    [string] $RepoRoot   = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [string] $BinDir     = "",
    [string[]] $Traces   = @("tr_sr_pravni.csv", "trace_sr.csv", "trace_code.csv"),
    [string] $TestTrace  = "trace_code.csv",   # B of analyze.py; only changes the printed hit rates
    [string] $Python     = "python",
    [string] $ResultsDir = "",
    [int]    $Ctx        = 4096,
    [int]    $Repeats    = 3,                  # llama-batched-bench runs per configuration
    [int]    $KldChunks  = 40,
    [double] $MinFreeRamGB = 20,
    [string] $TextPrompt = "",                 # held-out text prompt; default: heldout_sr.txt in the repo root if present
    [string] $CodePrompt = "",                 # held-out code prompt; default: a part of src/llama-sampler.cpp
    [int[]]  $ProfileThreads = @(4, 6, 8),    # decode thread sweep of the profile section
    [switch] $SkipKld,
    [switch] $SkipProfile
)

$ErrorActionPreference = "Continue"

if (-not $BinDir)     { $BinDir     = Join-Path $RepoRoot "build\bin\Release" }
if (-not $ResultsDir) { $ResultsDir = Join-Path $RepoRoot ("results\moe-" + (Get-Date -Format "yyyyMMdd-HHmmss")) }
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

$StepsLog = Join-Path $ResultsDir "steps.log"
$script:StepNo  = 0
$script:Results = @()

function Write-Log([string] $msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $StepsLog -Value $line
}

function Quote-Arg([string] $a) {
    if ($a -eq "") { return '""' }
    if ($a -match '[\s"]') { return '"' + ($a -replace '"', '\"') + '"' }
    return $a
}

# runs one step; stdout and stderr go to one log file; returns the log path
function Invoke-Step([string] $Name, [string] $Exe, [string[]] $ArgList, [hashtable] $EnvVars = @{}) {
    $script:StepNo++
    $log = Join-Path $ResultsDir ("{0:D2}-{1}.log" -f $script:StepNo, $Name)
    $argString = ($ArgList | ForEach-Object { Quote-Arg $_ }) -join " "
    Write-Log "START $Name"
    Set-Content -Path $log -Value ("# {0} {1}`n# started {2}`n" -f $Exe, $argString, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    foreach ($k in $EnvVars.Keys) {
        Set-Item -Path ("env:" + $k) -Value $EnvVars[$k]
        Add-Content -Path $log -Value ("# env {0}={1}" -f $k, $EnvVars[$k])
    }
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $code = -1
    try {
        $p = Start-Process -FilePath $Exe -ArgumentList $argString -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput "$log.out" -RedirectStandardError "$log.err"
        $code = $p.ExitCode
    } catch {
        Add-Content -Path $log -Value ("could not start: " + $_.Exception.Message)
    }
    $sw.Stop()
    foreach ($k in $EnvVars.Keys) {
        Remove-Item -Path ("env:" + $k) -ErrorAction SilentlyContinue
    }
    foreach ($f in @("$log.out", "$log.err")) {
        if (Test-Path $f) {
            $text = Get-Content -Path $f -Raw
            if ($text) { Add-Content -Path $log -Value $text }
            Remove-Item $f
        }
    }
    $status = if ($code -eq 0) { "OK" } else { "FAILED ($code)" }
    Add-Content -Path $log -Value ("`n# finished {0}, exit code {1}, {2:N0} s" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $code, $sw.Elapsed.TotalSeconds)
    Write-Log ("END   {0}: {1}, {2:N0} s, log {3}" -f $Name, $status, $sw.Elapsed.TotalSeconds, (Split-Path $log -Leaf))
    $script:Results += [pscustomobject]@{ Step = $Name; Status = $status; Seconds = [int]$sw.Elapsed.TotalSeconds; Log = (Split-Path $log -Leaf) }
    return $log
}

function Skip-Step([string] $Name, [string] $Why) {
    $script:StepNo++
    Write-Log "SKIP  $Name - $Why"
    $script:Results += [pscustomobject]@{ Step = $Name; Status = "SKIPPED: $Why"; Seconds = 0; Log = "" }
}

function Bin([string] $name) { return (Join-Path $BinDir ($name + ".exe")) }

# ---------------------------------------------------------------- 0. checks
Write-Log "results: $ResultsDir"
Write-Log "model:   $Model"
Write-Log "bins:    $BinDir"

$os = Get-CimInstance Win32_OperatingSystem
$freeRamGB  = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
$totalRamGB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
$pf = Get-CimInstance Win32_PageFileUsage
$swapTotalGB = [math]::Round((($pf | Measure-Object -Property AllocatedBaseSize -Sum).Sum) / 1KB, 2)
$swapUsedGB  = [math]::Round((($pf | Measure-Object -Property CurrentUsage -Sum).Sum) / 1KB, 2)
Write-Log ("RAM: {0} GB free of {1} GB; swap (page file): {2} GB used of {3} GB" -f $freeRamGB, $totalRamGB, $swapUsedGB, $swapTotalGB)
if ($freeRamGB -lt $MinFreeRamGB) {
    Write-Log ("ABORT: free RAM {0} GB is under {1} GB; close other programs and run again" -f $freeRamGB, $MinFreeRamGB)
    exit 1
}
$nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvsmi) {
    $gpu = & nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total --format=csv,noheader 2>&1
    Write-Log "GPU: $gpu"
}

foreach ($f in @($Model, (Bin "llama-debug"), (Bin "llama-perplexity"), (Bin "llama-batched-bench"), (Bin "llama-completion"), (Bin "llama-bench"))) {
    if (-not (Test-Path $f)) { Write-Log "missing: $f (steps that need it will fail)" }
}

# ---------------------------------------------------------------- 1. placements
$tracePaths = @()
foreach ($t in $Traces) {
    $p = Join-Path $RepoRoot $t
    if (-not (Test-Path $p)) { Write-Log "missing trace: $p" }
    $side = [IO.Path]::ChangeExtension($p, ".json")
    if (-not (Test-Path $side)) { Write-Log ("missing sidecar {0}: expert bytes are unknown, the 3G/5G hot sets will be wrong" -f $side) }
    $tracePaths += $p
}
$groupA  = $tracePaths -join ","
$traceB  = Join-Path $RepoRoot $TestTrace
$analyze = Join-Path $RepoRoot "tools\expert-trace\analyze.py"
$p3      = Join-Path $ResultsDir "placement-3g.json"
$p5      = Join-Path $ResultsDir "placement-5g.json"
$rank    = Join-Path $ResultsDir "ranking.json"
$cold    = Join-Path $ResultsDir "all-cold.json"
Set-Content -Path $cold -Value '{"layers": []}' -Encoding ascii

Invoke-Step "placement-3g" $Python @($analyze, $groupA, $traceB, "--no-plots", "--vram-budget", "3G", "--emit-placement", $p3) | Out-Null
Invoke-Step "placement-5g" $Python @($analyze, $groupA, $traceB, "--no-plots", "--vram-budget", "5G", "--emit-placement", $p5) | Out-Null
Invoke-Step "ranking"      $Python @($analyze, $groupA, $traceB, "--no-plots", "--emit-ranking", $rank) | Out-Null

# configurations measured below: name -> extra args
$configs = [ordered]@{
    "ncmoe48" = @("--n-cpu-moe", "48")
    "ncmoe40" = @("--n-cpu-moe", "40")
    "p3g"     = @("--moe-placement", $p3)
    "p5g"     = @("--moe-placement", $p5)
    "rank"    = @("--moe-placement", $rank, "--moe-vram-margin", "1G")
}
$common = @("-m", $Model, "-ngl", "99", "-c", "$Ctx")

# ---------------------------------------------------------------- 2. correctness
$compare = Join-Path $RepoRoot "tools\expert-trace\compare-logits.py"
$q = { param($s) '"' + $s + '"' }
Invoke-Step "correctness-exact" $Python @($compare, "--llama-debug", (Bin "llama-debug"), "-m", $Model,
    "--out-dir", (Join-Path $ResultsDir "compare-logits"),
    "--run", "ref=--cpu-moe",
    "--run", ("cold=--moe-placement " + (& $q $cold)),
    "--run", "noise=--n-cpu-moe 40",
    "--run", ("p3g=--moe-placement " + (& $q $p3)),
    "--run", ("p5g=--moe-placement " + (& $q $p5)),
    "--run", ("rank=--moe-placement " + (& $q $rank) + " --moe-vram-margin 1G"),
    "--exact", "cold", "--", "-ngl", "99", "-c", "$Ctx") | Out-Null

$wiki = Join-Path $RepoRoot "wikitext-2-raw\wiki.test.raw"
if (-not $SkipKld -and -not (Test-Path $wiki)) {
    Write-Log "wikitext-2 not found, downloading"
    try {
        $zip = Join-Path $RepoRoot "wikitext-2-raw-v1.zip"
        Invoke-WebRequest -Uri "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip" -OutFile $zip -UseBasicParsing
        Expand-Archive -Path $zip -DestinationPath $RepoRoot -Force
    } catch {
        Write-Log ("download failed: " + $_.Exception.Message)
    }
}
if ($SkipKld) {
    Skip-Step "kld" "-SkipKld"
} elseif (-not (Test-Path $wiki)) {
    Skip-Step "kld" "no $wiki"
} else {
    $kldBase = Join-Path $ResultsDir "ref.kld"
    $ppl = @("-m", $Model, "-ngl", "99", "-f", $wiki, "-c", "512", "--chunks", "$KldChunks")
    Invoke-Step "kld-ref-cpumoe" (Bin "llama-perplexity") ($ppl + @("--cpu-moe", "--kl-divergence-base", $kldBase)) | Out-Null
    $kldRuns = [ordered]@{ "noise-ncmoe40" = $configs["ncmoe40"]; "p3g" = $configs["p3g"]; "p5g" = $configs["p5g"]; "rank" = $configs["rank"] }
    foreach ($k in $kldRuns.Keys) {
        Invoke-Step ("kld-" + $k) (Bin "llama-perplexity") ($ppl + $kldRuns[$k] + @("--kl-divergence-base", $kldBase, "--kl-divergence")) | Out-Null
    }
    # where the KLD comes from: at -c 512 the baseline offloads all experts to the GPU (batch >= 32);
    # at -ub 16 nothing is offloaded, so the experts run on the CPU with plain kernels (mmap) or repack (--no-host)
    Invoke-Step "kld-cpu-vs-gpu-kernels" (Bin "llama-perplexity") ($ppl + @("--cpu-moe", "-ub", "16", "--kl-divergence-base", $kldBase, "--kl-divergence")) | Out-Null
    $kldBase16 = Join-Path $ResultsDir "ref-ub16.kld"
    Invoke-Step "kld-ref-cpumoe-ub16" (Bin "llama-perplexity") ($ppl + @("--cpu-moe", "-ub", "16", "--kl-divergence-base", $kldBase16)) | Out-Null
    Invoke-Step "kld-repack-vs-plain" (Bin "llama-perplexity") ($ppl + @("--cpu-moe", "--no-host", "-ub", "16", "--kl-divergence-base", $kldBase16, "--kl-divergence")) | Out-Null
}

# ---------------------------------------------------------------- 3. llama-batched-bench (random tokens)
foreach ($k in $configs.Keys) {
    for ($r = 1; $r -le $Repeats; $r++) {
        Invoke-Step ("bench-{0}-r{1}" -f $k, $r) (Bin "llama-batched-bench") ($common + $configs[$k] + @("-npp", "512", "-ntg", "128", "-npl", "1")) | Out-Null
    }
}

# ---------------------------------------------------------------- 4. real-text decode
if (-not $CodePrompt) {
    $CodePrompt = Join-Path $ResultsDir "prompt_code.txt"
    $src = Get-Content -Path (Join-Path $RepoRoot "src\llama-sampler.cpp") -Raw
    Set-Content -Path $CodePrompt -Value $src.Substring(0, [math]::Min(6000, $src.Length)) -Encoding utf8
}
if (-not $TextPrompt) {
    $held = Join-Path $RepoRoot "heldout_sr.txt"
    if (Test-Path $held) {
        $TextPrompt = $held
    } else {
        Write-Log "no heldout_sr.txt in the repo root: the text prompt is a part of README.md (English, not the profiled workload)"
        $TextPrompt = Join-Path $ResultsDir "prompt_text.txt"
        $src = Get-Content -Path (Join-Path $RepoRoot "README.md") -Raw
        Set-Content -Path $TextPrompt -Value $src.Substring(0, [math]::Min(6000, $src.Length)) -Encoding utf8
    }
}
$prompts = [ordered]@{ "code" = $CodePrompt; "text" = $TextPrompt }
foreach ($pk in $prompts.Keys) {
    foreach ($k in $configs.Keys) {
        Invoke-Step ("decode-{0}-{1}" -f $pk, $k) (Bin "llama-completion") ($common + $configs[$k] + @("-f", $prompts[$pk], "-n", "128", "--temp", "0", "--ignore-eos", "-no-cnv")) | Out-Null
    }
}

# ---------------------------------------------------------------- 4b. placement reports and buffer sizes (-v shows the loader lines)
foreach ($k in @("ncmoe48", "ncmoe40", "p3g", "p5g", "rank")) {
    Invoke-Step ("report-" + $k) (Bin "llama-completion") ($common + $configs[$k] + @("-p", "Hello", "-n", "1", "--temp", "0", "-no-cnv", "-v")) | Out-Null
}

# ---------------------------------------------------------------- 5. llama-bench
$benchCommon = @("-m", $Model, "-ngl", "99", "-ncmoe", "48,44,40,36", "-p", "512", "-n", "128", "-o", "md")
Invoke-Step "llama-bench-kv-f16"  (Bin "llama-bench") ($benchCommon + @("-fa", "off,on", "-ctk", "f16", "-ctv", "f16")) | Out-Null
Write-Log "q8_0 KV cache only with -fa on: a quantized V cache requires flash attention"
Invoke-Step "llama-bench-kv-q8_0" (Bin "llama-bench") ($benchCommon + @("-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0")) | Out-Null

# ---------------------------------------------------------------- 6. decode profile
$timingSummary = Join-Path $RepoRoot "tools\expert-trace\sched-timing-summary.py"
if ($SkipProfile) {
    Skip-Step "profile" "-SkipProfile"
} else {
    foreach ($k in @("ncmoe40", "rank")) {
        # per split and layer: GPU compute, CPU compute, copy + sync (each split is synchronized, see GGML_SCHED_TIMING)
        $l = Invoke-Step ("profile-" + $k) (Bin "llama-completion") ($common + $configs[$k] + @("-f", $CodePrompt, "-n", "64", "--temp", "0", "--ignore-eos", "-no-cnv")) @{ "GGML_SCHED_TIMING" = "2" }
        Invoke-Step ("profile-" + $k + "-summary") $Python @($timingSummary, $l, "--last", "48") | Out-Null
        # CUDA graphs on (default) vs off: no difference means they are not used or do not matter
        Invoke-Step ("profile-" + $k + "-nographs") (Bin "llama-completion") ($common + $configs[$k] + @("-f", $CodePrompt, "-n", "64", "--temp", "0", "--ignore-eos", "-no-cnv")) @{ "GGML_CUDA_DISABLE_GRAPHS" = "1" } | Out-Null
        foreach ($t in $ProfileThreads) {
            Invoke-Step ("profile-{0}-t{1}" -f $k, $t) (Bin "llama-completion") ($common + $configs[$k] + @("-f", $CodePrompt, "-n", "64", "--temp", "0", "--ignore-eos", "-no-cnv", "-t", "$t")) | Out-Null
        }
    }
    # CPU cold FFN of one layer at Qwen3-30B-A3B shapes: fixed cost per layer vs cost per active expert
    Invoke-Step "cold-ffn-bench-plain"  (Bin "llama-cold-ffn-bench") ([string[]] ($ProfileThreads + @(1))) | Out-Null
    Invoke-Step "cold-ffn-bench-repack" (Bin "llama-cold-ffn-bench") ([string[]] (@("repack") + $ProfileThreads + @(1))) | Out-Null
}

# ---------------------------------------------------------------- summary
function Get-Log([string] $step) {
    $r = $script:Results | Where-Object { $_.Step -eq $step } | Select-Object -First 1
    if (-not $r -or -not $r.Log) { return $null }
    return (Join-Path $ResultsDir $r.Log)
}

$sum = @()
$sum += "# MoE placement measurements"
$sum += ""
$sum += ("- date: {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
$sum += "- model: $Model"
$sum += ("- RAM free at start: {0} GB of {1} GB, swap used {2} GB of {3} GB" -f $freeRamGB, $totalRamGB, $swapUsedGB, $swapTotalGB)
if ($nvsmi) { $sum += "- GPU at start: $gpu" }
$sum += "- n_ctx $Ctx, batched-bench repeats $Repeats, KLD chunks $KldChunks"
$sum += ""
$sum += "## Steps"
$sum += ""
$sum += "| step | status | s | log |"
$sum += "|---|---|---|---|"
foreach ($r in $script:Results) { $sum += ("| {0} | {1} | {2} | {3} |" -f $r.Step, $r.Status, $r.Seconds, $r.Log) }

$sum += ""
$sum += "## Correctness (compare-logits)"
$sum += ""
$l = Get-Log "correctness-exact"
if ($l) { $sum += '```'; $sum += (Select-String -Path $l -Pattern "^(-- ubatch|run |ref|cold|noise|p3g|p5g|rank|result)" | ForEach-Object { $_.Line }); $sum += '```' }

$sum += ""
$sum += "## KL-divergence vs --cpu-moe"
$sum += ""
$sum += "| run | Mean KLD | 99.9% KLD | Same top p | PPL(Q)/PPL(base) |"
$sum += "|---|---|---|---|---|"
foreach ($k in @("noise-ncmoe40", "p3g", "p5g", "rank", "cpu-vs-gpu-kernels", "repack-vs-plain")) {
    $l = Get-Log ("kld-" + $k)
    if (-not $l) { continue }
    $get = { param($pat) $m = Select-String -Path $l -Pattern $pat | Select-Object -Last 1; if ($m) { ($m.Line -split ":", 2)[1].Trim() } else { "-" } }
    $sum += ("| {0} | {1} | {2} | {3} | {4} |" -f $k, (& $get "Mean\s+KLD:"), (& $get "99\.9%\s+KLD:"), (& $get "Same top p:"), (& $get "Mean PPL\(Q\)/PPL\(base\)"))
}

$sum += ""
$sum += "## llama-batched-bench (pp512 / tg128, random tokens, median of $Repeats)"
$sum += ""
$sum += "| config | S_PP t/s | S_TG t/s | runs |"
$sum += "|---|---|---|---|"
function Median($xs) { $s = @($xs | Sort-Object); if ($s.Count -eq 0) { return "-" }; return $s[[int][math]::Floor(($s.Count - 1) / 2)] }
foreach ($k in $configs.Keys) {
    $pp = @(); $tg = @()
    for ($r = 1; $r -le $Repeats; $r++) {
        $l = Get-Log ("bench-{0}-r{1}" -f $k, $r)
        if (-not $l) { continue }
        # | PP | TG | B | N_KV | T_PP s | S_PP t/s | T_TG s | S_TG t/s | T s | S t/s |
        $row = Select-String -Path $l -Pattern "^\|\s*512\s*\|\s*128\s*\|" | Select-Object -Last 1
        if ($row) {
            $c = $row.Line.Split("|") | ForEach-Object { $_.Trim() }
            $pp += [double]$c[6]; $tg += [double]$c[8]
        }
    }
    $sum += ("| {0} | {1} | {2} | {3} |" -f $k, (Median $pp), (Median $tg), $pp.Count)
}

$sum += ""
$sum += "## Real-text decode (llama-completion, 128 tokens)"
$sum += ""
$sum += "| prompt | config | prompt eval t/s | eval t/s |"
$sum += "|---|---|---|---|"
foreach ($pk in $prompts.Keys) {
    foreach ($k in $configs.Keys) {
        $l = Get-Log ("decode-{0}-{1}" -f $pk, $k)
        if (-not $l) { continue }
        $pe = Select-String -Path $l -Pattern "prompt eval time.*?([0-9.]+) tokens per second" | Select-Object -Last 1
        $ev = Select-String -Path $l -Pattern "\seval time.*?([0-9.]+) tokens per second" | Where-Object { $_.Line -notmatch "prompt eval" } | Select-Object -Last 1
        $pv = if ($pe) { $pe.Matches[0].Groups[1].Value } else { "-" }
        $evv = if ($ev) { $ev.Matches[0].Groups[1].Value } else { "-" }
        $sum += ("| {0} | {1} | {2} | {3} |" -f $pk, $k, $pv, $evv)
    }
}

$sum += ""
$sum += "## Placement reports and buffer sizes (report-* steps, n_ctx $Ctx)"
foreach ($k in @("ncmoe48", "ncmoe40", "p3g", "p5g", "rank")) {
    $l = Get-Log ("report-" + $k)
    if (-not $l) { continue }
    $sum += ""
    $sum += "### $k"
    $sum += '```'
    $sum += (Select-String -Path $l -Pattern "common_moe_placement_resolve|moe placement|model buffer size|KV buffer size|compute buffer size" |
        Where-Object { $_.Line -notmatch "= +0\.00 MiB" } | ForEach-Object { $_.Line })
    $sum += '```'
}

$sum += ""
$sum += "## Decode profile (GGML_SCHED_TIMING=2, code prompt, 64 tokens; the timing run syncs after every split)"
foreach ($k in @("ncmoe40", "rank")) {
    $l = Get-Log ("profile-" + $k + "-summary")
    if (-not $l) { continue }
    $sum += ""
    $sum += "### $k"
    $sum += '```'
    $sum += (Get-Content -Path $l | Where-Object { $_ -notmatch "^#" -and $_ -ne "" })
    $sum += '```'
    $sum += ""
    $sum += "| run | eval t/s |"
    $sum += "|---|---|"
    $names = @(("profile-" + $k + "-nographs")) + ($ProfileThreads | ForEach-Object { "profile-{0}-t{1}" -f $k, $_ })
    foreach ($n in $names) {
        $l2 = Get-Log $n
        if (-not $l2) { continue }
        $ev = Select-String -Path $l2 -Pattern "\seval time.*?([0-9.]+) tokens per second" | Where-Object { $_.Line -notmatch "prompt eval" } | Select-Object -Last 1
        $sum += ("| {0} | {1} |" -f $n, $(if ($ev) { $ev.Matches[0].Groups[1].Value } else { "-" }))
    }
}
foreach ($k in @("cold-ffn-bench-plain", "cold-ffn-bench-repack")) {
    $l = Get-Log $k
    if (-not $l) { continue }
    $sum += ""
    $sum += "### $k"
    $sum += '```'
    $sum += (Get-Content -Path $l | Where-Object { $_ -notmatch "^#" -and $_ -notmatch "repack tensor" -and $_ -ne "" })
    $sum += '```'
}

$sum += ""
$sum += "## llama-bench"
foreach ($k in @("llama-bench-kv-f16", "llama-bench-kv-q8_0")) {
    $l = Get-Log $k
    if (-not $l) { continue }
    $sum += ""
    $sum += "### $k"
    $sum += ""
    $sum += (Select-String -Path $l -Pattern "^\|" | ForEach-Object { $_.Line })
}

if ($nvsmi) {
    $sum += ""
    $sum += ("- GPU at end: {0}" -f (& nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total --format=csv,noheader 2>&1))
}

$summary = Join-Path $ResultsDir "summary.md"
Set-Content -Path $summary -Value $sum -Encoding utf8
Write-Log "summary: $summary"
