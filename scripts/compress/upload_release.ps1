# =============================================================================
#  Mage-VL 压缩产物 → GitHub Release 上传脚本（v2：断点续传 + 卡死检测 + 逐文件重试）
#
#  为什么要有 v2：2026-09-30 实测，链路退化到 0.2 MiB/s 时 `gh release upload`
#  会「挂着不动」——一个 1.85 GiB 的分片传了 21 分钟只读了 253 MiB，既不报错也不结束。
#  只靠「退出码重试」救不了，必须能识别"卡住"并主动掐掉重来。
#
#  v2 相比 v1：
#    1) 卡死检测：每 PollSec 秒采样 gh 进程的读盘字节；连续 StallMinutes 分钟无增长
#       即判为卡死 → 掐掉该进程 → 记一次失败 → 重试。
#    2) 成功判定以**远端资产**为准（存在且字节数一致），不再只信退出码。
#    3) 实时进度行：打印「已读取 X / 总 Y MiB」，一眼看出是否卡住。
#    4) -Only 只传指定包；-Rounds 控制多轮次数。
#
#  仍然是「可断点续传」：每轮先按「文件名 + 字节大小」比对远端资产，传完的自动跳过，
#  所以中途 Ctrl+C、断电、重跑都不会重复传。
#
#  ⚠️ 不要与 `git push` 同时跑：链路被占满会让 push 直接超时（实测 300s 失败）。
#  ⚠️ 挂机前把电源计划里「睡眠」设为「从不」；脚本也会用 SetThreadExecutionState
#     尝试阻止系统休眠（只对本次进程生效，失败不影响上传）。
#
#  用法：
#    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\compress\upload_release.ps1
#    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\compress\upload_release.ps1 -Only int4_all
# =============================================================================
param(
    [string]$Repo = 'wangcangxing/home-fall-detection',
    [string]$Root = 'E:\MageVL\compress',
    [string[]]$Only = @(),
    [int]$Rounds = 20,
    [int]$MaxRetryPerFile = 5,
    [int]$StallMinutes = 6,
    [int]$PollSec = 20,
    [int]$RetrySleepSec = 15
)

$ErrorActionPreference = 'Continue'

# ---- 各包 → Release 的对应关系（顺序 = 优先级，先传推荐包）-------------------
$allPackages = @(
    @{ tag = 'v0.2.0-int4-g64';        dir = 'int4_g64' },
    @{ tag = 'v0.2.0-pruned-int4-all'; dir = 'pruned_int4_all' },
    @{ tag = 'v0.2.0-int4-all';        dir = 'int4_all' },
    @{ tag = 'v0.2.0-pruned-int4';     dir = 'pruned_int4' },
    @{ tag = 'v0.2.0-int8-g64';        dir = 'int8_g64' },
    @{ tag = 'v0.2.0-prune-wanda75';   dir = 'pruneL30_wanda75' }
)
if ($Only.Count -gt 0) {
    $packages = @($allPackages | Where-Object { $Only -contains $_.dir })
    if ($packages.Count -eq 0) { Write-Host "错误：-Only 没匹配到任何包（可选：$(($allPackages | ForEach-Object { $_.dir }) -join ', ')）"; exit 2 }
} else {
    $packages = $allPackages
}

$logFile = Join-Path $Root ("upload-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmm'))
$script:t0 = Get-Date
$tmpErr = Join-Path $env:TEMP 'gh_upload_err.txt'
$tmpOut = Join-Path $env:TEMP 'gh_upload_out.txt'

function Say([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    Write-Host $line
    try { Add-Content -LiteralPath $logFile -Value $line -Encoding UTF8 } catch { }
}

# 返回 hashtable；查询失败返回 $null（调用方据此跳过该包而不是误判为"全都没传"）
function Get-RemoteAssets([string]$tag) {
    $raw = gh release view $tag --repo $Repo --json assets 2>$null
    if (-not $raw) { return $null }
    $map = @{}
    try {
        $obj = $raw | ConvertFrom-Json
        foreach ($a in $obj.assets) { $map[[string]$a.name] = [int64]$a.size }
    } catch { return $null }
    return $map
}

# ---- 尝试阻止系统休眠（只影响本进程）----------------------------------------
$keepAwakeOk = $false
try {
    if (-not ('Win32.Power' -as [type])) {
        Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
    }
    # ⚠️ 必须用十进制：PowerShell 把 0x80000001 当 Int32 负数，转 UInt32 会抛异常（实测踩过）
    # 2147483649 = ES_CONTINUOUS(0x80000000) | ES_SYSTEM_REQUIRED(0x1)
    $r = [Win32.Power]::SetThreadExecutionState([uint32]2147483649)
    $keepAwakeOk = ($r -ne 0)
} catch { }

Say "================================================================"
Say "Mage-VL 压缩产物上传 v2"
Say "  仓库     : $Repo"
Say "  日志     : $logFile"
Say ("  本次包   : {0}" -f (($packages | ForEach-Object { $_.dir }) -join ', '))
Say ("  阻止休眠 : {0}" -f $(if ($keepAwakeOk) { '已启用（仅本次进程）' } else { '未启用 —— 请自行确认电源计划不睡眠' }))
Say ("  卡死判定 : 连续 {0} 分钟无读盘进展即掐掉重来" -f $StallMinutes)
Say "================================================================"

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) { Say "错误：找不到 gh CLI"; exit 2 }
if ((gh auth status 2>&1 | Out-String) -notmatch 'Logged in') { Say "错误：gh 未登录，请先 gh auth login"; exit 2 }

# 上传单个文件；返回 $true = 远端已存在且大小一致
function Send-One([string]$tag, [System.IO.FileInfo]$f) {
    $totalMiB = [math]::Round($f.Length / 1MB, 1)
    $p = Start-Process -FilePath 'gh' -PassThru -NoNewWindow `
        -ArgumentList @('release', 'upload', $tag, $f.FullName, '--repo', $Repo, '--clobber') `
        -RedirectStandardError $tmpErr -RedirectStandardOutput $tmpOut
    $lastBytes = 0; $lastChange = Get-Date
    while (-not $p.HasExited) {
        Start-Sleep -Seconds $PollSec
        $info = Get-CimInstance Win32_Process -Filter "ProcessId=$($p.Id)" -ErrorAction SilentlyContinue
        if (-not $info) { break }
        $read = [int64]$info.ReadTransferCount
        if ($read -gt $lastBytes) {
            $lastBytes = $read; $lastChange = Get-Date
            $pct = if ($f.Length -gt 0) { [math]::Round($read / $f.Length * 100, 1) } else { 0 }
            Say ("      进行中 {0,8:N1} / {1:N1} MiB ({2}%)" -f ($read / 1MB), $totalMiB, $pct)
        } elseif (((Get-Date) - $lastChange).TotalMinutes -ge $StallMinutes) {
            Say ("      卡死：{0} 分钟无进展（已读 {1:N1} MiB / {2:N1} MiB），掐掉重来" -f $StallMinutes, ($read / 1MB), $totalMiB)
            Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
            break
        }
    }
    if (-not $p.HasExited) { try { $p.Kill() } catch { } }

    $remote = Get-RemoteAssets $tag
    if ($null -ne $remote -and $remote.ContainsKey([string]$f.Name) -and $remote[[string]$f.Name] -eq [int64]$f.Length) { return $true }
    return $false
}

$allDone = $false
for ($round = 1; $round -le $Rounds; $round++) {
    Say ""
    Say "########## 第 $round / $Rounds 轮 ##########"
    $pendingAll = 0
    foreach ($p in $packages) {
        $dir = Join-Path $Root $p.dir
        if (-not (Test-Path -LiteralPath $dir)) { Say "跳过（目录不存在）：$dir"; continue }
        $remote = Get-RemoteAssets $p.tag
        if ($null -eq $remote) { Say "$($p.tag) : 查询远端失败（网络），本轮跳过"; $pendingAll++; continue }

        $files = @(Get-ChildItem -LiteralPath $dir -File | Sort-Object Length -Descending)
        $pending = @($files | Where-Object { $remote[[string]$_.Name] -ne [int64]$_.Length })
        $bytes = ($pending | Measure-Object Length -Sum).Sum
        if ($null -eq $bytes) { $bytes = 0 }
        Say ("{0,-24} 待传 {1,2}/{2,2} 个 / {3,6:N2} GiB" -f $p.tag, $pending.Count, $files.Count, ($bytes / 1GB))
        $pendingAll += $pending.Count

        foreach ($f in $pending) {
            $ok = $false
            for ($i = 1; $i -le $MaxRetryPerFile; $i++) {
                Say ("    上传 [{0}/{1}] {2}  ({3:N1} MiB)" -f $i, $MaxRetryPerFile, $f.Name, ($f.Length / 1MB))
                $t1 = Get-Date
                if (Send-One $p.tag $f) {
                    $sec = ((Get-Date) - $t1).TotalSeconds
                    $rate = if ($sec -gt 0.1) { ($f.Length / 1MB) / $sec } else { 0 }
                    Say ("    OK   {0,9:N1} MiB / {1,6:N0}s = {2,5:N2} MiB/s" -f ($f.Length / 1MB), $sec, $rate)
                    $ok = $true
                    break
                }
                Say ("    失败（第 {0}/{1} 次），{2}s 后重试" -f $i, $MaxRetryPerFile, $RetrySleepSec)
                Start-Sleep -Seconds $RetrySleepSec
            }
            if (-not $ok) { Say ("    放弃（留到下一轮）：{0}" -f $f.Name) }
        }
    }
    if ($pendingAll -eq 0) { $allDone = $true; break }
    Say ("第 $round 轮结束，仍有 {0} 个文件待传" -f $pendingAll)
}

Say ""
Say "================ 最终核对 ================"
$grandLocal = 0; $grandRemote = 0; $missing = 0
foreach ($p in $packages) {
    $dir = Join-Path $Root $p.dir
    $local = @(Get-ChildItem -LiteralPath $dir -File)
    $remote = Get-RemoteAssets $p.tag
    $lBytes = ($local | Measure-Object Length -Sum).Sum
    $rBytes = 0; $miss = 0
    foreach ($f in $local) {
        if ($null -ne $remote -and $remote.ContainsKey([string]$f.Name) -and $remote[[string]$f.Name] -eq [int64]$f.Length) { $rBytes += $f.Length } else { $miss++ }
    }
    $grandLocal += $lBytes; $grandRemote += $rBytes; $missing += $miss
    Say ("{0,-24} 本地 {1,5:N2} GiB / 已传 {2,5:N2} GiB / 缺 {3} 个" -f $p.tag, ($lBytes / 1GB), ($rBytes / 1GB), $miss)
}
Say ("合计：本地 {0:N2} GiB / 已传 {1:N2} GiB / 缺失 {2} 个文件" -f ($grandLocal / 1GB), ($grandRemote / 1GB), $missing)
if ($missing -eq 0) {
    gh release edit v0.2.0-int4-g64 --repo $Repo --latest *> $null
    Say "全部完成；推荐包 v0.2.0-int4-g64 已设为 latest"
} else {
    Say "仍未传完 —— 重新运行本脚本即可接着传（已传的会自动跳过）"
}
Say ("总用时 {0:N1} 分钟" -f ((Get-Date) - $script:t0).TotalMinutes)
if ($missing -eq 0) { exit 0 } else { exit 1 }
