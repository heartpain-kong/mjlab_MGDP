param(
  [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9][A-Za-z0-9.-]*$')][string]$ServerHost,
  [ValidateRange(1,65535)][int]$Port=22,
  [ValidatePattern('^[a-z_][a-z0-9_-]*$')][string]$UserName='root',
  [ValidatePattern('^/[A-Za-z0-9_./-]+$')][string]$RemoteBase='/root/gpufree-data',
  [string]$Archive='',
  [switch]$SkipSystemDeps,
  [switch]$Smoke
)
$ErrorActionPreference='Stop'
if ($RemoteBase -match '(^|/)\.\.(/|$)' -or $RemoteBase.TrimEnd('/') -eq '') { throw 'RemoteBase must be a safe absolute data directory' }
if (!$Archive) {
  $Archive=Join-Path $PSScriptRoot 'unitree_rl_mjlab_lgdog_server.zip'
  if (!(Test-Path -LiteralPath $Archive -PathType Leaf)) { $Archive=Join-Path (Split-Path $PSScriptRoot -Parent) 'unitree_rl_mjlab_lgdog_server.zip' }
}
$Archive=(Resolve-Path -LiteralPath $Archive).Path
$Target="$UserName@$ServerHost"
$Stamp=(Get-Date -Format 'yyyyMMdd_HHmmss')+'_'+([guid]::NewGuid().ToString('N').Substring(0,6))
$RemoteBase=$RemoteBase.TrimEnd('/')
# Each upload is a new release: never overwrite old source, logs or checkpoints.
$Release="$RemoteBase/lgdog-releases/$Stamp"
$RemoteZip="$Release/source.zip"
& ssh -p $Port $Target "mkdir -p -- '$RemoteBase/lgdog-releases' && mkdir -- '$Release'"
if ($LASTEXITCODE -ne 0) { throw 'Remote directory creation failed' }
& scp -P $Port $Archive "${Target}:$RemoteZip"
if ($LASTEXITCODE -ne 0) { throw 'Upload failed' }
$Hash=(Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
$Flags=''
if ($SkipSystemDeps) { $Flags+=' --skip-system-deps' }
if ($Smoke) { $Flags+=' --smoke' }
$Command="printf '%s  %s\n' '$Hash' '$RemoteZip' | sha256sum -c - && python3 -m zipfile -e '$RemoteZip' '$Release' && cd '$Release/unitree_rl_mjlab_lgdog' && bash deploy_server.sh$Flags"
& ssh -p $Port $Target $Command
if ($LASTEXITCODE -ne 0) { throw "Deployment failed; uploaded release remains at $Release. Existing services were not stopped." }
Write-Host "Remote project: $Release/unitree_rl_mjlab_lgdog"
Write-Host "Open an SSH tunnel in another PowerShell window:"
Write-Host "ssh -N -p $Port -L 8765:127.0.0.1:8765 -L 8766:127.0.0.1:8766 -L 8770:127.0.0.1:8770 -L 6006:127.0.0.1:6006 $Target"
