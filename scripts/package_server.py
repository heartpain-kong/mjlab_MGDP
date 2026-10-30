"""Export the current worktree (including uncommitted code) without run data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

EXCLUDE_ROOTS={'.git','.vscode','archive','deploy','simulate','logs','outputs','wandb','.venv'}
EXCLUDE_PARTS={'__pycache__','.pytest_cache','.cache','node_modules','.venv','.ssh'}


def allowed(path):
  path=Path(path)
  if path.is_absolute() or '..' in path.parts:return False
  if path.parts[0] in EXCLUDE_ROOTS or EXCLUDE_PARTS.intersection(path.parts):return False
  if path.name.startswith('.env') or path.suffix.lower() in {'.pt','.onnx','.pyc','.key','.pem','.bak','.tmp','.log'}:return False
  return True


def digest(path):
  h=hashlib.sha256()
  with Path(path).open('rb') as stream:
    for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
  return h.hexdigest()


def verify(root):
  root=Path(root).resolve()
  data=json.loads((root/'SOURCE_MANIFEST.json').read_text('utf8'))
  for relative,expected in data['files'].items():
    target=root/relative
    if not target.resolve().is_relative_to(root) or not target.is_file() or digest(target)!=expected['sha256']:
      raise ValueError(f'Export verification failed: {relative}')
  return len(data['files'])


def export(source,destination,archive):
  source=Path(source).resolve();destination=Path(destination).resolve();archive=Path(archive).resolve()
  if destination==source or destination.is_relative_to(source):raise ValueError('Export must be outside the source checkout')
  if archive.exists():raise FileExistsError(f'Archive already exists: {archive}')
  paths=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=source).split(b'\0')
  files={}
  destination.mkdir(parents=True,exist_ok=False)
  for raw in sorted(set(paths)):
    if not raw:continue
    relative=raw.decode('utf8')
    if not allowed(relative):continue
    original=source/relative
    if not original.exists():continue  # deleted in the working tree
    if original.is_symlink() or not original.resolve().is_relative_to(source):raise ValueError(f'Unsafe source path: {relative}')
    target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
    if target.suffix=='.sh':target.write_bytes(original.read_bytes().replace(b'\r\n',b'\n'))
    else:shutil.copy2(original,target)
    files[relative]={'sha256':digest(target),'bytes':target.stat().st_size}
  manifest={'version':1,'created_utc':datetime.now(timezone.utc).isoformat(),
            'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
            'includes_uncommitted_changes':bool(subprocess.check_output(['git','status','--porcelain'],cwd=source)),
            'excluded_roots':sorted(EXCLUDE_ROOTS),'files':files}
  (destination/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
  verify(destination)
  archive.parent.mkdir(parents=True,exist_ok=True)
  with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
    for target in sorted(destination.rglob('*')):
      if target.is_file():
        name=f'{destination.name}/{target.relative_to(destination).as_posix()}'
        bundle.write(target,name)
        if target.suffix=='.sh':
          info=bundle.getinfo(name);info.create_system=3;info.external_attr=0o100755<<16
  checksum=digest(archive)
  archive.with_suffix(archive.suffix+'.sha256').write_text(f'{checksum}  {archive.name}\n',encoding='ascii')
  return {'directory':str(destination),'archive':str(archive),'files':len(files),
          'bytes':sum(item['bytes'] for item in files.values()),'archive_bytes':archive.stat().st_size,'sha256':checksum}


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1])
  parser.add_argument('--destination',type=Path)
  parser.add_argument('--archive',type=Path)
  parser.add_argument('--verify',type=Path,help='Verify an already exported checkout')
  args=parser.parse_args()
  if args.verify:print(f'PASS: {verify(args.verify)} exported files verified');return
  if not args.destination or not args.archive:parser.error('--destination and --archive are required')
  print(json.dumps(export(args.source,args.destination,args.archive),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
