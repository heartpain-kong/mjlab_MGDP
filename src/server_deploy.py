"""Manage only this checkout's web processes, without stopping training jobs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import time
from urllib.error import URLError
from urllib.request import urlopen

from filelock import FileLock
import psutil


def specifications(root, python, *, ui_port=8765, kline_port=8766, portal_port=8770, tb_port=6006):
  # Do not resolve a venv's bin/python symlink to the base interpreter.
  root=Path(root).resolve(); python=os.path.abspath(os.fspath(python))
  ports=[ui_port,kline_port,portal_port,tb_port]
  if len(set(ports))!=4 or any(type(p) is not int or not 1024<=p<=65535 for p in ports):
    raise ValueError("Four distinct ports in 1024-65535 are required")
  base=[python,str(root/'tools'/'lgdog.py'),'--python',python]
  return {
    'dashboard':{'port':ui_port,'argv':[*base,'ui','--port',str(ui_port),'--no-browser']},
    'kline':{'port':kline_port,'argv':[python,'-m','src.kline','--port',str(kline_port),'--no-browser']},
    'tensorboard':{'port':tb_port,'argv':[python,'-m','tensorboard.main','--logdir',str(root/'logs/rsl_rl'),
                                      '--host','127.0.0.1','--port',str(tb_port)]},
    # This supervisor owns the children, so the portal must not start duplicates.
    'portal':{'port':portal_port,'argv':[*base,'web','--port',str(portal_port),'--host','127.0.0.1',
                                      '--dashboard-port',str(ui_port),'--kline-port',str(kline_port),
                                      '--tensorboard-port',str(tb_port),
                                      '--no-autostart','--no-browser']},
  }


class ServerServices:
  def __init__(self,root,python,**ports):
    self.root=Path(root).resolve()
    self.specs=specifications(self.root,python,**ports)
    self.directory=self.root/'logs/server'
    self.path=self.directory/'services.json'
    self.children={}

  def load(self):
    if not self.path.exists():return {}
    state=json.loads(self.path.read_text('utf8'))
    if state.get('root')!=str(self.root):raise ValueError('Service registry belongs to another checkout')
    return state.get('services',{})

  def save(self,services):
    self.directory.mkdir(parents=True,exist_ok=True)
    temp=self.path.with_suffix('.tmp')
    temp.write_text(json.dumps({'root':str(self.root),'services':services},indent=2),encoding='utf8')
    temp.replace(self.path)

  def owned(self,record):
    try:
      proc=psutil.Process(record['pid'])
      if (abs(proc.create_time()-record['created'])>.01 or proc.cmdline()!=record['cmdline']
          or Path(proc.cwd()).resolve()!=self.root):return None
      return proc if proc.is_running() and proc.status()!=psutil.STATUS_ZOMBIE else None
    except (KeyError,TypeError,ValueError,psutil.Error):return None

  @staticmethod
  def listening(port):
    try:
      with socket.create_connection(('127.0.0.1',port),timeout=.2):return True
    except OSError:return False

  @staticmethod
  def ready(port):
    try:
      with urlopen(f'http://127.0.0.1:{port}/',timeout=1) as response:
        return response.status==200
    except (URLError,OSError,TimeoutError):return False

  def status(self,records=None):
    records=self.load() if records is None else records
    result=[]
    for name,spec in self.specs.items():
      record=records.get(name,{})
      port=record.get('port',spec['port'])
      owned=self.owned(record)
      state=('running' if self.ready(port) else 'starting') if owned else ('external' if self.listening(port) else 'stopped')
      result.append({'name':name,'status':state,'pid':record.get('pid') if owned else None,
                     'port':port,'url':f'http://127.0.0.1:{port}/','log':str(self.directory/f'{name}.log')})
    return result

  def start(self,timeout=45.):
    records=self.load()
    for name,spec in self.specs.items():
      old=records.get(name,{})
      if self.owned(old):
        if old['port']!=spec['port']:raise RuntimeError(f'{name} already managed on port {old["port"]}; stop it before changing ports')
      elif self.listening(spec['port']):
        raise RuntimeError(f'Port {spec["port"]} is occupied by another service; left unchanged')
    self.directory.mkdir(parents=True,exist_ok=True)
    started=[]
    try:
      for name,spec in self.specs.items():
        if self.owned(records.get(name,{})):continue
        env=dict(os.environ,PYTHONPATH=str(self.root)+(os.pathsep+os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else ''),
                 PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8')
        kwargs={'cwd':str(self.root),'env':env,'stdin':subprocess.DEVNULL}
        if os.name=='nt':kwargs['creationflags']=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NO_WINDOW
        else:kwargs['start_new_session']=True
        with (self.directory/f'{name}.log').open('a',encoding='utf8') as log:
          child=subprocess.Popen(spec['argv'],stdout=log,stderr=subprocess.STDOUT,**kwargs)
        self.children[name]=child
        proc=psutil.Process(child.pid)
        record={'pid':child.pid,'created':proc.create_time(),'cmdline':proc.cmdline(),
                'port':spec['port']}
        records[name]=record;started.append(name);self.save(records)
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
          if child.poll() is not None:raise RuntimeError(f'{name} exited {child.returncode}; read {self.directory/name}.log')
          if self.ready(spec['port']):break
          time.sleep(.25)
        else:raise RuntimeError(f'{name} did not become ready; read {self.directory/name}.log')
    except BaseException:
      # Roll back only processes created in this invocation, never old services.
      for name in reversed(started):self.stop_one(records[name])
      self.save(records)
      raise
    return self.status(records)

  def stop_one(self,record):
    proc=self.owned(record)
    if proc is None:return
    proc.terminate()
    try:proc.wait(timeout=8)
    except psutil.TimeoutExpired:proc.kill();proc.wait(timeout=5)
    for child in self.children.values():
      if child.pid==record['pid']:child.poll()

  def stop(self):
    records=self.load()
    # No recursive child kill: independent training jobs keep running.
    for name in reversed(list(records)):self.stop_one(records[name])
    return self.status(records)


def manage(root,python,action,**ports):
  manager=ServerServices(root,python,**ports)
  manager.directory.mkdir(parents=True,exist_ok=True)
  with FileLock(str(manager.directory/'control.lock'),timeout=5):
    result=getattr(manager,action)()
  print(json.dumps(result,ensure_ascii=False,indent=2))
  return 0
