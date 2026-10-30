"""CPU checks for deployment packaging and ownership-safe service control."""

import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import psutil

from scripts.package_server import export,verify
from src.server_deploy import ServerServices,specifications


def free_port():
  with socket.socket() as sock:sock.bind(('127.0.0.1',0));return sock.getsockname()[1]


class PackagingTests(unittest.TestCase):
  def test_worktree_export_includes_uncommitted_code_but_not_models_or_secrets(self):
    with tempfile.TemporaryDirectory() as td:
      parent=Path(td);source=parent/'source';source.mkdir()
      subprocess.run(['git','init','-q'],cwd=source,check=True)
      (source/'main.py').write_text('old\n')
      subprocess.run(['git','add','main.py'],cwd=source,check=True)
      subprocess.run(['git','-c','user.name=Deployment Test','-c','user.email=test@localhost','-c','commit.gpgsign=false','commit','-qm','fixture'],cwd=source,check=True)
      (source/'main.py').write_text('current uncommitted\n')
      (source/'deploy_server.sh').write_bytes(b'#!/bin/bash\r\necho ready\r\n')
      (source/'model_100.pt').write_bytes(b'not a model')
      (source/'.env').write_text('EXAMPLE=not-exported')
      (source/'logs').mkdir();(source/'logs/event').write_text('not-exported')
      destination=parent/'export';archive=parent/'server.zip'
      result=export(source,destination,archive)
      self.assertEqual((destination/'main.py').read_text(),'current uncommitted\n')
      self.assertNotIn(b'\r', (destination/'deploy_server.sh').read_bytes())
      self.assertFalse((destination/'logs').exists());self.assertFalse((destination/'.env').exists())
      self.assertFalse((destination/'model_100.pt').exists());self.assertEqual(verify(destination),result['files'])
      self.assertTrue(json.loads((destination/'SOURCE_MANIFEST.json').read_text())['includes_uncommitted_changes'])
      with zipfile.ZipFile(archive) as bundle:
        self.assertEqual(bundle.testzip(),None)
        self.assertTrue(bundle.getinfo('export/deploy_server.sh').external_attr>>16 & 0o111)
      (destination/'main.py').write_text('changed')
      with self.assertRaisesRegex(ValueError,'verification failed'):verify(destination)
      with self.assertRaises(FileExistsError):export(source,destination,archive)


class ServerControlTests(unittest.TestCase):
  def test_specs_preserve_venv_path_and_forward_custom_ports_to_portal(self):
    with tempfile.TemporaryDirectory() as td:
      python=Path(td)/'venv/bin/python'
      specs=specifications(td,python,ui_port=18001,kline_port=18002,portal_port=18003,tb_port=18004)
      self.assertIn(str(python.absolute()),specs['dashboard']['argv'])
      self.assertIn('--no-autostart',specs['portal']['argv'])
      self.assertIn('18002',specs['portal']['argv'])
      self.assertEqual(specs['tensorboard']['argv'][-3:],['127.0.0.1','--port','18004'])
      with self.assertRaises(ValueError):specifications(td,sys.executable,ui_port=8766)

  def test_managed_http_service_is_idempotent_and_only_owned_pid_is_stopped(self):
    with tempfile.TemporaryDirectory() as td:
      port=free_port();manager=ServerServices(td,sys.executable)
      # Bypass Windows venv launcher stubs in the child fixture.
      python=psutil.Process().exe()
      manager.specs={'fixture':{'port':port,'argv':[python,'-m','http.server',str(port),'--bind','127.0.0.1']}}
      try:
        first=manager.start(timeout=8)[0]
        self.assertEqual(first['status'],'running')
        self.assertEqual(manager.start(timeout=8)[0]['pid'],first['pid'])
        forged={**manager.load()['fixture'],'created':0.}
        with patch.object(psutil.Process,'terminate') as terminate:
          manager.stop_one(forged);terminate.assert_not_called()
      finally:manager.stop()
      self.assertEqual(manager.status()[0]['status'],'stopped')

  def test_external_port_is_not_killed_or_replaced(self):
    with tempfile.TemporaryDirectory() as td,socket.socket() as listener:
      listener.bind(('127.0.0.1',0));listener.listen()
      manager=ServerServices(td,sys.executable)
      manager.specs={'external':{'port':listener.getsockname()[1],'argv':['must-not-run']}}
      with self.assertRaisesRegex(RuntimeError,'occupied'):
        manager.start()
      self.assertFalse(manager.path.exists())


if __name__=='__main__':unittest.main()
