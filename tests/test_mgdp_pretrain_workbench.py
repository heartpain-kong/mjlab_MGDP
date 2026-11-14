"""The wall pretrain button must produce a command accepted by its real CLI."""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import shlex, sys, tempfile, unittest

from src.config.mgdp_pretrain import WALL_PRETRAIN, WALL_PRETRAIN_CHECKPOINT, pretrain_resources
from src.dashboard import routes
from src.dashboard.catalog import TASKS

ROOT=Path(__file__).resolve().parents[1]
TASK='LgDog1-Race-MGDP-WallTraining'


class WallPretrainWorkbenchTests(unittest.TestCase):
  def context(self, root=ROOT):
    return NS(root=root, python=Path(sys.executable), jobs=Mock())

  def test_button_command_parses_with_real_script_and_wall_defaults(self):
    from scripts.pretrain_mgdp import parse_args
    ctx=self.context()
    status,payload=routes.pretrain(ctx,{'task':'race_mgdp_wall'})
    argv,mode,alias=ctx.jobs.launch.call_args.args
    self.assertEqual((status,mode,alias),(200,'pretrain','race_mgdp_wall'))
    with patch('sys.argv',argv[1:]):args=parse_args()
    self.assertEqual(args.task,TASK)
    self.assertEqual((args.num_envs,args.terrain_rows,args.terrain_cols,args.batch_size),(64,8,10,64))
    self.assertEqual(Path(args.out),ROOT/WALL_PRETRAIN_CHECKPOINT)
    self.assertEqual(payload['output'],args.out)

  def test_old_invalid_grid_is_rejected_before_starting_a_process(self):
    ctx=self.context()
    with self.assertRaisesRegex(ValueError,'8行'):
      routes.pretrain(ctx,{'task':'race_mgdp_wall','terrain_rows':12})
    with self.assertRaises(ValueError):
      routes.pretrain(ctx,{'task':'race_mgdp_wall','terrain_cols':9})
    ctx.jobs.launch.assert_not_called()

  def test_help_hint_and_cli_defaults_use_same_task_and_output(self):
    from scripts.pretrain_mgdp import parse_args
    spec=TASKS['race_mgdp_wall'];command=shlex.split(spec.pretrain_command)
    with patch('sys.argv',['pretrain',*command[3:]]):args=parse_args()
    self.assertEqual((args.task,args.terrain_rows,args.out),(TASK,8,WALL_PRETRAIN_CHECKPOINT))
    with patch('sys.argv',['pretrain','--task',TASK]):defaults=parse_args()
    self.assertEqual((defaults.terrain_rows,defaults.out),(8,WALL_PRETRAIN_CHECKPOINT))
    self.assertEqual(pretrain_resources('race_mgdp_wall'),WALL_PRETRAIN)

  def test_wall_factory_and_catalog_use_isolated_encoder(self):
    from src.tasks.velocity.config.lgdog1.mgdp_env_cfg import lgdog1_wall_training_env_cfg,lgdog1_race_mgdp_env_cfg
    cfg=lgdog1_wall_training_env_cfg()
    self.assertEqual(cfg.observations['perception'].terms['depth_feature'].params['checkpoint'],WALL_PRETRAIN_CHECKPOINT)
    old=lgdog1_race_mgdp_env_cfg()
    self.assertTrue(old.observations['perception'].terms['depth_feature'].params['checkpoint'].endswith('mgdp_race_v3.pt'))
    self.assertNotEqual(TASKS['race_mgdp']['pretrain_checkpoint'],TASKS['race_mgdp_wall']['pretrain_checkpoint'])

  def test_old_encoder_does_not_make_new_course_ready(self):
    from src.dashboard.commands import build_command
    with tempfile.TemporaryDirectory() as tmp:
      root=Path(tmp);old=root/TASKS['race_mgdp']['pretrain_checkpoint']
      old.parent.mkdir(parents=True);old.write_bytes(b'old encoder must remain')
      with self.assertRaisesRegex(ValueError,'预训练'):
        build_command(root,sys.executable,{'mode':'train','task':'race_mgdp_wall'})
      (root/WALL_PRETRAIN_CHECKPOINT).write_bytes(b'new encoder fixture')
      self.assertIn(TASK,build_command(root,sys.executable,{'mode':'train','task':'race_mgdp_wall'}))
      self.assertEqual(old.read_bytes(),b'old encoder must remain')


if __name__=='__main__':unittest.main()
