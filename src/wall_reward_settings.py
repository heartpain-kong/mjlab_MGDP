"""Task-scoped wall settings shared by the stdlib workbench and training."""
import ast
from copy import deepcopy
import hashlib,json,math
from pathlib import Path
from pprint import pformat
import shutil

SOURCE_RELATIVE=Path('src/wall_reward_settings.py')
WALL_REWARD_DEFAULTS={
  'progress_gate':{'enabled':True,'progress_reference_mps':.1,'min_command_mps':.1},
  'wall_waiting':{'enabled':True,'weight':-1.,'approach_m':.75,'grace_s':.75,'ramp_s':1.5,
                  'max_rate':2.,'min_progress_m':.02,'min_command_mps':.1},
  'wall_crossing':{'weight':2.,'root_clearance_m':.35,'foot_clearance_m':.03},
  'passage_progress':{'weight':1.},
  'passage_success':{'weight':5.},
  'no_passage_progress':{'enabled':True,'timeout_s':4.,'min_progress_m':.05,'min_command_mps':.1},
}
WALL_REWARD_OVERRIDES={}


def number(label,unit,minimum,maximum,step=.01):
  return {'label':label,'type':'number','unit':unit,'min':minimum,'max':maximum,'step':step}


SWITCH={'label':'启用','type':'boolean'}
WALL_REWARD_GROUPS={
  'progress_gate':{'label':'基础正奖励进展门控','description':'跟踪、姿态、步态和抬脚的正奖励随新增最远距离发放。',
    'fields':{'enabled':SWITCH,'progress_reference_mps':number('完整奖励所需前进速度','m/s',.001,5),
              'min_command_mps':number('移动指令阈值','m/s',.001,5)}},
  'wall_waiting':{'label':'墙前等待惩罚','description':'靠近下一道未完成的墙，无有效进展时递增扣分。',
    'fields':{'enabled':SWITCH,'weight':number('代价权重','',-10000,0,.1),
      'approach_m':number('墙前检测范围','m',.01,5),'grace_s':number('准备宽限时间','s',0,30,.05),
      'ramp_s':number('升到最大代价所需时间','s',.01,60,.05),
      'max_rate':number('最大代价率（乘权重前）','分/s',0,100,.1),
      'min_progress_m':number('刷新等待所需新增进展','m',.001,1,.005),
      'min_command_mps':number('移动指令阈值','m/s',.001,5)}},
  'wall_crossing':{'label':'每墙一次奖励','description':'四脚和根部满足越墙条件后发放，每墙每回合一次。',
    'fields':{'weight':number('每墙奖励','分',0,10000,.1),
      'root_clearance_m':number('根部越过墙后缘距离','m',0,2),
      'foot_clearance_m':number('四脚越过墙后缘距离','m',0,1)}},
  'passage_progress':{'label':'新增路线距离奖励','description':'只奖励新的最远路线距离，来回摆动不重复发放。',
    'fields':{'weight':number('距离奖励权重','',0,10000,.1)}},
  'passage_success':{'label':'出口成功奖励','description':'到达最终出口且没有同时失败时发放。',
    'fields':{'weight':number('出口奖励','分',0,10000,.1)}},
  'no_passage_progress':{'label':'无进展终止','description':'训练中持续没有足够新进展时结束回合；回放模式不启用此终止。',
    'fields':{'enabled':SWITCH,'timeout_s':number('无进展时限','s',.1,120,.1),
      'min_progress_m':number('刷新时限所需新增进展','m',.001,1,.005),
      'min_command_mps':number('移动指令阈值','m/s',.001,5)}},
}


def validate_wall_reward_settings(settings):
  if not isinstance(settings,dict) or set(settings)!=set(WALL_REWARD_GROUPS):
    raise ValueError('高墙规则的设置分组不完整或包含未知分组')
  result={}
  for group,schema in WALL_REWARD_GROUPS.items():
    values=settings[group]
    if not isinstance(values,dict) or set(values)!=set(schema['fields']):
      raise ValueError(f"{schema['label']}的参数不完整或包含未知参数")
    result[group]={}
    for key,field in schema['fields'].items():
      value=values[key]
      if field['type']=='boolean':
        if type(value) is not bool:raise ValueError(f"{field['label']}必须是布尔值")
      else:
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
          raise ValueError(f"{field['label']}必须是有限数字")
        if not field['min']<=value<=field['max']:
          raise ValueError(f"{field['label']}需在{field['min']}到{field['max']}之间")
        value=float(value)
      result[group][key]=value
  return result


def _read(path):
  source=path.read_text(encoding='utf-8-sig');tree=ast.parse(source,filename=str(path));items={}
  for node in tree.body:
    if isinstance(node,ast.Assign):
      for target in node.targets:
        if isinstance(target,ast.Name) and target.id in ('WALL_REWARD_DEFAULTS','WALL_REWARD_OVERRIDES'):
          items[target.id]=(ast.literal_eval(node.value),node)
  defaults=validate_wall_reward_settings(items['WALL_REWARD_DEFAULTS'][0])
  overrides=items['WALL_REWARD_OVERRIDES'][0]
  if not isinstance(overrides,dict) or set(overrides)-set(defaults):raise ValueError('高墙规则覆盖配置无效')
  settings=deepcopy(defaults)
  for group,values in overrides.items():
    if not isinstance(values,dict) or set(values)-set(defaults[group]):raise ValueError('高墙规则包含未知参数')
    settings[group].update(values)
  return source,items,defaults,validate_wall_reward_settings(settings)


def load_wall_reward_settings(path=None):
  return _read(Path(path) if path else Path(__file__))[3]


def wall_reward_contract(settings):
  # Identical defaults retain the V5 checkpoint contract. A changed objective
  # must not inherit the old critic or mastery statistics through full resume.
  if settings==WALL_REWARD_DEFAULTS:return 1
  return '1:'+hashlib.sha256(json.dumps(settings,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def read_wall_reward_settings(root):
  path=Path(root)/SOURCE_RELATIVE;_,_,defaults,settings=_read(path)
  return {'available':True,'settings':settings,'defaults':defaults,'groups':WALL_REWARD_GROUPS,
          'source':str(path),'changed':settings!=defaults,
          'contract_changed':wall_reward_contract(settings)!=1}


def save_wall_reward_settings(root,settings):
  settings=validate_wall_reward_settings(settings);path=Path(root)/SOURCE_RELATIVE
  source,items,defaults,_=_read(path)
  overrides={group:{key:value for key,value in values.items() if value!=defaults[group][key]}
             for group,values in settings.items()}
  overrides={group:values for group,values in overrides.items() if values}
  node=items['WALL_REWARD_OVERRIDES'][1];lines=source.splitlines(keepends=True)
  block='WALL_REWARD_OVERRIDES='+pformat(overrides,width=100,sort_dicts=False)+'\n'
  updated=''.join(lines[:node.lineno-1])+block+''.join(lines[node.end_lineno:])
  ast.parse(updated)
  backup=path.with_suffix('.py.bak');shutil.copy2(path,backup)
  temp=path.with_suffix('.py.tmp');temp.write_text(updated,encoding='utf-8',newline='\n');temp.replace(path)
  return {**read_wall_reward_settings(root),'backup':str(backup)}
