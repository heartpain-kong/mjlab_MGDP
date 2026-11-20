"""Reset-only, cumulative termination-location counters for skill V5."""
import json,time
import torch


class PassageDiagnostics:
  def __init__(self,env,names,kind_of_column):
    self.names=list(names);self.kinds=kind_of_column;self.rows=env.scene.terrain.max_terrain_level
    self.reasons=[name for name,term in env.cfg.terminations.items() if term is not None]
    self.phases=20;shape=(len(names),self.rows,self.phases)
    self.count=torch.zeros(shape,device=env.device,dtype=torch.long)
    self.success=torch.zeros_like(self.count)
    self.terms=torch.zeros((*shape,len(self.reasons)),device=env.device,dtype=torch.long)
    # elapsed seconds, route progress, root X/Y (spawn-relative).
    self.sums=torch.zeros((*shape,4),device=env.device)
    self.labels={name:['整关'] for name in names}
    self.labels['slalom_brick']=[f'路线段{i+1}' for i in range(self.phases)]
    self.labels['high_wall']=['接近第一墙','第一墙附近','墙间','第二墙附近','出口段']
    self.labels['gap_steps']=['上楼梯/入口','第一道沟','下坡','连接平地','上坡','第二道沟','下楼梯/出口']
    self.labels['stepping_stones']=['入口','桩区','出口段']
    g=env.scene.terrain.cfg.terrain_generator
    gap=g.sub_terrains['gap_steps'];cuts=[]
    for row in range(self.rows):
      _,length,width,up,_=gap.gap_parameters((row+.5)/self.rows)
      a=2+3*up+.8;b=a+width;c=b+.8+length;d=c+gap.connector_length;e=d+length+.8
      cuts.append((a,b,c,d,e,e+width))
    self.gap_cuts=torch.tensor(cuts,device=env.device)
    centres=getattr(g.sub_terrains['high_wall'],'wall_centres',(2.3,3.7,5.1))
    self.wall_cuts=torch.tensor((centres[0]-.6,centres[0]+.6,centres[1]-.6,centres[1]+.6),device=env.device)
    self.stone_exits=torch.tensor(env.cfg.terminations['stone_ground_contact'].params['exit_by_level'],device=env.device)

  def record(self,env,ids,passed):
    terrain=env.scene.terrain;kind=self.kinds[terrain.terrain_types[ids]];level=terrain.terrain_levels[ids]
    cmd=env.command_manager.get_term('twist');xy=cmd.local_xy()[ids];x=xy[:,0]+1.
    phase=torch.zeros_like(level)
    for name in ('slalom_brick','high_wall','gap_steps','stepping_stones'):
      mask=kind==self.names.index(name)
      if name=='slalom_brick':value=(cmd.waypoint_index[ids]-1).clamp(0,self.phases-1)
      elif name=='high_wall':value=(x[:,None]>=self.wall_cuts).sum(1)
      elif name=='gap_steps':value=(x[:,None]>=self.gap_cuts[level]).sum(1)
      else:value=(x>2.2).long()+(x>=self.stone_exits[level]).long()
      phase=torch.where(mask,value,phase)
    index=(kind*self.rows+level)*self.phases+phase
    self.count.view(-1).index_add_(0,index,torch.ones_like(index))
    self.success.view(-1).index_add_(0,index,passed.long())
    flags=torch.stack([env.termination_manager.get_term(name)[ids] for name in self.reasons],dim=1).long()
    self.terms.view(-1,len(self.reasons)).index_add_(0,index,flags)
    values=torch.cat((env.episode_length_buf[ids,None]*env.step_dt,cmd.route_state()[3][ids,None],xy),dim=1)
    self.sums.view(-1,4).index_add_(0,index,values)

  def snapshot(self):
    count=self.count.cpu().tolist();success=self.success.cpu().tolist()
    terms=self.terms.cpu().tolist();sums=self.sums.cpu().tolist();items=[]
    for k,name in enumerate(self.names):
      for row in range(self.rows):
        for phase,n in enumerate(count[k][row]):
          if not n:continue
          values=sums[k][row][phase]
          items.append({'terrain':name,'level':row+1,'phase':self.labels[name][phase],
            'episodes':n,'successes':success[k][row][phase],
            'reasons':dict(zip(self.reasons,terms[k][row][phase])),
            'mean_seconds':values[0]/n,'mean_progress_m':values[1]/n,
            'mean_root_xy_m':[v/n for v in values[2:]]})
    return {'version':1,'skill_version':5,'updated':time.time(),'episodes':int(self.count.sum().item()),
      'scope':'累计已完成回合，按终止位置归类；多项终止原因可重叠；阶段成功不等于完整赛道通过',
      'records':items}

  def write(self,path):
    data=self.snapshot();tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False),encoding='utf8');tmp.replace(path)

  def state_dict(self):
    return {'names':self.names,'reasons':self.reasons,
      **{name:getattr(self,name).detach().cpu().clone() for name in ('count','success','terms','sums')}}

  def load_state_dict(self,state):
    if state.get('names')!=self.names or state.get('reasons')!=self.reasons:raise ValueError('Diagnostic counters differ')
    values={}
    for name in ('count','success','terms','sums'):
      current=getattr(self,name);value=torch.as_tensor(state[name],device=current.device)
      if value.shape!=current.shape or not torch.isfinite(value).all():raise ValueError('Invalid diagnostic counters')
      if name!='sums' and ((value<0).any() or (value!=value.long()).any()):raise ValueError('Invalid diagnostic counts')
      values[name]=value
    if (values['success']>values['count']).any() or (values['terms']>values['count'][...,None]).any():
      raise ValueError('Diagnostic events exceed completed episodes')
    for name,value in values.items():getattr(self,name).copy_(value)
