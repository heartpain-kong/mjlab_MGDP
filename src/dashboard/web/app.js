"use strict";
const $ = id => document.getElementById(id);
let state = null, mode = "train", models = [], rewardData = null, selectedJob = null, previewVersion = 0, timer = null;
let lastCommand = "";
const modeNames = {train:"训练", play:"Play 回放", sim:"Sim2Sim", rewards:"查看奖励", curves:"训练曲线"};
const statuses = {running:"运行中", stopping:"停止中", completed:"已完成", failed:"失败", stopped:"已停止", unknown:"历史·状态未知"};
const rewardLabels = {track_linear_velocity:"线速度跟踪",track_angular_velocity:"角速度跟踪",body_orientation_l2:"机身姿态",pose:"关节姿态",foot_gait:"足端步态",body_ang_vel:"机身角速度",angular_momentum:"角动量",joint_acc_l2:"关节加速度",joint_pos_limits:"关节限位",action_rate_l2:"动作变化率",foot_clearance:"抬脚高度",foot_slip:"足端滑移",soft_landing:"落足冲击",feet_stumble:"足端绊倒",feet_air_time:"足端滞空",leg_contact_force:"腿部接触力",stand_still:"静止站立",is_terminated:"终止惩罚"};

async function api(path, body) {
  const options = body === undefined ? {} : {method:"POST", headers:{"Content-Type":"application/json", "X-LgDog-Token":state.token}, body:JSON.stringify(body)};
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "请求失败");
  return data;
}
function toast(text) { $("toast").textContent=text; $("toast").classList.remove("hidden"); clearTimeout(timer); timer=setTimeout(()=>$("toast").classList.add("hidden"),4500); }
async function copy(text) { try { await navigator.clipboard.writeText(text); toast("已复制"); } catch { toast("浏览器不允许复制，请手动选择文本"); } }
function task() { return state.tasks.find(t=>t.alias===$("task").value); }
function matchingTasks() {
  const algorithm=$("algorithm")?.value, scene=$("scene")?.value;
  return state.tasks.filter(item=>(!algorithm||item.algorithm===algorithm)&&(!scene||item.scene===scene));
}
function renderModelSceneSelectors(preferredAlias=$("task")?.value||"standard") {
  if(!state?.algorithms||!state?.scenes)return;
  const current=state.tasks.find(item=>item.alias===preferredAlias)||state.tasks[0];
  const algorithm=$("algorithm"), scene=$("scene"), taskSelect=$("task");
  const algorithmValue=algorithm.value||current.algorithm||"him";
  algorithm.replaceChildren(...state.algorithms.map(item=>new Option(`${item.label} · ${item.description}`,item.alias)));
  algorithm.value=state.algorithms.some(item=>item.alias===algorithmValue)?algorithmValue:(current.algorithm||"him");
  const availableScenes=state.scenes.filter(item=>state.tasks.some(taskItem=>taskItem.algorithm===algorithm.value&&taskItem.scene===item.alias));
  const sceneValue=current.algorithm===algorithm.value?current.scene:(availableScenes[0]?.alias||"rough");
  scene.replaceChildren(...availableScenes.map(item=>new Option(`${item.label} · ${item.description}`,item.alias)));
  scene.value=availableScenes.some(item=>item.alias===sceneValue)?sceneValue:(availableScenes[0]?.alias||"");
  const matches=matchingTasks();
  taskSelect.replaceChildren(...matches.map(item=>new Option(item.label,item.alias)));
  const defaults={"him|flat":"him_flat","him|rough":"standard","him|rough_gentle":"him","ppo|flat":"flat","ppo|rough":"blind","ppo|rough_gentle":"gentle","ppo|rough_scan":"rough"};
  const fallback=defaults[`${algorithm.value}|${scene.value}`];
  const selected=matches.some(item=>item.alias===preferredAlias)?preferredAlias:(matches.some(item=>item.alias===fallback)?fallback:(matches[0]?.alias||""));
  taskSelect.value=selected;
  return selected;
}
function syncSelectorsFromTask() {
  const current=task();
  if(!current)return;
  if($("algorithm"))$("algorithm").value=current.algorithm||"ppo";
  if($("scene"))$("scene").value=current.scene||"rough";
}
function collect() {
  const model = $("custom-model").value.trim() || $("model").value;
  return {mode, task:$("task").value, envs:$(mode==="play"?"play-envs":"train-envs").value,
    iterations:$("iterations").value, seed:$("seed").value, run_name:$("run-name").value,
    terrain_num_rows:$("terrain-num-rows")?.value || "",
    terrain_difficulty_max:$("terrain-difficulty-max")?.value || "",
    resume:$("resume").checked, checkpoint:model, onnx:model, viewer:$("viewer").value,
    map:$("map").value, map_file:$("map-file")?.value||"", duration:$("duration").value,
    vx:$("vx").value, vy:$("vy").value, wz:$("wz").value,
    spawn_x:$("spawn-x")?.value||"0", spawn_y:$("spawn-y").value, spawn_yaw:$("spawn-yaw")?.value||"0",
    speed_x:$("speed-x")?.value||"1", speed_y:$("speed-y")?.value||"0.5", speed_yaw:$("speed-yaw")?.value||"1",
    action_clip:$("action-clip")?.value||"1", heading_hold:$("heading-hold")?.checked??true,
    check_obs:$("check-obs")?.checked??false, show_viewer:$("show-viewer").checked,
    keyboard:$("keyboard").checked, term:$("term")?.value || ""};
}
function updateTask() {
  const t=task();
  if(!t)return;
  $("summary-task").textContent=t.label;
  $("summary-obs").textContent=`${t.obs} 维`;
  $("task-description").textContent=t.description;
  $("critic-chip").textContent=`critic ${t.critic} 维`;
  $("profile-chip").textContent=`reward: ${t.profile}`;
  $("experiment-path").textContent=`logs/rsl_rl/${t.experiment}`;
  $("model-hint").textContent=t.resume_sources?.length?"当前任务模型和兼容阶段模型都会显示；Flat-HIM checkpoint 可用于 Rough-HIM 续训。":(t.shared?"Flat/Rough 共用目录，请手动选择与任务匹配的模型。":"模型来自当前任务的独立实验目录。");
}
function renderMapInfo(){
  const item=state?.sim2sim_maps?.find(mapItem=>mapItem.alias===$("map")?.value);
  if(item&&$("sim-map-info"))$("sim-map-info").textContent=`${item.description} · 约 ${item.geoms} 个地形 geom · 推荐出生 y=${item.spawn_y} m`;
}
function renderSimMaps(){
  if(!state?.sim2sim_maps||!$("map"))return;
  const previous=$("map").value||"flat";
  $("map").replaceChildren(...state.sim2sim_maps.map(item=>new Option(`${item.label} · ${item.geoms} geom`,item.alias)));
  $("map").value=state.sim2sim_maps.some(item=>item.alias===previous)?previous:"flat";
  renderMapInfo();
}
function fillModels() {
  const previous=$("model").value;
  const kind=mode==="sim"?"onnx":"checkpoint";
  const matching=models.filter(m=>m.kind===kind);
  $("model").replaceChildren(new Option(matching.length?"选择模型文件":"该任务尚无对应模型", ""));
  for(const m of matching) $("model").add(new Option(`${m.source_label?`${m.source_label} · `:""}${m.run} / ${m.name}`,m.path));
  if(matching.some(m=>m.path===previous)) $("model").value=previous;
  else if(matching.length&&!task().shared) $("model").value=matching[0].path;
  $("summary-models").textContent=models.length;
}
async function loadModels() { const alias=$("task").value; const data=await api(`/api/models?task=${encodeURIComponent(alias)}`); if(alias!==$("task").value)return; models=data.models;fillModels(); }
function metricValue(data,name){return data?.metrics?.[name]?.value;}
function renderActionMetrics(data){
  const rawMax=metricValue(data,"raw_action_abs_max"), saturation=metricValue(data,"action_saturation_fraction"), policyStd=metricValue(data,"policy_mean_std"), reward=metricValue(data,"mean_reward");
  $("metric-raw-max").textContent=rawMax==null?"—":rawMax.toFixed(2);
  $("metric-saturation").textContent=saturation==null?"—":`${(saturation*100).toFixed(1)}%`;
  $("metric-policy-std").textContent=policyStd==null?"—":policyStd.toFixed(2);
  $("metric-reward").textContent=reward==null?"—":reward.toFixed(2);
  const status=$("action-health-status");
  status.classList.toggle("warning",Boolean(data?.warning));
  status.textContent=data?.warning?"需要检查":data?.available?"正常":"等待数据";
  if(data?.warning){
    $("action-health-detail").textContent=`${data.warning} · ${data.run||"未命名 run"} · step ${data.metrics?.raw_action_abs_max?.step??"—"}`;
  }else if(data?.available){
    $("action-health-detail").textContent=`${data.run||"未命名 run"} · 最新 step ${data.metrics?.raw_action_abs_max?.step??data.metrics?.mean_reward?.step??"—"} · 原始动作在裁剪前统计`;
  }else{
    $("action-health-detail").textContent=data?.warning||"训练启动后自动读取最新 TensorBoard 记录。";
  }
}
async function loadActionMetrics(){
  if(!state||!$("task")?.value)return;
  const alias=$("task").value;
  try{const data=await api(`/api/metrics?task=${encodeURIComponent(alias)}`);if(alias===$("task").value)renderActionMetrics(data);}
  catch(error){renderActionMetrics({available:false,warning:error.message,metrics:{}});}
}
function showMode() {
  document.querySelectorAll("[data-section]").forEach(el=>{
    const inactive=el.dataset.section!==mode;
    el.classList.toggle("hidden",inactive);
    el.querySelectorAll("input,select,button").forEach(control=>{control.disabled=inactive;});
  });
  document.querySelectorAll("[data-mode]").forEach(el=>{const active=el.dataset.mode===mode;el.classList.toggle("selected",active);el.setAttribute("aria-selected",String(active));});
  $("model-field").classList.toggle("hidden",!(mode==="play"||mode==="sim"||(mode==="train"&&$("resume").checked)));
  $("mode-tag").textContent=modeNames[mode];
  $("launch").textContent={train:"启动训练 ↗",play:"开始回放 ↗",sim:"开始验证 ↗",rewards:"查看奖励 ↗"}[mode];
  $("launch-hint").textContent={train:"按显存调整环境数；开始后查看下方日志。",play:"使用 checkpoint；不会自动刷新已启动的策略。",sim:"使用 ONNX；开启窗口时会显示在运行服务的电脑上。",rewards:"读取配置，无需 GPU。首次导入依赖可能需要等待。"}[mode];
  fillModels();
  if(mode==="rewards") loadRewards();
}
function renderRewards(){
  const list=$("reward-list"); list.replaceChildren();
  if(!rewardData?.rewards?.length){const empty=document.createElement("div");empty.className="empty";empty.textContent="没有读取到 reward。";list.append(empty);return;}
  for(const reward of rewardData.rewards){
    const row=document.createElement("div");row.className="reward-row"+(reward.changed?" changed":"");row.dataset.reward=reward.name;row.dataset.search=`${reward.name} ${rewardLabels[reward.name]||""}`.toLowerCase();row.title="双击查看实现";row.addEventListener("dblclick",()=>showRewardDetail(reward.name));
    const enabled=document.createElement("input");enabled.type="checkbox";enabled.checked=reward.enabled;enabled.title="启用 reward";
    const name=document.createElement("label");name.textContent=rewardLabels[reward.name]||reward.name;const code=document.createElement("small");code.textContent=`${reward.name}${reward.changed?" · 已覆盖":""}`;name.append(code);
    const weight=document.createElement("input");weight.type="number";weight.step="any";weight.value=reward.weight;weight.setAttribute("aria-label",`${reward.name} 权重`);weight.title=`基线：${reward.baseline}`;
    const params=document.createElement("textarea");params.value=JSON.stringify(reward.params||{});params.setAttribute("aria-label",`${reward.name} 参数 JSON`);params.placeholder="参数 JSON（例如 {\"target_height\":0.1}）";
    row.append(enabled,name,weight,params);list.append(row);
  }
  $("reward-source").textContent=`${rewardData.profile} · ${rewardData.source}`;
  filterRewards();
}
function filterRewards(){const query=($("reward-filter")?.value||"").trim().toLowerCase();document.querySelectorAll(".reward-row").forEach(row=>{row.classList.toggle("hidden",Boolean(query&&!row.dataset.search.includes(query)));});}
function resetRewards(){if(!rewardData?.rewards)return;document.querySelectorAll(".reward-row").forEach(row=>{const reward=rewardData.rewards.find(item=>item.name===row.dataset.reward);const inputs=row.querySelectorAll("input,textarea");inputs[0].checked=reward.baseline!==0;inputs[1].value=reward.baseline;inputs[2].value=JSON.stringify(reward.baseline_params||{});row.classList.remove("changed");});toast("已恢复本方案基线，点击保存后才会写入文件");}
async function loadRewards(){try{const alias=$("task").value;rewardData=await api(`/api/rewards?task=${encodeURIComponent(alias)}`);if(alias===$("task").value)renderRewards();}catch(error){$("reward-list").textContent=error.message;toast(error.message);}}
async function showRewardDetail(name){try{const data=await api(`/api/reward-detail?task=${encodeURIComponent($("task").value)}&name=${encodeURIComponent(name)}`);$("document-title").textContent=`奖励实现 · ${data.name}`;$("document-path").textContent=`${data.source}${data.line?`:${data.line}`:""} · ${data.binding}`;$("document-text").textContent=data.text;$("document-dialog").showModal();}catch(error){toast(error.message);}}
function collectRewards(){
  const changes=[];
  document.querySelectorAll(".reward-row").forEach(row=>{const inputs=row.querySelectorAll("input,textarea");let params={};try{params=JSON.parse(inputs[2].value||"{}");}catch{throw new Error(`${row.dataset.reward} 的参数不是有效 JSON`);}changes.push({name:row.dataset.reward,weight:inputs[0].checked?inputs[1].value:0,params});});
  return changes;
}
async function preview() {
  if(!state)return;
  const version=++previewVersion;
  try {
    const data=await api("/api/preview",collect());
    if(version!==previewVersion)return;
    lastCommand=data.command; $("command").textContent=data.command; $("preview-error").classList.add("hidden"); $("launch").disabled=false;
  } catch(error) {
    if(version!==previewVersion)return;
    lastCommand="";$("command").textContent="准备好参数和模型后，这里会显示实际执行命令。";
    $("preview-error").textContent=error.message;$("preview-error").classList.remove("hidden");$("launch").disabled=true;
  }
}
let previewTimer;
function queuePreview() {clearTimeout(previewTimer);previewTimer=setTimeout(preview,180);}
function renderJobs() {
  const list=$("jobs");
  $("job-count").textContent=state.jobs.length;
  $("summary-running").textContent=state.jobs.filter(j=>j.can_stop).length;
  if(!state.jobs.length)return;
  if(!selectedJob||!state.jobs.some(j=>j.id===selectedJob))selectedJob=state.jobs[0].id;
  list.replaceChildren();
  for(const job of state.jobs) {
    const row=document.createElement("button"); row.type="button";row.className="job"+(job.id===selectedJob?" selected":"");
    const top=document.createElement("div");top.className="job-top";
    const title=document.createElement("span");title.textContent=modeNames[job.mode]||job.mode;
    const status=document.createElement("span");status.className=`status ${job.status}`;status.textContent=statuses[job.status]||job.status;
    top.append(title,status);
    const detail=document.createElement("small");detail.textContent=`${job.task} · PID ${job.pid}\n${new Date(job.created).toLocaleString()}`;
    row.append(top,detail);row.addEventListener("click",()=>{selectedJob=job.id;renderJobs();loadLog();});list.append(row);
  }
  const active=state.jobs.find(j=>j.id===selectedJob);
  $("stop").disabled=!active?.can_stop;
  $("log-title").textContent=active?`${modeNames[active.mode]||active.mode} · ${active.id}`:"运行日志";
}
async function loadLog() {
  if(!selectedJob)return;
  const id=selectedJob;
  try {const data=await api(`/api/log?id=${encodeURIComponent(id)}`);if(id!==selectedJob)return;
    if($("logs").textContent!==data.text){$("logs").textContent=data.text||"进程启动中，等待输出…";if($("auto-scroll").checked)$("logs").scrollTop=$("logs").scrollHeight;}
  }catch(error){$("logs").textContent=error.message;}
}
async function refresh(all=false) {
  try {
    const data=await api("/api/state");state=data;
    $("connection").lastChild.textContent=" 本地服务运行中";
    renderJobs();if(all)await loadModels();await loadActionMetrics();await loadLog();
  }catch(error){$("connection").textContent="服务未连接";if(all)toast(error.message);}
}
async function showDocument(id) {
  try {const data=await api(`/api/document?id=${encodeURIComponent(id)}`);$("document-title").textContent=data.title;$("document-path").textContent=data.path;$("document-text").textContent=data.text;$("document-dialog").showModal();}catch(error){toast(error.message);}
}
function formatTime(value){ if(!value)return "—"; const date=new Date(value); return Number.isNaN(date.getTime())?String(value):date.toLocaleString(); }
function historySection(title,count){
  const section=document.createElement("section");section.className="history-section";
  const heading=document.createElement("h3");heading.textContent=`${title} · ${count}`;section.append(heading);
  if(!count){const empty=document.createElement("div");empty.className="empty";empty.textContent="还没有记录。";section.append(empty);}
  return section;
}
function historyRewardRow(edit){
  const row=document.createElement("div");row.className="history-row";
  const top=document.createElement("div");top.className="history-row-top";
  const when=document.createElement("span");when.textContent=formatTime(edit.time);
  const who=document.createElement("span");who.className="history-chip";who.textContent=`${edit.task||"?"} · ${edit.profile||"?"}`;
  top.append(when,who);row.append(top);
  for(const change of edit.changes||[]){
    const line=document.createElement("code");line.className="history-change";
    const [before,after]=change.weight||[];
    line.textContent=`${change.name}: ${before} → ${after}`;
    const [oldParams,newParams]=change.params||[];
    if(JSON.stringify(oldParams||{})!==JSON.stringify(newParams||{})){
      line.textContent+=` · params ${JSON.stringify(oldParams||{})} → ${JSON.stringify(newParams||{})}`;
    }
    row.append(line);
  }
  if(edit.backup){const note=document.createElement("small");note.className="hint";note.textContent=`备份：${edit.backup}`;row.append(note);}
  return row;
}
function historyRunRow(run){
  const row=document.createElement("div");row.className="history-row";
  const top=document.createElement("div");top.className="history-row-top";
  const when=document.createElement("span");when.textContent=formatTime(run.started||run.ended);
  const status=document.createElement("span");status.className=`history-chip status-${run.status||"unknown"}`;status.textContent=statuses[run.status]||run.status||"未知";
  top.append(when,status);row.append(top);
  const what=document.createElement("code");what.className="history-change";
  const bits=[modeNames[run.mode]||run.mode,run.task];
  if(run.run_name)bits.push(`run=${run.run_name}`);
  if(run.envs)bits.push(`envs=${run.envs}`);
  if(run.iterations)bits.push(`iter=${run.iterations}`);
  if(run.seed!==undefined&&run.seed!==null)bits.push(`seed=${run.seed}`);
  if(run.map)bits.push(`map=${run.map}`);
  if(run.map_file)bits.push(`xml=${run.map_file}`);
  if(run.duration_s!=null)bits.push(`${run.duration_s}s`);
  what.textContent=bits.filter(Boolean).join(" · ");row.append(what);
  if(run.log){const log=document.createElement("small");log.className="hint";log.textContent=run.log;row.append(log);}
  return row;
}
function renderHistory(data){
  const body=$("history-body");body.replaceChildren();
  const edits=data.reward_edits||[], runs=data.runs||[];
  const editSection=historySection("奖励修改",edits.length);
  for(const edit of edits)editSection.append(historyRewardRow(edit));
  const runSection=historySection("运行记录",runs.length);
  for(const run of runs)runSection.append(historyRunRow(run));
  body.append(editSection,runSection);
  $("copy-history").dataset.text=historyText(data);
}
function historyText(data){
  const lines=[];
  for(const edit of data.reward_edits||[]){
    lines.push(`[奖励修改] ${edit.time} ${edit.task} / ${edit.profile}`);
    for(const change of edit.changes||[]){
      const [before,after]=change.weight||[];
      lines.push(`  ${change.name}: ${before} -> ${after}`);
    }
  }
  for(const run of data.runs||[]){
    lines.push(`[运行] ${run.started||run.ended} ${modeNames[run.mode]||run.mode} ${run.task} ${statuses[run.status]||run.status||""}${run.run_name?` run=${run.run_name}`:""}${run.map?` map=${run.map}`:""}${run.map_file?` xml=${run.map_file}`:""}`);
  }
  return lines.join("\n");
}
async function showHistory(){
  try{
    const data=await api("/api/history?limit=80");
    renderHistory(data);
    if(!$("history-dialog").open)$("history-dialog").showModal();
  }catch(error){toast(error.message);}
}
async function start() {
  try {
    state=await api("/api/state");
    renderModelSceneSelectors("standard");
    renderSimMaps();
    $("copy-root").textContent=state.root;$("python-path").textContent=state.python;
    updateTask();await loadModels();showMode();await preview();renderJobs();await loadActionMetrics();await loadLog();
    document.querySelectorAll("[data-mode]").forEach(button=>button.addEventListener("click",()=>{mode=button.dataset.mode;showMode();preview();}));
    const changeTask=async()=>{$("custom-model").value="";$("run-name").value="";updateTask();await loadModels();showMode();await loadActionMetrics();preview();};
    $("task").addEventListener("change",async()=>{syncSelectorsFromTask();await changeTask();});
    $("algorithm").addEventListener("change",async()=>{const alias=renderModelSceneSelectors($("task").value);if(alias)await changeTask();});
    $("scene").addEventListener("change",async()=>{const alias=renderModelSceneSelectors($("task").value);if(alias)await changeTask();});
    $("map").addEventListener("change",()=>{renderMapInfo();queuePreview();});
    $("resume").addEventListener("change",()=>{showMode();preview();});
    $("run-form").addEventListener("input",queuePreview);$("run-form").addEventListener("change",queuePreview);
    document.querySelectorAll("[data-preset]").forEach(button=>button.addEventListener("click",()=>{const smoke=button.dataset.preset==="smoke";$("train-envs").value=smoke?64:1024;$("iterations").value=smoke?2:16000;preview();}));
    $("run-form").addEventListener("submit",async event=>{event.preventDefault();$("launch").disabled=true;try{const data=await api("/api/run",collect());selectedJob=data.job.id;toast("已启动，查看下方真实日志");await refresh();}catch(error){toast(error.message);}finally{preview();}});
    $("save-rewards").addEventListener("click",async()=>{try{$("save-rewards").disabled=true;const data=await api("/api/rewards/save",{task:$('task').value,changes:collectRewards()});rewardData=data;renderRewards();toast("奖励修改已保存，重启训练后生效；原文件已备份");}catch(error){toast(error.message);}finally{$("save-rewards").disabled=false;}});
    $("reload-rewards").addEventListener("click",loadRewards);
    $("reset-rewards").addEventListener("click",resetRewards);
    $("reward-filter").addEventListener("input",filterRewards);
    $("stop").addEventListener("click",async()=>{try{$("stop").disabled=true;await api("/api/stop",{id:selectedJob});await refresh();toast("停止指令已执行，已保存的 checkpoint 保留");}catch(error){toast(error.message);}});
    $("curves").addEventListener("click",async()=>{try{const data=await api("/api/run",{mode:"curves",task:$("task").value,port:6006});selectedJob=data.job.id;await refresh();toast("TensorBoard 正在启动，稍后点击曲线页面");}catch(error){toast(error.message);}});
    $("refresh").addEventListener("click",async()=>{await refresh(true);preview();});
    $("copy-command").addEventListener("click",()=>lastCommand?copy(lastCommand):toast("当前没有可复制的命令"));
    $("copy-root").addEventListener("click",()=>copy(state.root));
    document.querySelectorAll("[data-doc]").forEach(button=>button.addEventListener("click",()=>showDocument(button.dataset.doc)));
    $("close-document").addEventListener("click",()=>$("document-dialog").close());
    $("copy-document-path").addEventListener("click",()=>copy($("document-path").textContent));
    $("history").addEventListener("click",showHistory);
    $("history-refresh").addEventListener("click",showHistory);
    $("close-history").addEventListener("click",()=>$("history-dialog").close());
    $("copy-history").addEventListener("click",()=>{const text=$("copy-history").dataset.text||"";text?copy(text):toast("还没有可复制的记录");});
    setInterval(()=>refresh(),1800);
  }catch(error){toast(error.message);$("command").textContent="面板初始化失败，请检查服务终端。";}
}
start();
