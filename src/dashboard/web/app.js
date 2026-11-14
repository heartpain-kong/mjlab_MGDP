"use strict";
const $ = id => document.getElementById(id);
let state = null, mode = "train", models = [], rewardData = null, selectedJob = null, previewVersion = 0, timer = null, runNameVersion = 0;
let lastCommand = "";
let wallRewardData = null;
const {modeNames, statuses, rewardLabels, metricValue, healthText} = window.LgDogFormat;
const pendingStops = new Set();
let pendingDelete = null, deletingJobs = false;
let pretrainStarting = false;
let presetTaskAlias = null;

// 面板每次重启都会换一个 token（server.py 用 secrets 生成）。页面只在加载时取一次，
// 所以重启过服务后所有 POST 都会被 403 拒绝、启动按钮变灰。这里静默重取 token 再重试一次。
let tokenRefresh = null;
async function refreshToken() {
  if (!tokenRefresh) {
    tokenRefresh = (async () => {
      const response = await fetch("/api/state", {cache:"no-store"});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "无法重新连接本地面板服务");
      state = data;
      $("connection").lastChild.textContent = " 本地服务运行中";
      return data.token;
    })().finally(() => { tokenRefresh = null; });
  }
  return tokenRefresh;
}
async function api(path, body, retry = true) {
  const options = body === undefined ? {} : {method:"POST", headers:{"Content-Type":"application/json", "X-LgDog-Token":state.token}, body:JSON.stringify(body)};
  const response = await fetch(path, options);
  let data;
  try { data = await response.json(); } catch { data = {error:`服务返回 HTTP ${response.status}`}; }
  if (response.status === 403 && retry) {
    await refreshToken();
    toast("面板服务重启过，已自动重连并重试");
    return api(path, body, false);
  }
  if (!response.ok) {
    if (response.status === 403) throw new Error("面板服务已重启，请刷新页面（Ctrl+R）后重试");
    throw new Error(data.error || "请求失败");
  }
  return data;
}
function toast(text) { $("toast").textContent=text; $("toast").classList.remove("hidden"); clearTimeout(timer); timer=setTimeout(()=>$("toast").classList.add("hidden"),4500); }
async function copy(text) { try { await navigator.clipboard.writeText(text); toast("已复制"); } catch { toast("浏览器不允许复制，请手动选择文本"); } }
function task() { return state?.tasks?.find(t=>t.alias===$("task")?.value)||null; }
function matchingTasks() {
  const algorithm=$("algorithm")?.value, scene=$("scene")?.value;
  return state.tasks.filter(item=>(!algorithm||item.algorithm===algorithm)&&(!scene||item.scene===scene));
}
function fillTaskSelect(preferred){
  const select=$("task"); if(!select||!state?.tasks)return null;
  const groups=new Map();
  for(const item of state.tasks){
    const key=item.algorithm||"ppo";
    if(!groups.has(key)){
      const group=document.createElement("optgroup");
      group.label=state.algorithms?.find(a=>a.alias===key)?.label||key;
      groups.set(key,group);
    }
    groups.get(key).append(new Option(item.label,item.alias));
  }
  select.replaceChildren(...groups.values());
  const selected=state.tasks.some(item=>item.alias===preferred)?preferred:(state.tasks[0]?.alias||"");
  select.value=selected;
  select.disabled=false;
  return selected;
}
function renderModelSceneSelectors(preferredAlias=$("task")?.value||"standard") {
  if(!state?.algorithms||!state?.scenes)return;
  const current=state.tasks.find(item=>item.alias===preferredAlias)||state.tasks[0];
  const algorithm=$("algorithm"), scene=$("scene");
  algorithm.replaceChildren(...state.algorithms.map(item=>new Option(`${item.label} · ${item.description}`,item.alias)));
  scene.replaceChildren(...state.scenes.map(item=>new Option(`${item.label} · ${item.description}`,item.alias)));
  algorithm.value=current.algorithm||"ppo";
  scene.value=current.scene||"rough";
  // The task list is the primary control; algorithm/scene are derived context
  // so a task such as mgdp can never be hidden by a stale filter.
  algorithm.disabled=true; scene.disabled=true;
  algorithm.title="由所选任务自动决定";
  scene.title="由所选任务自动决定";
  return fillTaskSelect(preferredAlias);
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
    terrain_num_rows:$("terrain-num-rows")?.disabled?"":$("terrain-num-rows")?.value || "",
    terrain_num_cols:$("terrain-num-cols")?.disabled?"":$("terrain-num-cols")?.value || "",
    terrain_difficulty_max:$("terrain-difficulty-max")?.disabled?"":$("terrain-difficulty-max")?.value || "",
    terrain_curriculum:$("terrain-curriculum")?.value||"manual",
    lp_temperature:$("lp-temperature")?.value||"2",
    lp_uniform_mix:$("lp-uniform-mix")?.value??"0.5",
    lp_min_episodes:$("lp-min-episodes")?.value||"32",
    lp_update_interval_s:$("lp-update-interval")?.value||"100",
    lp_progress_alpha:$("lp-progress-alpha")?.value||"0.25",
    lp_probability_alpha:$("lp-probability-alpha")?.value||"0.2",
    lp_max_task_probability:$("lp-max-task-probability")?.value||"0.25",
    velocity_min_abs:task()?.directed_passage?"":$("velocity-min-abs")?.value || "",
    velocity_max_abs:task()?.directed_passage?"":$("velocity-max-abs")?.value || "",
    enable_nan_guard:$("enable-nan-guard").checked,
    resume:$("resume").checked, checkpoint:model, onnx:model, viewer:$("viewer").value,
    map:$("map").value, map_file:$("map-file")?.value||"",
    vx:$("vx").value, vy:$("vy").value, wz:$("wz").value,
    spawn_x:$("spawn-x")?.value||"0", spawn_y:$("spawn-y").value, spawn_yaw:$("spawn-yaw")?.value||"0",
    speed_x:$("speed-x")?.value||"1", speed_y:$("speed-y")?.value||"0.5", speed_yaw:$("speed-yaw")?.value||"1",
    action_clip:$("action-clip")?.value||"1", heading_hold:$("heading-hold")?.checked??true,
    check_obs:$("check-obs")?.checked??false, show_viewer:$("show-viewer").checked,
    keyboard:$("keyboard").checked, term:$("term")?.value || ""};
}
function updateTask() {
  const t=task();
  updateCurriculumOptions();
  if(!t){
    $("summary-task").textContent="未注册任务";
    $("summary-obs").textContent="—";
    $("task-description").textContent=`${$("algorithm")?.value||"算法"} + ${$("scene")?.value||"场景"} 没有对应的训练任务，请更换组合。`;
    $("critic-chip").textContent="";$("profile-chip").textContent="";$("experiment-path").textContent="—";
    $("task-capabilities")?.replaceChildren();
    $("task-prereq")?.classList.add("hidden");
    $("task-prereq-actions")?.classList.add("hidden");
    $("model-hint").textContent="任务由算法和场景决定；当前组合没有可用模型。";
    updateRunNameHint();
    return;
  }
  updateTrainingPresets(t);
  $("summary-task").textContent=t.label;
  $("summary-obs").textContent=`${t.obs} 维`;
  $("task-description").textContent=t.description;
  $("critic-chip").textContent=`critic ${t.critic} 维`;
  $("profile-chip").textContent=`reward: ${t.profile}`;
  renderTaskCapabilities(t);
  renderTaskPrerequisite(t);
  $("experiment-path").textContent=`logs/rsl_rl/${t.experiment}`;
  $("model-hint").textContent=t.shared?"Flat/Rough 共用目录，请手动选择与任务匹配的模型。":"续训只能使用当前算法和场景的 checkpoint。";
  updateRunNameHint();
}
function applyTrainingPreset(kind){
  const preset=task()?.capabilities?.train_presets?.[kind] || {
    envs:kind==="smoke"?64:1024,iterations:kind==="smoke"?2:16000,terrain_num_rows:10,terrain_num_cols:""};
  $("train-envs").value=preset.envs;
  $("iterations").value=preset.iterations;
  $("terrain-num-rows").value=preset.terrain_num_rows;
  $("terrain-num-cols").value=preset.terrain_num_cols;
}
function updateTrainingPresets(t){
  const presets=t.capabilities?.train_presets||{};
  for(const kind of ["smoke","full"]){
    const button=document.querySelector(`[data-preset="${kind}"]`);
    if(!button)continue;
    const preset=presets[kind];
    button.textContent=preset?`${kind==="smoke"?"快速检查":"正式训练"} · ${preset.envs} 环境 / ${preset.iterations} 轮（${preset.terrain_num_rows}×${preset.terrain_num_cols}）`:
      kind==="smoke"?"快速检查 · 64 环境 / 2 轮":"正式训练 · 1024 / 16000";
  }
  if(presetTaskAlias!==t.alias){
    const previous=state?.tasks?.find(item=>item.alias===presetTaskAlias);
    if(presets.full || previous?.capabilities?.train_presets?.full)applyTrainingPreset("full");
    presetTaskAlias=t.alias;
  }
}
function renderTaskCapabilities(t){
  const box=$("task-capabilities"); if(!box)return;
  box.replaceChildren();
  const cap=t.capabilities||{};
  const add=(text,title,cls)=>{const span=document.createElement("span");span.className="capability"+(cls?` ${cls}`:"");span.textContent=text;if(title)span.title=title;box.append(span);};
  if(cap.needs_pretrain) add("需预训练",cap.pretrain_command||"","warn");
  if(cap.supports_curriculum===false) add("无自动课程","平地任务不使用地形采样课程","muted");
  if(cap.docs) add("文档",cap.docs,"muted");
  if(!box.children.length) add("就绪","");
}
function renderTaskPrerequisite(t){
  const box=$("task-prereq"); if(!box)return;
  const actions=$("task-prereq-actions");
  const cap=t.capabilities||{};
  if(!cap.needs_pretrain){ box.className="task-prereq hidden"; actions?.classList.add("hidden"); return; }
  actions?.classList.remove("hidden");
  const active=(state?.jobs||[]).find(job=>job.mode==="pretrain" && ["running","stopping"].includes(job.status));
  const button=$("pretrain"), hint=$("pretrain-state");
  if(button){
    button.disabled=pretrainStarting||Boolean(active);
    button.textContent=pretrainStarting?"预训练启动中…":active?(active.status==="stopping"?"预训练停止中…":"预训练运行中…"):
      cap.pretrain_ready?"重新预训练感知模型":"预训练感知模型";
  }
  const latest=(state?.jobs||[]).find(job=>job.mode==="pretrain" && job.task===t.alias);
  if(hint){
    if(active) hint.textContent=`${active.task} · ${active.id}，日志在运行记录里`;
    else if(latest && ["failed","stopped"].includes(latest.status))
      hint.textContent=`上次预训练${latest.status==="failed"?"失败":"已停止"}，可重新启动；${cap.pretrain_ready?"现有感知模型仍可用":"查看运行日志"}`;
    else if(latest && ["completed","finished"].includes(latest.status) && !cap.pretrain_ready)
      hint.textContent="上次预训练进程已退出，但没有生成感知权重；请查看运行日志后重试。";
    else hint.textContent=cap.pretrain_ready?"可重新预训练，完成后替换当前感知模型。":"预训练完成后即可启动策略训练。";
  }
  if(cap.pretrain_ready){
    box.className="task-prereq ready";
    box.textContent=`感知模型已就绪：${cap.pretrain_checkpoint}`;
  }else{
    box.className="task-prereq missing";
    box.textContent=`该任务需要先预训练感知模型：${cap.pretrain_command||"python scripts/pretrain_mgdp.py"}（检查点 ${cap.pretrain_checkpoint||""} 不存在）`;
  }
}
async function startPretrain(){
  const button=$("pretrain");
  if(!button || button.disabled)return;
  const alias=$("task").value;
  pretrainStarting=true;
  button.disabled=true; button.textContent="预训练启动中…";
  try{
    const data=await api("/api/pretrain", {task:alias});
    state.jobs=[data.job,...(state.jobs||[]).filter(job=>job.id!==data.job.id)];
    $("pretrain-state").textContent=`已启动 ${data.job.id}，日志在运行记录里`;
    toast("预训练已启动，完成后会自动刷新状态");
    await refresh();
    pollPretrain(data.job.id,alias);
  }catch(error){
    toast(error.message);
  }finally{
    pretrainStarting=false;
    if(task())renderTaskPrerequisite(task());
  }
}
function pollPretrain(jobId,alias){
  const timer=setInterval(async()=>{
    try{
      const data=await api("/api/state");
      state=data;
      updateTask();
      const job=(data.jobs||[]).find(item=>item.id===jobId);
      if(job && ["failed","stopped","completed","finished"].includes(job.status)){
        clearInterval(timer);
        const ready=data.tasks.find(item=>item.alias===alias)?.capabilities?.pretrain_ready;
        if(job.status==="completed" && ready) toast("感知预训练完成，可以启动策略训练");
        else if(job.status==="finished") toast("预训练进程已退出，请查看运行日志确认结果");
        else toast(`预训练${job.status==="failed"?"失败":job.status==="stopped"?"已停止":"结束但检查点未生成"}，请查看运行日志`);
        if(task()?.alias===alias)await preview();
        await refresh();
      }
    }catch(error){ /* panel polling is best-effort */ }
  },4000);
}
function updateCurriculumOptions(){
  const select=$("terrain-curriculum");if(!select)return;
  const unsupported=!task()||task().scene==="flat";
  const directed=Boolean(task()?.directed_passage);
  select.disabled=unsupported||directed;if(unsupported||directed)select.value="manual";
  $("velocity-min-abs").disabled=directed;
  $("velocity-max-abs").disabled=directed;
  $("terrain-num-rows").disabled=unsupported;
  $("terrain-num-cols").disabled=unsupported;
  $("terrain-difficulty-max").disabled=unsupported;
  const automatic=select.value!=="manual";
  $("lp-options").classList.toggle("hidden",!automatic);
  // 隐藏的输入框仍会参与表单校验并被打包。手工模式下必须真正禁用它们：
  // 否则被隐藏的 LP 字段一旦不合法，浏览器无法聚焦提示，表现成「点启动没反应」。
  $("lp-options").querySelectorAll("input,select,button").forEach(control=>{control.disabled=!automatic;});
  $("terrain-curriculum-hint").textContent=directed?"定向越障：初始 0.25–0.35 m/s，所有障碍通过率和跟踪达标后扩速。":unsupported?"平地任务不使用地形采样课程。":automatic?"自动模式固定使用填写的速度包络，关闭原地形升降与扩速；先固定奖励做对照。":"手工模式按已有进度、存活与跟踪质量升降地形和扩速。";
}
async function updateRunNameHint(){
  const hint=$("run-name-hint"), input=$("run-name");
  if(!hint||!input||!state||!task()){if(hint)hint.textContent="当前组合没有自动命名信息。";return;}
  const version=++runNameVersion;const seed=$("seed")?.value||"42";
  try{
    const data=await api(`/api/run-name?task=${encodeURIComponent($("task").value)}&seed=${encodeURIComponent(seed)}`);
    if(version!==runNameVersion)return;
    input.placeholder=data.name;hint.textContent=`留空时实时使用：${data.name}`;
  }catch(error){if(version===runNameVersion)hint.textContent=`自动命名读取失败：${error.message}`;}
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
  else if(matching.length&&!task()?.shared) $("model").value=matching[0].path;
  $("summary-models").textContent=models.length;
}
async function loadModels() { const alias=$("task").value; if(!alias){models=[];fillModels();return;} const data=await api(`/api/models?task=${encodeURIComponent(alias)}`); if(alias!==$("task").value)return; models=data.models;fillModels(); }
function setHealthPill(element,status){
  if(!element)return;
  element.classList.toggle("warning",status==="warning");
  element.classList.toggle("observe",status==="observe");
  element.textContent=healthText(status);
}
function renderPassage(data){
  const card=$("passage-status-card");if(!card)return;
  card.classList.toggle("hidden",!task()?.directed_passage);
  const speed=metricValue(data,"passage_speed_max");
  $("passage-speed").textContent=speed==null?"等待越障训练数据…":`当前前进速度 0.25–${speed.toFixed(3)} m/s；各类达标后扩速`;
  const labels={start_flat:"平地",stepping_stones:"梅花桩",balance_beam:"单木桥（净距400mm）",gap_steps:["LgDog1-Race-MGDP-WallTraining","LgDog1-Race-MGDP-WallTraining-D435i20"].includes(task()?.id)?"楼梯过沟下坡→上坡过沟下楼梯":"坡道→过沟→下楼梯",high_wall:"高墙",clearance_frame:"限高笼",tire_array:"轮胎阵",random_hills:"随机高低起伏",mjlab_rough:"mjlab原生随机粗糙",slalom_brick:"绕桩→砖木坑"};
  const list=$("passage-results");list.replaceChildren();
  for(const [key,label] of Object.entries(labels)){
    const item=data?.passage?.[key];const div=document.createElement("div");
    const name=document.createElement("small");name.textContent=label;
    const value=document.createElement("strong");value.textContent=item?.samples>0?`${(100*item.rate).toFixed(1)}% · n=${Math.round(item.samples)}`:"未收集";
    div.append(name,value);list.append(div);
  }
}
function renderActionMetrics(data){
  renderPassage(data);
  const rawMax=metricValue(data,"raw_action_abs_max"), saturation=metricValue(data,"action_saturation_fraction"), policyStd=metricValue(data,"policy_mean_std"), reward=metricValue(data,"mean_reward");
  $("metric-raw-max").textContent=rawMax==null?"—":rawMax.toFixed(2);
  $("metric-saturation").textContent=saturation==null?"—":`${(saturation*100).toFixed(1)}%`;
  $("metric-policy-std").textContent=policyStd==null?"—":policyStd.toFixed(2);
  $("metric-reward").textContent=reward==null?"—":reward.toFixed(2);
  const status=$("action-health-status");
  const state=data?.available?(data.status|| (data.warning?"warning":"healthy")):"waiting";
  setHealthPill(status,state);
  if(data?.warning||data?.advisory){
    $("action-health-detail").textContent=`${data.warning||data.advisory} · ${data.note||""} · ${data.run||"未命名 run"} · step ${data.metrics?.raw_action_abs_max?.step??"—"}`;
  }else if(data?.available){
    $("action-health-detail").textContent=`${data.run||"未命名 run"} · 最新 step ${data.metrics?.raw_action_abs_max?.step??data.metrics?.mean_reward?.step??"—"} · 原始动作在裁剪前统计，不代表力矩饱和`;
  }else{
    $("action-health-detail").textContent=data?.warning||"训练启动后自动读取最新 TensorBoard 记录。";
  }
  renderTrainingMetrics(data);
}
function renderTrainingMetrics(data){
  renderTerrainCurriculum(data?.terrain_curriculum);
  const clearance=metricValue(data,"foot_clearance_height"),valid=metricValue(data,"foot_clearance_valid");
  $("metric-swing-height").textContent=clearance==null||!(valid>0)?"—":`${(clearance/valid).toFixed(3)} m`;
  const xy=metricValue(data,"velocity_error_xy"), yaw=metricValue(data,"velocity_error_yaw");
  const activeScore=metricValue(data,"velocity_tracking_score");
  const activeCommand=metricValue(data,"velocity_tracking_active");
  const score=metricValue(data,"command_linear_score_ema")??metricValue(data,"command_linear_score");
  const commandProgress=metricValue(data,"command_progress_ratio");
  const commandProgressThreshold=metricValue(data,"command_progress_threshold");
  const commandProgressGate=metricValue(data,"command_progress_gate");
  const commandScoreGate=metricValue(data,"command_score_gate");
  const survival=metricValue(data,"command_episode_fraction");
  const commandFailure=metricValue(data,"command_failure_fraction");
  const promotionAllowed=metricValue(data,"command_promotion_allowed");
  const fixedRanges=data?.terrain_curriculum?.fixed_velocity_ranges;
  const fixedX=(fixedRanges?.fields||fixedRanges)?.lin_vel_x;
  const vxMin=metricValue(data,"command_vx_min")??fixedX?.[0], vxMax=metricValue(data,"command_vx_max")??fixedX?.[1];
  const level=metricValue(data,"terrain_level"), ratio=metricValue(data,"terrain_tracking_ratio");
  const promote=metricValue(data,"terrain_promote_fraction"), demote=metricValue(data,"terrain_demote_fraction");
  const failure=metricValue(data,"terrain_failure_fraction");
  const velocityGate=metricValue(data,"terrain_velocity_gate_fraction");
  const progressGate=metricValue(data,"terrain_progress_gate_fraction");
  const ratioGate=metricValue(data,"terrain_ratio_gate_fraction");
  const distanceGate=metricValue(data,"terrain_distance_gate_fraction");
  const exposureGate=metricValue(data,"terrain_exposure_gate_fraction");
  $("metric-vel-xy").textContent=xy==null?"—":`${xy.toFixed(3)} m/s`;
  $("metric-vel-yaw").textContent=yaw==null?"—":`${yaw.toFixed(3)} rad/s`;
  $("metric-active-track-score").textContent=activeScore==null?"—":activeScore.toFixed(3);
  $("metric-active-command").textContent=activeCommand==null?"—":`${(activeCommand*100).toFixed(1)}%`;
  $("metric-command-score").textContent=score==null?"—":score.toFixed(3);
  $("metric-command-progress").textContent=commandProgress==null?"—":`${(commandProgress*100).toFixed(1)}%${commandProgressThreshold==null?"":` / ≥${(commandProgressThreshold*100).toFixed(0)}%`}`;
  $("metric-command-progress-gate").textContent=commandProgressGate==null?"—":commandProgressGate>0?"通过":"未通过";
  $("metric-command-score-gate").textContent=commandScoreGate==null?"—":commandScoreGate>0?"通过":"未通过";
  $("metric-command-survival").textContent=survival==null?"—":`${(survival*100).toFixed(1)}%`;
  $("metric-command-failure").textContent=commandFailure==null?"—":`${(commandFailure*100).toFixed(1)}%`;
  $("metric-command-gate").textContent=promotionAllowed==null?"—":promotionAllowed>0?"通过":"暂缓扩速";
  $("metric-vx-range").textContent=vxMin==null||vxMax==null?"—":`[${vxMin.toFixed(2)}, ${vxMax.toFixed(2)}]`;
  $("metric-terrain-level").textContent=level==null?"—":level.toFixed(2);
  $("metric-terrain-track").textContent=ratio==null?"—":ratio.toFixed(3);
  $("metric-terrain-balance").textContent=promote==null||demote==null?"—":`${(promote*100).toFixed(1)}% / ${(demote*100).toFixed(1)}%`;
  $("metric-terrain-failure").textContent=failure==null?"—":`${(failure*100).toFixed(1)}%`;
  $("metric-velocity-gate").textContent=velocityGate==null?"—":`${(velocityGate*100).toFixed(1)}%`;
  $("metric-progress-gate").textContent=progressGate==null?"—":`${(progressGate*100).toFixed(1)}%`;
  $("metric-ratio-gate").textContent=ratioGate==null?"—":`${(ratioGate*100).toFixed(1)}%`;
  $("metric-distance-gate").textContent=distanceGate==null?"—":`${(distanceGate*100).toFixed(1)}%`;
  $("metric-exposure-gate").textContent=exposureGate==null?"—":`${(exposureGate*100).toFixed(1)}%`;
  const status=$("training-health-status");
  const terrainObserve=promote!=null&&demote!=null&&demote>promote*1.5;
  const terrainWarning=failure!=null&&failure>0.35;
  const progressBlocked=commandProgressGate===0;
  const state=terrainWarning?"warning":terrainObserve||progressBlocked||commandScoreGate===0?"observe":data?.available?"healthy":"waiting";
  setHealthPill(status,state);
  if(!data?.available){
    $("training-health-detail").textContent="训练启动后自动读取速度追踪和地形课程指标。";
  }else if(terrainWarning){
    $("training-health-detail").textContent="回合失败率偏高，先检查接触终止、速度追踪和地形难度。";
  }else if(data?.terrain_curriculum){
    $("training-health-detail").textContent="自动地形采样已启用：等级均值允许回访下降，请结合任务概率、学习进度和固定评估判断表现。";
  }else if(progressBlocked){
    $("training-health-detail").textContent=`速度课程暂缓扩速：有效指令进度${commandProgress==null?"尚无数据":`仅${(commandProgress*100).toFixed(1)}%`}，或窗口内移动距离不足；绝对误差分数高不代表移动达标。`;
  }else if(terrainObserve){
    $("training-health-detail").textContent="地形降级多于升级；速度误差可以正常，但课程还没有稳定向上。";
  }else if(promotionAllowed===0){
    $("training-health-detail").textContent="速度课程暂缓扩速：存活、失败率或移动指令样本尚未满足条件。";
  }else if(commandScoreGate===0){
    $("training-health-detail").textContent="速度课程暂缓扩速：跟踪分数 EMA 尚未达到阈值。";
  }else{
    $("training-health-detail").textContent=`${data.run||"未命名 run"} · 扩速要求跟踪分数、有效进度、存活和失败率连续达标。`;
  }
}
function renderTerrainCurriculum(data){
  const card=$("lp-status-card");if(!card)return;
  card.classList.toggle("hidden",!data?.tasks?.length);
  if(!data?.tasks?.length)return;
  const largest=Math.max(...data.tasks.map(t=>Number(t.probability)||0));
  $("lp-status-summary").textContent=`${data.mode==="lp_acrl"?"LP 地形采样":"固定基础比例对照"} · ${data.sampling_version===2?"稳定版 v2":"旧版"} · ${data.tasks.length} 个任务 · 最大概率 ${(largest*100).toFixed(1)}% · 更新 ${data.updates??0} 次 · 控制步 ${data.control_step??0}`;
  const list=$("lp-task-grid");list.replaceChildren();
  const maximum=Math.max(...data.tasks.map(t=>Number(t.probability)),1e-9);
  for(const task of data.tasks){
    const tile=document.createElement("div");tile.className="lp-task-tile";
    const probability=Math.min(1,Math.max(0,Number(task.probability)||0));
    tile.style.background=`rgba(8,127,119,${.05+.35*probability/maximum})`;
    const label=document.createElement("strong");label.textContent=task.name;
    const value=document.createElement("b");value.textContent=`${(probability*100).toFixed(2)}%`;
    const detail=document.createElement("small");
    detail.textContent=`LP ${(Number(task.learning_progress)||0).toFixed(3)} · 回报 ${task.mean_return==null?"待采样":Number(task.mean_return).toFixed(2)} · n=${task.samples??0}`;
    tile.title=`${task.terrain} · 行 ${task.rows?.join(",")||"—"} · 待统计 ${task.pending_samples??0} 回合 · 失败率 ${task.failure_fraction==null?"—":`${(task.failure_fraction*100).toFixed(1)}%`}`;
    tile.append(label,value,detail);list.append(tile);
  }
}
function renderGpu(data){
  const status=$("gpu-status");
  if(!status)return;
  const gpu=data?.gpus?.[0];
  if(!data?.available||!gpu){
    status.classList.add("warning");status.textContent="不可用";
    $("gpu-memory").textContent="—";$("gpu-util").textContent="—";$("gpu-temp").textContent="—";$("gpu-power").textContent="—";
    $("gpu-detail").textContent=data?.reason||"未找到 nvidia-smi；服务器面板启动时需继承 NVIDIA 驱动环境。";return;
  }
  status.classList.remove("warning");status.textContent=`GPU ${gpu.index}`;
  const total=Number(gpu.memory_total_mib),used=Number(gpu.memory_used_mib),free=Number(gpu.memory_free_mib);
  const usedText=Number.isFinite(used)&&Number.isFinite(total)?`${(used/1024).toFixed(1)} / ${(total/1024).toFixed(1)} GB`:"—";
  $("gpu-memory").textContent=usedText;
  $("gpu-util").textContent=Number.isFinite(Number(gpu.utilization_gpu))?`${gpu.utilization_gpu}%`:"—";
  $("gpu-temp").textContent=Number.isFinite(Number(gpu.temperature_c))?`${gpu.temperature_c}°C`:"—";
  $("gpu-power").textContent=Number.isFinite(Number(gpu.power_w))?`${Number(gpu.power_w).toFixed(0)} W`:"—";
  const freeText=Number.isFinite(free)?` · 空闲 ${(free/1024).toFixed(1)} GB`:"";
  $("gpu-detail").textContent=`${gpu.name}${freeText}${data.gpus.length>1?` · 共 ${data.gpus.length} 张卡`:""}`;
}
async function loadActionMetrics(){
  if(!state||!$("task")?.value){renderActionMetrics({available:false,warning:"当前算法与场景没有注册任务",metrics:{}});return;}
  const alias=$("task").value;
  try{const data=await api(`/api/metrics?task=${encodeURIComponent(alias)}`);if(alias===$("task").value)renderActionMetrics(data);}
  catch(error){renderActionMetrics({available:false,warning:error.message,metrics:{}});}
}
let terrainData=null;
function renderTerrainEditor(data){
  terrainData=data;
  $("terrain-editor-path").textContent=`${data.path||"terrain_cfg.py"} · 修改后需重启训练`;
  const list=$("terrain-editor-list");list.replaceChildren();
  const global=document.createElement("div");global.className="terrain-global-fields";
  global.innerHTML='<label>地形行数<input data-terrain-global="rows" type="number" min="2" max="100"></label><label>地形列数<input data-terrain-global="cols" type="number" min="1" max="100"></label>';
  global.querySelector('[data-terrain-global="rows"]').value=data.rows;
  global.querySelector('[data-terrain-global="cols"]').value=data.cols;
  list.append(global);
  for(const item of data.terrains||[]){
    const row=document.createElement("div");row.className="terrain-row";row.dataset.terrain=item.name;
    const range=item.field?`<label>最小<input data-field="min" type="number" step="0.01" value="${item.min}"></label><label>最大<input data-field="max" type="number" step="0.01" value="${item.max}"></label>`:"<span></span><span></span>";
    row.innerHTML=`<div class="terrain-row-title"><strong>${item.label}</strong><small>${item.name} · ${item.description}</small></div><label>比例<input data-field="proportion" type="number" min="0" max="1" step="0.01" value="${item.proportion}"></label>${range}`;
    list.append(row);
  }
  $("terrain-editor-detail").textContent="比例总和必须为 1.0；保存后重启训练才会使用新地形。";
}
async function loadTerrainEditor(){
  try{renderTerrainEditor(await api("/api/terrain"));}
  catch(error){$("terrain-editor-detail").textContent=error.message;toast(error.message);}
}
function collectTerrain(){
  const rows=Number(document.querySelector('[data-terrain-global="rows"]').value);
  const cols=Number(document.querySelector('[data-terrain-global="cols"]').value);
  const terrains=[...document.querySelectorAll(".terrain-row")].map(row=>{
    const value=name=>{const input=row.querySelector(`[data-field="${name}"]`);return input?Number(input.value):null;};
    return {name:row.dataset.terrain,proportion:value("proportion"),min:value("min"),max:value("max")};
  });
  return {rows,cols,terrains};
}
async function saveTerrainEditor(){
  try{const data=await api("/api/terrain/save",collectTerrain());renderTerrainEditor(data);toast("地形配置已保存；重启训练后生效");}
  catch(error){toast(error.message);}
}
async function openTerrainViewer(){
  try{const data=await api("/api/terrain/view",{task:$("task").value});window.open(data.url,"_blank","noopener");toast("当前任务的 3D 地形查看器已启动");await refresh();}
  catch(error){toast(error.message);}
}
function showMode() {
  document.querySelector(".workspace-grid")?.classList.toggle("workspace-rewards",mode==="rewards");
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
function setWorkspaceView(view){
  const grid=document.querySelector(".workspace-grid");
  if(!grid)return;
  const mode=view==="monitor"?"monitor":view==="settings"?"settings":"both";
  grid.classList.toggle("workspace-monitor",mode==="monitor");
  grid.classList.toggle("workspace-settings",mode==="settings");
  grid.classList.toggle("workspace-both",mode==="both");
  for(const [id,value] of [["both-view","both"],["settings-view","settings"],["monitor-view","monitor"]]){
    $(id)?.classList.toggle("selected",mode===value);
    $(id)?.setAttribute("aria-pressed",String(mode===value));
  }
  try{localStorage.setItem("lgdog-workbench-view-v2",mode);}catch{}
}
function renderRewards(){
  const list=$("reward-list"); list.replaceChildren();
  if(!rewardData?.rewards?.length){const empty=document.createElement("div");empty.className="empty";empty.textContent="没有读取到 reward。";list.append(empty);return;}
  for(const reward of rewardData.rewards){
    // These task-specific values are edited in the wall panel above.
    if(wallRewardData?.available&&["passage_progress","passage_success"].includes(reward.name))continue;
    const row=document.createElement("div");row.className="reward-row"+(reward.changed?" changed":"");row.dataset.reward=reward.name;row.dataset.search=`${reward.name} ${rewardLabels[reward.name]||""}`.toLowerCase();row.title="双击查看实现";row.addEventListener("dblclick",()=>showRewardDetail(reward.name));
    const enabled=document.createElement("input");enabled.type="checkbox";enabled.checked=reward.enabled;enabled.title="启用 reward";
    const name=document.createElement("label");name.textContent=rewardLabels[reward.name]||reward.name;const code=document.createElement("small");code.textContent=`${reward.name}${reward.changed?" · 已覆盖":""}`;name.append(code);
    const weight=document.createElement("input");weight.type="number";weight.step="any";weight.value=reward.weight;weight.setAttribute("aria-label",`${reward.name} 权重`);weight.title=`基线：${reward.baseline}`;
    const params=document.createElement("textarea");params.value=JSON.stringify(reward.params||{});params.setAttribute("aria-label",`${reward.name} 参数 JSON`);params.placeholder="参数 JSON（例如 {\"target_height\":0.1}）";
    const semantics=document.createElement("small");name.append(semantics);
    const updateSignHint=()=>{
      const invalid=reward.kind==="cost"&&enabled.checked&&Number(weight.value)>0;
      row.classList.toggle("sign-error",invalid);
      semantics.textContent=invalid?"方向错误：正数会奖励误差/碰撞，请用 0 或负数":reward.weight_hint||"";
      semantics.className=invalid?"reward-sign-warning":"reward-weight-hint";
    };
    weight.addEventListener("input",updateSignHint);enabled.addEventListener("change",updateSignHint);updateSignHint();
    row.append(enabled,name,weight,params);list.append(row);
  }
  $("reward-source").textContent=`${rewardData.profile} · ${rewardData.source}`+(wallRewardData?.available?" · 下方为MGDP共享基础权重；距离和出口奖励请在高墙规则中修改。":"");
  filterRewards();
}
function filterRewards(){const query=($("reward-filter")?.value||"").trim().toLowerCase();document.querySelectorAll(".reward-row").forEach(row=>{row.classList.toggle("hidden",Boolean(query&&!row.dataset.search.includes(query)));});}
function resetRewards(){if(!rewardData?.rewards)return;document.querySelectorAll(".reward-row").forEach(row=>{const reward=rewardData.rewards.find(item=>item.name===row.dataset.reward);const inputs=row.querySelectorAll("input,textarea");inputs[0].checked=reward.baseline!==0;inputs[1].value=reward.baseline;inputs[2].value=JSON.stringify(reward.baseline_params||{});inputs[1].dispatchEvent(new Event("input"));row.classList.remove("changed");});toast("已恢复本方案基线，点击保存后才会写入文件");}
async function loadRewards(){
  try{
    const alias=$("task").value;
    const [base,wall]=await Promise.all([api(`/api/rewards?task=${encodeURIComponent(alias)}`),api(`/api/wall-rewards?task=${encodeURIComponent(alias)}`)]);
    if(alias!==$("task").value)return;
    rewardData=base;wallRewardData=wall;renderRewards();renderWallRules();
  }catch(error){$("reward-list").textContent=error.message;$("wall-rule-panel").classList.add("hidden");toast(error.message);}
}
function renderWallRules(values=wallRewardData?.settings){
  const panel=$("wall-rule-panel"),list=$("wall-rule-fields");
  panel.classList.toggle("hidden",!wallRewardData?.available);list.replaceChildren();
  if(!wallRewardData?.available)return;
  $("wall-rule-state").textContent=wallRewardData.changed?"已保存覆盖":"默认设置";
  $("wall-rule-note").textContent=wallRewardData.task==="race_mgdp_wall_d435i20"
    ?"本入口需要D435i下俯20°的配套感知与新策略。高墙规则与原相机强化课程共用；修改奖励目标后需用相同相机与感知迁移策略。"
    :wallRewardData.contract_changed?"已修改奖励目标：旧参数的检查点需策略迁移，不能直接完整续训。":"默认参数兼容原V5检查点；修改奖励目标后需策略迁移。";
  for(const [group,schema] of Object.entries(wallRewardData.groups)){
    const card=document.createElement("section");card.className="wall-rule-card";
    const title=document.createElement("h3");title.textContent=schema.label;
    const description=document.createElement("p");description.textContent=schema.description;
    const fields=document.createElement("div");fields.className="wall-rule-inputs";
    const effect=document.createElement("small");effect.className="wall-rule-effect";
    const update=()=>{
      $("wall-rule-state").textContent="有未保存修改";
      if(group==="wall_waiting"){
        const get=key=>fields.querySelector(`[data-key="${key}"]`);
        const rate=Math.abs(Number(get("weight").value))*Number(get("max_rate").value);
        effect.textContent=get("enabled").checked?`最大实际扣分：${Number.isFinite(rate)?rate.toFixed(2):"—"} 分/秒`:"等待惩罚已关闭（保存后生效）";
      }
    };
    for(const [key,field] of Object.entries(schema.fields)){
      const label=document.createElement("label");label.className="field wall-rule-field";
      const text=document.createElement("span");text.textContent=field.label+(field.unit?`（${field.unit}）`:"");
      const input=document.createElement("input");input.id=`wall-rule-${group}-${key}`;
      input.dataset.group=group;input.dataset.key=key;input.setAttribute("aria-label",`${schema.label} · ${field.label}`);
      if(field.type==="boolean"){input.type="checkbox";input.checked=values[group][key];label.classList.add("wall-rule-switch");}
      else{input.type="number";input.step="any";input.min=field.min;input.max=field.max;input.value=values[group][key];}
      input.addEventListener("input",update);label.append(text,input);fields.append(label);
    }
    card.append(title,description,fields);
    if(group==="wall_waiting"){
      card.append(effect);update();
    }
    list.append(card);
  }
  $("wall-rule-state").textContent=wallRewardData.changed?"已保存覆盖":"默认设置";
}
function collectWallRules(){
  const settings={};
  document.querySelectorAll("#wall-rule-fields input").forEach(input=>{
    let value=input.type==="checkbox"?input.checked:Number(input.value);
    if(input.type!=="checkbox"&&(!input.value.trim()||!Number.isFinite(value)))throw new Error(`${input.getAttribute("aria-label")}必须是数字`);
    (settings[input.dataset.group]??={})[input.dataset.key]=value;
  });return settings;
}
async function saveWallRules(){
  const button=$("save-wall-rules"),alias=$("task").value;
  try{
    button.disabled=true;
    const data=await api("/api/wall-rewards/save",{task:alias,settings:collectWallRules()});
    if(alias===$("task").value){wallRewardData=data;renderWallRules();}
    toast("高墙规则已保存，新启动的训练生效；原配置已备份");
  }catch(error){toast(error.message);}finally{button.disabled=false;}
}
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
    // 启动按钮被禁用时必须在显眼处提示原因，否则表现为「按键点了没反应」。
    if (error.message !== lastPreviewError) { toast(`无法启动：${error.message}`); lastPreviewError = error.message; }
  }
}
let previewTimer, lastPreviewError = "";
function queuePreview() {clearTimeout(previewTimer);previewTimer=setTimeout(preview,180);}
function renderJobs() {
  const list=$("jobs");
  const counts=state.job_counts;
  $("job-count").textContent=counts?.total??state.jobs.length;
  $("job-count").title=counts?.total>state.jobs.length?`列表显示最近 ${state.jobs.length} 条；历史记录可查看更早任务`:"运行记录数量";
  $("summary-running").textContent=counts?.active??state.jobs.filter(j=>["running","stopping"].includes(j.status)).length;
  const clearable=counts?.clearable??state.jobs.filter(j=>j.can_delete&&["completed","failed","stopped","finished"].includes(j.status)).length;
  $("clear-finished").disabled=deletingJobs||!clearable;
  if(!state.jobs.length){
    selectedJob=null;
    list.innerHTML='<div class="empty">还没有运行记录。<br>选择方案和参数后，点击启动。</div>';
    $("log-title").textContent="运行日志";$("logs").textContent="没有选中的运行记录。";
    $("stop").disabled=true;$("force-stop").disabled=true;$("delete-job").disabled=true;
    $("stop").textContent="停止任务";$("job-control-detail").classList.remove("control-error");
    $("job-control-detail").textContent="删除记录只清理面板记录与启动日志；训练数据和模型保留。";
    return;
  }
  if(!selectedJob||!state.jobs.some(j=>j.id===selectedJob))selectedJob=state.jobs[0].id;
  list.replaceChildren();
  for(const job of state.jobs) {
    const row=document.createElement("button"); row.type="button";row.className="job"+(job.id===selectedJob?" selected":"");
    const top=document.createElement("div");top.className="job-top";
    const title=document.createElement("span");title.textContent=modeNames[job.mode]||job.mode;
    const status=document.createElement("span");status.className=`status ${job.status}`;status.textContent=(statuses[job.status]||job.status)+
      (job.status==="failed" && Number.isInteger(job.returncode)?` · 退出码 ${job.returncode}`:"");
    top.append(title,status);
    const detail=document.createElement("small");detail.textContent=`${job.task} · PID ${job.pid}\n${new Date(job.created).toLocaleString()}`;
    row.append(top,detail);row.addEventListener("click",()=>{selectedJob=job.id;renderJobs();loadLog();});list.append(row);
  }
  const active=state.jobs.find(j=>j.id===selectedJob);
  const stopping=active?.status==="stopping"||pendingStops.has(active?.id);
  $("stop").disabled=!active?.can_stop||stopping;
  $("stop").textContent=stopping?"正在停止…":"停止任务";
  $("force-stop").disabled=!active?.can_stop||stopping;
  $("delete-job").disabled=!active?.can_delete||stopping||deletingJobs;
  $("job-control-detail").textContent=active?.stop_error?`停止失败：${active.stop_error}；可重试或强制结束。`:stopping?"正在结束进程及子进程，核验退出后才会显示已停止。":active?.stop_reason||(active?.recovered?"已核验并恢复旧任务控制；可停止进程及其子进程。":"删除记录只清理面板记录与启动日志；训练数据和模型保留。");
  $("job-control-detail").classList.toggle("control-error",Boolean(active?.stop_error));
  $("log-title").textContent=active?`${modeNames[active.mode]||active.mode} · ${active.id}`:"运行日志";
}
async function stopJob(force=false){
  const id=selectedJob;
  if(!id||pendingStops.has(id))return;
  pendingStops.add(id);renderJobs();
  try{
    const data=await api("/api/stop",{id,force});
    toast(data.job.status==="stopping"?"任务正在停止，请稍候":"任务进程已结束，已保存的 checkpoint 保留");
  }catch(error){toast(error.message);}
  finally{pendingStops.delete(id);await refresh();}
}
function requestDeleteJob(id){
  const job=state?.jobs.find(j=>j.id===id);
  if(job&&!job.can_delete){toast("请先停止任务再删除记录");return;}
  pendingDelete={id};
  $("delete-record-title").textContent="删除这条运行记录？";
  $("delete-record-detail").textContent=`将删除 ${job?`${modeNames[job.mode]||job.mode} · `:""}${id} 的面板记录、启动日志和历史运行条目。训练数据、checkpoint、ONNX 和奖励修改记录都会保留。`;
  $("confirm-delete-record").disabled=false;$("delete-record-dialog").showModal();
}
function requestClearFinished(){
  pendingDelete={all:true};
  $("delete-record-title").textContent="清理已结束的运行记录？";
  $("delete-record-detail").textContent="清理已完成、失败、已停止和已结束任务的面板记录与启动日志。运行中、停止中及身份未核验的旧任务会保留；训练数据、模型和奖励修改记录不会删除。";
  $("confirm-delete-record").disabled=false;$("delete-record-dialog").showModal();
}
async function confirmDeleteRecords(){
  if(!pendingDelete||deletingJobs)return;
  deletingJobs=true;$("confirm-delete-record").disabled=true;renderJobs();
  try{
    const data=pendingDelete.all?await api("/api/jobs/clear-finished",{}):await api("/api/jobs/delete",{id:pendingDelete.id});
    if(data.deleted.includes(selectedJob))selectedJob=null;
    $("delete-record-dialog").close();pendingDelete=null;
    await refresh();if($("history-dialog").open)await showHistory();
    toast(`已删除 ${data.deleted.length} 条运行记录，训练数据和模型保留`);
  }catch(error){toast(error.message);}
  finally{deletingJobs=false;$("confirm-delete-record").disabled=false;renderJobs();}
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
    renderGpu(data.gpu);renderJobs();if(task())renderTaskPrerequisite(task());if(all)await loadModels();await loadActionMetrics();await loadLog();
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
  const actions=document.createElement("div");actions.className="history-row-actions";
  const remove=document.createElement("button");remove.type="button";remove.className="small-button danger-button";remove.textContent="删除记录";remove.disabled=run.can_delete===false;
  remove.addEventListener("click",()=>requestDeleteJob(run.id));actions.append(status,remove);
  top.append(when,actions);row.append(top);
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
    let savedView="both";
    try{savedView=localStorage.getItem("lgdog-workbench-view-v2")||"both";}catch{}
    setWorkspaceView(savedView);
    const requested=new URLSearchParams(location.search).get("task");
    const initialTask=requested&&state.tasks.some(item=>item.alias===requested)?requested:"standard";
    renderModelSceneSelectors(initialTask);
    renderSimMaps();
    $("copy-root").textContent=state.root;$("python-path").textContent=state.python;
    updateTask();renderGpu(state.gpu);await updateRunNameHint();await loadModels();showMode();await preview();renderJobs();await loadActionMetrics();await loadTerrainEditor();await loadLog();
    document.querySelectorAll("[data-mode]").forEach(button=>button.addEventListener("click",()=>{mode=button.dataset.mode;showMode();preview();}));
    const changeTask=async()=>{$("custom-model").value="";$("run-name").value="";updateTask();await updateRunNameHint();await loadModels();showMode();await loadActionMetrics();preview();};
    $("task").addEventListener("change",async()=>{syncSelectorsFromTask();await changeTask();});
    $("algorithm").addEventListener("change",async()=>{renderModelSceneSelectors($("task").value);await changeTask();});
    $("scene").addEventListener("change",async()=>{renderModelSceneSelectors($("task").value);await changeTask();});
    $("map").addEventListener("change",()=>{renderMapInfo();queuePreview();});
    $("resume").addEventListener("change",()=>{showMode();preview();});
    $("terrain-curriculum").addEventListener("change",()=>{updateCurriculumOptions();queuePreview();});
    $("run-form").addEventListener("input",event=>{if(event.target.id==="seed")updateRunNameHint();queuePreview();});$("run-form").addEventListener("change",queuePreview);
    document.querySelectorAll("[data-preset]").forEach(button=>button.addEventListener("click",()=>{applyTrainingPreset(button.dataset.preset);preview();}));
    $("run-form").addEventListener("submit",async event=>{event.preventDefault();$("launch").disabled=true;try{const data=await api("/api/run",collect());selectedJob=data.job.id;toast("已启动，查看下方真实日志");await refresh();}catch(error){toast(error.message);}finally{preview();}});
    $("save-rewards").addEventListener("click",async()=>{try{$("save-rewards").disabled=true;const data=await api("/api/rewards/save",{task:$('task').value,changes:collectRewards()});rewardData=data;renderRewards();toast("奖励修改已保存，重启训练后生效；原文件已备份");}catch(error){toast(error.message);}finally{$("save-rewards").disabled=false;}});
    $("reload-rewards").addEventListener("click",loadRewards);
    $("reload-wall-rules").addEventListener("click",loadRewards);
    $("reset-wall-rules").addEventListener("click",()=>{if(wallRewardData?.available){renderWallRules(wallRewardData.defaults);$("wall-rule-state").textContent="默认值尚未保存";toast("已恢复默认显示，点击保存高墙规则后生效");}});
    $("save-wall-rules").addEventListener("click",saveWallRules);
    $("reset-rewards").addEventListener("click",resetRewards);
    $("reward-filter").addEventListener("input",filterRewards);
    $("stop").addEventListener("click",()=>stopJob());
    $("force-stop").addEventListener("click",()=>stopJob(true));
    $("delete-job").addEventListener("click",()=>requestDeleteJob(selectedJob));
    $("clear-finished").addEventListener("click",requestClearFinished);
    $("confirm-delete-record").addEventListener("click",confirmDeleteRecords);
    $("cancel-delete-record").addEventListener("click",()=>$("delete-record-dialog").close());
    $("curves").addEventListener("click",async()=>{try{const data=await api("/api/run",{mode:"curves",task:$("task").value,port:6006});selectedJob=data.job.id;await refresh();toast("TensorBoard 正在启动，稍后点击曲线页面");}catch(error){toast(error.message);}});
    $("refresh").addEventListener("click",async()=>{await refresh(true);preview();});
    $("terrain-reload").addEventListener("click",loadTerrainEditor);
    $("terrain-save").addEventListener("click",saveTerrainEditor);
    $("terrain-view").addEventListener("click",openTerrainViewer);
    $("copy-command").addEventListener("click",()=>lastCommand?copy(lastCommand):toast("当前没有可复制的命令"));
    $("copy-root").addEventListener("click",()=>copy(state.root));
    document.querySelectorAll("[data-doc]").forEach(button=>button.addEventListener("click",()=>showDocument(button.dataset.doc)));
    $("close-document").addEventListener("click",()=>$("document-dialog").close());
    $("copy-document-path").addEventListener("click",()=>copy($("document-path").textContent));
    $("history").addEventListener("click",showHistory);
    $("history-refresh").addEventListener("click",showHistory);
    $("close-history").addEventListener("click",()=>$("history-dialog").close());
    $("copy-history").addEventListener("click",()=>{const text=$("copy-history").dataset.text||"";text?copy(text):toast("还没有可复制的记录");});
    $("both-view").addEventListener("click",()=>setWorkspaceView("both"));
    $("settings-view").addEventListener("click",()=>setWorkspaceView("settings"));
    $("monitor-view").addEventListener("click",()=>setWorkspaceView("monitor"));
    $("pretrain")?.addEventListener("click",()=>startPretrain());
    setInterval(()=>refresh(),1800);
  }catch(error){toast(error.message);$("command").textContent="面板初始化失败，请检查服务终端。";}
}
start();


