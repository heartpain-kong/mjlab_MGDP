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
function collect() {
  const model = $("custom-model").value.trim() || $("model").value;
  return {mode, task:$("task").value, envs:$(mode==="play"?"play-envs":"train-envs").value,
    iterations:$("iterations").value, seed:$("seed").value, run_name:$("run-name").value,
    resume:$("resume").checked, checkpoint:model, onnx:model, viewer:$("viewer").value,
    map:$("map").value, duration:$("duration").value, vx:$("vx").value, vy:$("vy").value,
    wz:$("wz").value, spawn_y:$("spawn-y").value, show_viewer:$("show-viewer").checked,
    keyboard:$("keyboard").checked, term:$("term")?.value || ""};
}
function updateTask() {
  const t=task();
  $("summary-task").textContent=t.label;
  $("summary-obs").textContent=`${t.obs} 维`;
  $("task-description").textContent=t.description;
  $("critic-chip").textContent=`critic ${t.critic} 维`;
  $("profile-chip").textContent=`reward: ${t.profile}`;
  $("experiment-path").textContent=`logs/rsl_rl/${t.experiment}`;
  $("model-hint").textContent=t.shared?"Flat/Rough 共用目录，请手动选择与任务匹配的模型。":"模型来自当前任务的独立实验目录。";
}
function fillModels() {
  const previous=$("model").value;
  const kind=mode==="sim"?"onnx":"checkpoint";
  const matching=models.filter(m=>m.kind===kind);
  $("model").replaceChildren(new Option(matching.length?"选择模型文件":"该任务尚无对应模型", ""));
  for(const m of matching) $("model").add(new Option(`${m.run} / ${m.name}`,m.path));
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
async function start() {
  try {
    state=await api("/api/state");
    $("task").replaceChildren(...state.tasks.map(t=>new Option(t.label,t.alias)));
    $("task").value="standard";$("copy-root").textContent=state.root;$("python-path").textContent=state.python;
    updateTask();await loadModels();showMode();await preview();renderJobs();await loadActionMetrics();await loadLog();
    document.querySelectorAll("[data-mode]").forEach(button=>button.addEventListener("click",()=>{mode=button.dataset.mode;showMode();preview();}));
    $("task").addEventListener("change",async()=>{$("custom-model").value="";$("run-name").value=`${$("task").value}_seed${$("seed").value}`;updateTask();await loadModels();showMode();await loadActionMetrics();preview();});
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
    setInterval(()=>refresh(),1800);
  }catch(error){toast(error.message);$("command").textContent="面板初始化失败，请检查服务终端。";}
}
start();
