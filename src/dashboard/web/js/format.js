/* Pure formatting/label helpers shared by the dashboard.
 *
 * Loaded as a classic script before app.js (no module rewiring, no defer-order
 * change).  Kept free of DOM access so it can be unit-tested in Node.
 */
(function (root) {
  "use strict";
  const modeNames = {train:"训练", play:"Play 回放", sim:"Sim2Sim", rewards:"查看奖励", curves:"训练曲线", terrain:"地形查看"};
  const statuses = {running:"运行中", stopping:"停止中", completed:"已完成", failed:"失败", stopped:"已停止", finished:"已结束", unknown:"历史·未接管"};
  const rewardLabels = {track_linear_velocity:"线速度跟踪",track_angular_velocity:"角速度跟踪",body_orientation_l2:"机身姿态",pose:"关节姿态",foot_gait:"足端步态",body_ang_vel:"机身角速度",angular_momentum:"角动量",joint_acc_l2:"关节加速度",joint_pos_limits:"关节限位",action_rate_l2:"动作变化率",foot_clearance:"抬脚高度",foot_slip:"足端滑移",soft_landing:"落足冲击",feet_stumble:"足端绊倒",feet_air_time:"足端滞空",leg_contact_force:"腿部接触力",stand_still:"静止站立",is_terminated:"终止惩罚"};

  function metricValue(data, name) { return data?.metrics?.[name]?.value; }
  function healthText(status) { return {healthy:"正常",observe:"观察",warning:"需要检查"}[status] || "等待数据"; }

  root.LgDogFormat = Object.freeze({modeNames, statuses, rewardLabels, metricValue, healthText});
})(typeof globalThis !== "undefined" ? globalThis : this);
