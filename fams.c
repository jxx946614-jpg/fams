#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "types.h"
#include "config.h"

#include "fams.h"

#define FAMS_MAX_MUTATORS 17
#define COVERAGE_HIST_SIZE 256  // 覆盖率历史记录长度
#define COVERAGE_UPDATE_INTERVAL 10000
#define COVERAGE_SATURATION_THRESHOLD 0.001  // 饱和阈值（0.01% 覆盖率增长）
#define FRONTIER_EDGE_FREQ 8
#define FAMS_DEFAULT_WEIGHT 1.0 // 未使用算子的默认权重（避免被完全禁用）
#define FAMS_MIN_WEIGHT 0.05   // 最小权重阈值（防止权重为0）

// 算子统计结构：仅用于变异抑制
typedef struct {
  u64 total_use;          // 算子总使用次数
  u64 frontier_hits;      // 算子触发的edge frontier命中次数
  double weight; // 抑制权重
} fams_mutator_t;

// 全局变异算子统计数组
static u32 cur_mutator_id = 0;

static fams_mutator_t fams_mutators[FAMS_MAX_MUTATORS];

static u8 frontier_map[MAP_SIZE];

static u32 fams_edge_hit_cnt[MAP_SIZE];

static u64 fams_edge_first_discovered[MAP_SIZE];

// 总执行次数（用于控制前沿更新频率）
static u64 fams_total_execs = 0;

// 覆盖率历史数组（循环存储最近的覆盖率值）
static double coverage_history[COVERAGE_HIST_SIZE];
// 覆盖率历史索引（循环更新）
static u32 coverage_hist_idx = 0;


static void fams_init(void) {

  for (u32 i = 0; i < FAMS_MAX_MUTATORS; i++) {

    fams_mutators[i].weight = 1.0 / FAMS_MAX_MUTATORS;
    fams_mutators[i].total_use = 0;
    fams_mutators[i].frontier_hits = 0;

  }

  memset(fams_edge_hit_cnt, 0, sizeof(fams_edge_hit_cnt));
  memset(frontier_map, 0, sizeof(frontier_map));
  memset(coverage_history, 0, sizeof(coverage_history));
   coverage_hist_idx = 0;
  fams_total_execs = 0;
  for (u32 i = 0; i < MAP_SIZE; i++)
    fams_edge_first_discovered[i] = UINT64_MAX;
}
/*覆盖率统计*/
static double fams_compute_coverage(void) {

  u32 edges = 0;

  for (u32 i = 0; i < MAP_SIZE; i++) {

    if (virgin_bits[i] != 0xff)
      edges++;

  }

  return (double)edges / MAP_SIZE;
}
/* 更新全局覆盖统计（用于饱和检测）（基于 virgin_bits，无 CFG 依赖）*/
static void fams_update_coverage(void) {
  double cov = fams_compute_coverage();

  // 记录覆盖率历史
  coverage_history[coverage_hist_idx] = cov;
  coverage_hist_idx = (coverage_hist_idx + 1) % COVERAGE_HIST_SIZE;
}
/*判断覆盖率是否进入饱和状态*/
static u8 fams_is_saturated(void) {
  if (fams_total_execs < COVERAGE_HIST_SIZE)
    return 0;

  // 获取半个缓冲区周期的覆盖率值
  u32 prev_idx = (coverage_hist_idx-1 + COVERAGE_HIST_SIZE / 2) % COVERAGE_HIST_SIZE;
  double prev = coverage_history[prev_idx];
  // 计算覆盖率增长值
  double cur = coverage_history[(coverage_hist_idx-1 + COVERAGE_HIST_SIZE) % COVERAGE_HIST_SIZE];
  if (prev < 1e-6)
    return 0;
  double growth = (cur - prev) / prev;

  return (growth < COVERAGE_SATURATION_THRESHOLD);
}
/*frontier map goujian*/
static void fams_compute_frontier(void) {

  for (u32 i = 0; i < MAP_SIZE; i++) {

    if (trace_bits[i]) {

      fams_edge_hit_cnt[i]++;

      if (fams_edge_hit_cnt[i] < FRONTIER_EDGE_FREQ)
        frontier_map[i] = 1;
      else
        frontier_map[i] = 0;

    } else {
      frontier_map[i] = 0;
    }
  }
}
/* 判断当前执行是否 hit edge frontier */
static int fams_hit_frontier(void) {

  for (u32 i = 0; i < MAP_SIZE; i++) {
    /* 只检测本次新增 edge */
    if (trace_bits[i] && frontier_map[i]) {
      return 1;
    }
  }
  return 0;
}
/*更新变异算子的抑制权重：高效算子权重低，低效算子权重高（基于 edge frontier 命中频率）*/
static void fams_update_weights(void) {
  double total_weight = 0.0;
  // 遍历所有算子，逐一枚算权重
  for (u32 i = 0; i < FAMS_MAX_MUTATORS; i++) {
    double freq = 0.0;
    // 1. 计算前沿命中频率（处理除零：算子未被使用时，freq为0）
    if (fams_mutators[i].total_use > 0) {
      freq = (double)fams_mutators[i].frontier_hits / fams_mutators[i].total_use;
      //反向映射：高效算子→ 低权重
      freq = 1.0 - freq;
    }else {
      // 2. 未使用算子：赋予探索型默认权重
      freq = FAMS_DEFAULT_WEIGHT;
    }

    fams_mutators[i].weight = freq;

    total_weight += freq;

  }

  for (u32 i = 0; i < FAMS_MAX_MUTATORS; i++) {

    fams_mutators[i].weight /= total_weight;

  }
}
// FAMS: 根据抑制权重加权随机选择变异算子
static u32 fams_choose_mutator(u32 max_id) {
  double r = (double)rand() / RAND_MAX;
  double cumulative = 0;

  for (u32 i = 0; i < max_id; i++) {

    cumulative += fams_mutators[i].weight;

    if (r <= cumulative)
      return i;

  }

  return max_id - 1;
}
/* ================== 对外接口实现 ================== */

void fams_global_init(void) {
  fams_init();
}

/* 在每次执行后调用 */
void fams_on_exec(void) {

  fams_total_execs++;

  fams_update_coverage();

  if (fams_total_execs % COVERAGE_UPDATE_INTERVAL == 0) {

    if (fams_is_saturated()) {
      fams_compute_frontier();
      fams_update_weights();
    }

  }

}

/* 选择 mutator */
u32 fams_select_mutator(u32 max_id) {
  return fams_choose_mutator(max_id);
}

/* 设置当前 mutator */
void fams_set_current_mutator(u32 id) {
  cur_mutator_id = id;
  fams_mutators[id].total_use++;
}

/* 执行后反馈 */
void fams_feedback(void) {

  if (fams_hit_frontier()) {
    fams_mutators[cur_mutator_id].frontier_hits++;
  }

}

