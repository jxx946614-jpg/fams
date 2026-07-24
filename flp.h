#ifndef _HAVE_FLP_H
#define _HAVE_FLP_H

/* flp.h 顶部，已有的 FLP_USE_TOPK 附近加： */

/* 消融：禁用 lineage 传播。定义此宏后：
 * - flp_reward_parent_ancestor 变为空操作
 * - flp_selection_score / flp_score 中 lineage + success 权重归零
 * 用于验证血统信号对覆盖率的贡献。                          */
#ifdef FLP_DISABLE_LINEAGE
#  define FLP_LINEAGE_WEIGHT  0.00
#  define FLP_SUCCESS_WEIGHT  0.00
/* 剩余权重补到 frontier 和 potential，保持归一 */
#  define FLP_FRONTIER_WEIGHT 0.50
#  define FLP_POTENTIAL_WEIGHT 0.50
#else
#  define FLP_LINEAGE_WEIGHT   0.10
#  define FLP_SUCCESS_WEIGHT   0.05
#  define FLP_FRONTIER_WEIGHT  0.40
#  define FLP_POTENTIAL_WEIGHT 0.35
#endif

#include "types.h"
#include "config.h"

#define FLP_FRONTIER_MIN 0.01
#define FLP_PLATEAU_WINDOWS 3
#define FLP_EPOCH_EVERY 10000

#define FLP_MAX_MUTATORS        17
#define FRONTIER_EDGE_FREQ       2
#define FLP_ALPHA               1.0
#define FLP_RECENCY_LAMBDA      0.00000005
#define FLP_SCORE_EPOCH_STEP    50000
#define FLP_WARMUP_EXECS        10000
#define FLP_WINDOW_EXECS        10000
#define FLP_MIN_PLATEAU_EXECS   100000
#define FLP_MIN_PLATEAU_PATHS   200
#define FLP_PATH_RATE_THRES     0.60
#define FLP_PLATEAU_THRES       0.4
#define FLP_CRASH_NOISE_THRES   0.10
#define FLP_SLOW_PENALTY_THRES  3
#define FLP_LARGE_PENALTY_THRES 3
#define FLP_EDGE_FRONTIER_THRES FRONTIER_EDGE_FREQ
#define FLP_LONG_SEED_THRES     2.0
#define FLP_SCORE_WEIGHT        0.45
#define FLP_ADAPTIVE_BOOST_BETA 1.0
#define FLP_POTENTIAL_RADIUS    4
#define FLP_SPARSITY_RADIUS     4

#define FLP_BASE_STRENGTH          0.05
#define FLP_PLATEAU_STRENGTH_MIN   0.70
#define FLP_LOW_SCORE_THRES        0.3
#define FLP_LOW_SCORE_PENALTY      0.15
#define FLP_MID_SCORE_THRES        0.18
#define FLP_MID_SCORE_PENALTY      0.75

#define FLP_SUCCESS_BETA        1.5
#define FLP_SUCCESS_DECAY_EXEC  200000.0
#define FLP_MAX_SUCCESS_CREDIT  1.0

#define FLP_LINEAGE_BETA          1.2
#define FLP_LINEAGE_DECAY_EXEC    500000

#define FLP_SUCCESS_MIN           0.05

#define FLP_POTENTIAL_MIN 0.01
#define FLP_LINEAGE_MIN   0.01

#define FLP_LINEAGE_PARENT_DECAY  0.70
#define FLP_LINEAGE_MAX_DEPTH     5
#define FLP_QUEUE_MAGIC           0x46414d53

#define FLP_LOW_VALUE_THRESHOLD    0.25
#define FLP_LOW_VALUE_SKIP_PROB    80

#define FLP_RECOMPUTE_CUR_SEED_EVERY   128
#define FLP_RECOMPUTE_QUEUE_EVERY      4096
#define FLP_DECAY_EDGE_EVERY           524288
#define FLP_EARLY_EXEC_LIMIT           5000000ULL

#define FLP_POTENTIAL_SCALE            6.0
#define FLP_SCORE_SCALE                5.0

#define FLP_MAX_LINEAGES               65536

/* -----------------------------------------------
 * EP-FLP v2: new macros
 * ----------------------------------------------- */

/* Top-k softmax sampling */
#ifndef FLP_TOPK_SIZE
#define FLP_TOPK_SIZE              64
#endif

#define FLP_TOPK_MARK_COUNT  (FLP_TOPK_SIZE / 4)  /* seeds marked favored per cull cycle */

#ifndef FLP_SOFTMAX_T_INIT
#define FLP_SOFTMAX_T_INIT         3.0
#endif

#ifndef FLP_SOFTMAX_T_FINAL
#define FLP_SOFTMAX_T_FINAL        1.0
#endif

#ifndef FLP_SOFTMAX_ANNEAL_CYCLES
#define FLP_SOFTMAX_ANNEAL_CYCLES  5000u
#endif

/* Early exploration boost threshold (unique edges found) */
#ifndef FLP_EARLY_EDGE_THRESHOLD
#define FLP_EARLY_EDGE_THRESHOLD   1000u
#endif

/* Lineage decay and cap */
#ifndef FLP_LINEAGE_CAP
#define FLP_LINEAGE_CAP            7.0
#endif

#ifndef FLP_LINEAGE_DECAY
#define FLP_LINEAGE_DECAY          0.05
#endif

#ifndef FLP_LINEAGE_EWMA_ALPHA
#define FLP_LINEAGE_EWMA_ALPHA     0.10
#endif

/* -----------------------------------------------
 * flp_lineage struct
 * ----------------------------------------------- */
struct queue_entry;

struct flp_lineage {
  u32 id;
  u32 parent_id;

  u32 seeds;
  u32 pulls;

  double reward_ewma;
  double frontier_rate;
  double success_rate;
  double potential;

  u64 last_success_exec;
  u64 last_seen_exec;

  /* EP-FLP v2: per-cycle decay tracking
   * Records the flp_global_epoch value at which decay
   * was last applied; prevents multiple decays per epoch. */
  u32 last_decay_cycle;
};

/* -----------------------------------------------
 * Global state
 * ----------------------------------------------- */
extern u32 flp_lineage_cnt;

extern u8*  trace_bits;
extern u64  flp_edge_first_discovered[MAP_SIZE];
extern u32  queued_favored;
extern u32  pending_favored;

extern u64  flp_total_execs;

extern u32  flp_active_edges[MAP_SIZE];
extern u32  flp_active_cnt;

extern u32  flp_edge_hit_cnt[MAP_SIZE];
extern u64  flp_edge_last_seen[MAP_SIZE];

extern struct flp_lineage flp_lineages[FLP_MAX_LINEAGES];

/* EP-FLP v2: global unique-edge counter
 * Incremented in save_if_interesting() each time a new
 * coverage edge is first observed across the campaign. */
extern u32  flp_total_edges_found;

/* NOTE: flp_global_epoch is static inside flp.c — not exported */

/* -----------------------------------------------
 * API
 * ----------------------------------------------- */
void flp_global_init(void);

void flp_collect_active_edges(void);
void flp_update_edge_stats(u64 total_execs);
void flp_decay_edge_stats(void);

void flp_on_new_seed(struct queue_entry* q, u8* trace_bits, u64 total_execs);

u32 flp_apply_perf_score(struct queue_entry* q, u32  base_perf_score, u64  total_execs, u64  avg_exec_us, u32  avg_len);

void flp_reward_parent_ancestor(struct queue_entry* parent,
                                  struct queue_entry* child,
                                  u8 hnb, u64 total_execs);

u32  flp_create_lineage(u32 parent_id);

void flp_reward_lineage(u32 lineage_id, struct queue_entry* child,
                          u8 hnb, u64 total_execs);

/* EP-FLP v2: queue_cycle passed from afl-fuzz.c main loop counter */
void flp_preserve_topk(struct queue_entry* queue, u32 queued_paths, u32 queue_cycle);

u32  choose_block_len(u32 limit, struct queue_entry* q);

#endif /* _HAVE_FLP_H */
