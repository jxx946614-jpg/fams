#ifndef _HAVE_QUEUE_ENTRY_FLP_H
#define _HAVE_QUEUE_ENTRY_FLP_H

#include "types.h"

struct queue_entry {

  /* ================= AFL ================= */

  u8* fname;
  u32 len;

  u8  cal_failed,
      trim_done,
      was_fuzzed,
      passed_det,
      has_new_cov,
      var_behavior,
      favored,
      fs_redundant;

  u32 bitmap_size,
      exec_cksum;

  u64 exec_us,
      handicap,
      depth;

  u8* trace_mini;
  u32 tc_ref;

  struct queue_entry *next,
                     *next_100;

  /* ================= FLP ================= */

  u32 flp_magic;

  struct queue_entry* flp_parent;

  /* coverage frontier */

  u32 flp_path_edges;
  u32 flp_frontier_edges;

  double flp_density;
  double flp_rarity;
  double flp_recency;
  double flp_sparsity;

  double flp_frontier_potential;

  double flp_base_frontier_score;
  double flp_frontier_score;

  /* lineage */

  double flp_lineage_score;
  double flp_success_credit;

  /* scheduler */

  double flp_selection_score;
  double flp_multiplier;
  double flp_adaptive_boost;

  u32 flp_lineage_id;
  u32 flp_depth;

  /* timestamps */

  u64 flp_last_lineage_exec;
  u64 flp_last_success_exec;
  u64 flp_last_recompute_exec;

  /* UCB */

  u64 flp_times_fuzzed;

  /*
   * lineage-level visit count
   * FLP / Cerebro / SLIME 风格调度
   */
  u64 flp_lineage_visits;

  /* reward stats */

  u32 flp_child_finds;
  u32 flp_frontier_child_finds;

  /* bookkeeping */

  u32 flp_score_epoch;

  u8 hits_frontier;
  u8 flp_preserved;

  u32 flp_new_edges_found;

};

#endif /* _HAVE_QUEUE_ENTRY_FLP_H */
