#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "types.h"
#include "config.h"
#include "flp.h"
#include "alloc-inl.h"
#include "debug.h"
#include "queue_entry_flp.h"

/* =========================================================
 * EP-FLP v2 — top-level definitions
 * Place these BEFORE all function definitions in flp.c
 * ========================================================= */

/* Global edge counter: incremented in save_if_interesting() */
u32 flp_total_edges_found = 0;

#ifndef FLP_TOPK_SIZE
#define FLP_TOPK_SIZE              20
#endif

#ifndef FLP_SOFTMAX_T_INIT
#define FLP_SOFTMAX_T_INIT         3.0
#endif

#ifndef FLP_SOFTMAX_T_FINAL
#define FLP_SOFTMAX_T_FINAL        1.0
#endif

#ifndef FLP_SOFTMAX_ANNEAL_CYCLES
#define FLP_SOFTMAX_ANNEAL_CYCLES  5000u
#endif

#ifndef FLP_EARLY_EDGE_THRESHOLD
#define FLP_EARLY_EDGE_THRESHOLD   1000u
#endif

#ifndef FLP_LINEAGE_CAP
#define FLP_LINEAGE_CAP            7.0
#endif

#ifndef FLP_LINEAGE_DECAY
#define FLP_LINEAGE_DECAY          0.05
#endif

#ifndef FLP_LINEAGE_EWMA_ALPHA
#define FLP_LINEAGE_EWMA_ALPHA     0.10
#endif

u64 flp_edge_last_seen[MAP_SIZE];

static u64 flp_last_window_execs;
static u32 flp_last_window_paths;
static u64 flp_last_window_crashes;

static u32 flp_global_epoch;
u64 flp_edge_first_discovered[MAP_SIZE];

/* edge 被命中的全局次数 */
u32 flp_edge_hit_cnt[MAP_SIZE];

/* 总执行次数 */
u64 flp_total_execs = 0;

u32 flp_active_edges[MAP_SIZE];
u32 flp_active_cnt = 0;

struct flp_lineage flp_lineages[FLP_MAX_LINEAGES];
u32 flp_lineage_cnt = 0;

/* 初始化 */
void flp_global_init(void) {

  memset(flp_edge_hit_cnt, 0, sizeof(flp_edge_hit_cnt));
  memset(flp_edge_last_seen, 0, sizeof(flp_edge_last_seen));
  memset(flp_edge_first_discovered, 0, sizeof(flp_edge_first_discovered));

  memset(flp_lineages, 0, sizeof(flp_lineages));
  flp_lineage_cnt = 0;

  flp_total_execs = 0;
  flp_active_cnt = 0;

  flp_last_window_execs = 0;
  flp_last_window_paths = 0;
  flp_last_window_crashes = 0;

  flp_global_epoch = 0;
}

u32 flp_create_lineage(u32 parent_id) {

  struct flp_lineage* L;
  struct flp_lineage* P = NULL;
  u32 id;

  if (parent_id > flp_lineage_cnt)
    parent_id = 0;

  if (flp_lineage_cnt + 1 >= FLP_MAX_LINEAGES)
    return 0;

  id = ++flp_lineage_cnt;
  L  = &flp_lineages[id];

  memset(L, 0, sizeof(*L));

  L->id        = id;
  L->parent_id = parent_id;
  L->seeds     = 1;
  L->pulls     = 0;

  L->last_decay_cycle = flp_global_epoch;

  if (parent_id) {

    P = &flp_lineages[parent_id];

    L->reward_ewma   = 0.50 * P->reward_ewma;
    L->frontier_rate = 0.50 * P->frontier_rate;
    L->success_rate  = 0.50 * P->success_rate;
    L->potential     = 0.50 * P->potential;

    /* 继承父代的时间上下文，让 recency_bonus 计算有合理起点 */
    L->last_success_exec = P->last_success_exec;
    L->last_seen_exec    = P->last_seen_exec;

  } else {

    /* 根 lineage：全部从零开始 */
    L->reward_ewma   = 0.0;
    L->frontier_rate = 0.0;
    L->success_rate  = 0.0;
    L->potential     = 0.0;

    L->last_success_exec = 0;
    L->last_seen_exec    = 0;
  }

  return id;
}

void flp_reward_lineage(u32 lineage_id,
                         struct queue_entry* child,
                         u8 hnb,
                         u64 total_execs) {

#if !FLP_USE_LINEAGE
  (void)lineage_id; (void)child; (void)hnb; (void)total_execs;
  return;
#endif

  struct flp_lineage* L;

  double reward        = 0.0;
  double frontier      = 0.0;
  double potential     = 0.0;
  double success       = 0.0;
  double recency_bonus = 0.0;
  double alpha         = FLP_LINEAGE_EWMA_ALPHA;

  if (!child) return;
  if (!lineage_id || lineage_id >= flp_lineage_cnt) return;
  if (lineage_id >= FLP_MAX_LINEAGES) return;

  L = &flp_lineages[lineage_id];

  /* -------------------------------------------------------
     PER-EPOCH DECAY
     ------------------------------------------------------- */
  if (L->last_decay_cycle != flp_global_epoch) {

    u32 cycles_elapsed = flp_global_epoch - L->last_decay_cycle;

    if (cycles_elapsed > 0 && cycles_elapsed < 100) {
      double decay_factor = pow(1.0 - FLP_LINEAGE_DECAY,
                                (double)cycles_elapsed);
      L->reward_ewma *= decay_factor;
      L->potential   *= decay_factor;
    }

    L->last_decay_cycle = flp_global_epoch;
  }

  /* -------------------------------------------------------
     CLAMP INPUT FEATURES
     ------------------------------------------------------- */
  frontier  = child->flp_frontier_score;
  potential = child->flp_frontier_potential;

  if (frontier  < 0.0 || isnan(frontier)  || isinf(frontier))  frontier  = 0.0;
  if (potential < 0.0 || isnan(potential) || isinf(potential)) potential = 0.0;

  if (frontier  > 10.0) frontier  = 10.0;
  if (potential > 10.0) potential = 10.0;

  /* -------------------------------------------------------
     REWARD SIGNAL (LOG-SATURATED)
     ------------------------------------------------------- */
  if      (hnb == 2) reward += 2.0;
  else if (hnb == 1) reward += 1.0;

  reward += 0.25 * frontier;
  reward += 0.15 * potential;

  if (child->hits_frontier)
    reward += 0.5;

  reward = log1p(reward);
  if (reward > 3.0) reward = 3.0;

  /* -------------------------------------------------------
     SUCCESS SIGNAL
     ------------------------------------------------------- */
  success = (hnb ? 1.0 : 0.0);

  /* -------------------------------------------------------
     UPDATE LINEAGE STATE
     ------------------------------------------------------- */
  L->pulls++;
  L->last_seen_exec = total_execs;

  if (success > 0.0)
    L->last_success_exec = total_execs;

  if (L->last_success_exec) {
    u64 age       = total_execs - L->last_success_exec;
    recency_bonus = exp(-FLP_RECENCY_LAMBDA * (double)age);
  } else {
    recency_bonus = 0.0;
  }

  if (recency_bonus < 0.0 || isnan(recency_bonus) || isinf(recency_bonus))
    recency_bonus = 0.0;
  if (recency_bonus > 1.0) recency_bonus = 1.0;

  double raw =
      0.60 * reward        +
      0.25 * recency_bonus +
      0.15 * success;

  L->reward_ewma =
      (1.0 - alpha) * L->reward_ewma +
      alpha          * raw;

  L->frontier_rate =
      0.90 * L->frontier_rate +
      0.10 * (child->hits_frontier ? 1.0 : 0.0);

  L->success_rate =
      0.95 * L->success_rate +
      0.05 * success;

  L->potential =
      0.70 * L->reward_ewma            +
      0.20 * (10.0 * L->frontier_rate) +
      0.10 * (10.0 * recency_bonus);

  if (L->potential > FLP_LINEAGE_CAP) L->potential = FLP_LINEAGE_CAP;
  if (L->potential < 0.0 || isnan(L->potential) || isinf(L->potential))
    L->potential = 0.0;

  {
    double old_lineage_score  = child->flp_lineage_score; /* 先读旧值 */
    double old_lineage_contrib = 0.10 * old_lineage_score;  /* 旧贡献 */

    child->flp_lineage_score = L->potential;               /* 再写新值 */

    double success_credit = child->flp_success_credit;
    if (success_credit < 0.0 || isnan(success_credit) || isinf(success_credit))
      success_credit = 0.0;
    if (success_credit > FLP_MAX_SUCCESS_CREDIT)
      success_credit = FLP_MAX_SUCCESS_CREDIT;

    double new_lineage_contrib = 0.10 * child->flp_lineage_score;

    child->flp_selection_score =
        child->flp_selection_score
        - old_lineage_contrib
        + new_lineage_contrib;

    if (child->flp_selection_score > 10.0)
      child->flp_selection_score = 10.0;
    if (child->flp_selection_score < 0.0         ||
        isnan(child->flp_selection_score)         ||
        isinf(child->flp_selection_score))
      child->flp_selection_score = 0.0;
  }

  {
    struct queue_entry* ancestor = child->flp_parent;
    double prop_reward           = reward * 0.5;
    double anc_alpha             = 0.20;
    u32    depth                 = 0;

    while (ancestor && ancestor != child && depth < FLP_LINEAGE_MAX_DEPTH) {

      double anc_lineage = ancestor->flp_lineage_score;

      if (anc_lineage < 0.0 || isnan(anc_lineage) || isinf(anc_lineage))
        anc_lineage = 0.0;

      double updated = (1.0 - anc_alpha) * anc_lineage
                       + anc_alpha       * prop_reward;

      if (updated > FLP_LINEAGE_CAP) updated = FLP_LINEAGE_CAP;
      if (updated < 0.0)              updated = 0.0;

      ancestor->flp_lineage_score = updated;

      /* 同步更新祖先的 selection_score，让调度器立即感知 */
      ancestor->flp_selection_score += anc_alpha * prop_reward * 0.10;
      if (ancestor->flp_selection_score > 10.0)
        ancestor->flp_selection_score = 10.0;

      prop_reward *= 0.5;
      ancestor     = ancestor->flp_parent;
      depth++;
    }
  }
}

void flp_preserve_topk(struct queue_entry* queue,
                        u32 queued_paths,
                        u32 queue_cycle) {

#ifndef FLP_DISABLE_TOPK

  extern u32 queued_favored;
  extern u32 pending_favored;

  struct queue_entry* q;
  u32 k, filled, i;
  double T, max_score, weight_sum, r, cumulative;

  typedef struct {
    double              score;
    struct queue_entry* entry;
  } ScoreEntry;

  ScoreEntry topk[FLP_TOPK_SIZE];
  double     weights[FLP_TOPK_SIZE];

  if (!queue || !queued_paths) return;

  queued_favored  = 0;
  pending_favored = 0;

  for (q = queue; q; q = q->next)
    q->flp_preserved = 0;

  k = (queued_paths < (u32)FLP_TOPK_SIZE)
      ? queued_paths : (u32)FLP_TOPK_SIZE;

  filled = 0;

  for (q = queue; q; q = q->next) {

    double s;

    if (!q->len) continue;

    s = q->flp_selection_score;
    if (s < 0.0 || isnan(s) || isinf(s)) s = 0.0;

    if (filled < k) {

      topk[filled].score = s;
      topk[filled].entry = q;
      filled++;

      u32 pos = filled - 1;
      while (pos > 0) {
        u32 parent = (pos - 1) / 2;
        if (topk[parent].score > topk[pos].score) {
          ScoreEntry tmp = topk[parent];
          topk[parent]   = topk[pos];
          topk[pos]      = tmp;
          pos = parent;
        } else break;
      }

    } else if (s > topk[0].score) {

      topk[0].score = s;
      topk[0].entry = q;

      u32 pos = 0;
      while (1) {
        u32 left     = 2 * pos + 1;
        u32 right    = 2 * pos + 2;
        u32 smallest = pos;

        if (left  < filled && topk[left].score  < topk[smallest].score) smallest = left;
        if (right < filled && topk[right].score < topk[smallest].score) smallest = right;
        if (smallest == pos) break;

        ScoreEntry tmp = topk[pos];
        topk[pos]      = topk[smallest];
        topk[smallest] = tmp;
        pos = smallest;
      }
    }
  }

  if (filled == 0) return;

  if (queue_cycle >= FLP_SOFTMAX_ANNEAL_CYCLES) {
    T = FLP_SOFTMAX_T_FINAL;
  } else {
    double progress = (double)queue_cycle / (double)FLP_SOFTMAX_ANNEAL_CYCLES;
    T = FLP_SOFTMAX_T_INIT
        + (FLP_SOFTMAX_T_FINAL - FLP_SOFTMAX_T_INIT) * progress;
  }

  if (T < 0.1) T = 0.1;

  max_score = topk[0].score;
  for (i = 1; i < filled; i++)
    if (topk[i].score > max_score) max_score = topk[i].score;

  weight_sum = 0.0;
  for (i = 0; i < filled; i++) {
    weights[i]  = exp((topk[i].score - max_score) / T);
    weight_sum += weights[i];
  }

  {
    u32 mark_target = (filled < (u32)FLP_TOPK_MARK_COUNT)
                      ? filled : (u32)FLP_TOPK_MARK_COUNT;
    u32 marked = 0;

    for (u32 draw = 0; draw < mark_target && weight_sum > 0.0; draw++) {

      r          = ((double)(rand() % 1000000) / 1000000.0) * weight_sum;
      cumulative = 0.0;

      struct queue_entry* selected = NULL;
      u32 sel_i = 0;

      for (i = 0; i < filled; i++) {
        cumulative += weights[i];
        if (r <= cumulative && topk[i].entry) {
          selected = topk[i].entry;
          sel_i    = i;
          break;
        }
      }

      /* Floating-point residual fallback: pick highest remaining weight */
      if (!selected) {
        double best_w = -1.0;
        for (i = 0; i < filled; i++) {
          if (topk[i].entry && weights[i] > best_w) {
            best_w = weights[i];
            sel_i  = i;
          }
        }
        selected = topk[sel_i].entry;
      }

      if (selected) {

        u8 was_favored = selected->favored;

        selected->flp_preserved = 1;

        if (!selected->favored) {
          selected->favored = 1;
          queued_favored++;
        }

        if (!was_favored && !selected->was_fuzzed)
          pending_favored++;

        marked++;
      }

      /* Remove this seed from the pool so it cannot be redrawn */
      weight_sum    -= weights[sel_i];
      weights[sel_i] = 0.0;
      topk[sel_i].entry = NULL;
    }

    /* Absolute fallback: pool exhausted with nothing marked — pick the
       highest-score seed directly from the queue.                       */
    if (marked == 0) {

      double best   = -1.0;
      struct queue_entry* best_q = NULL;

      for (q = queue; q; q = q->next) {
        if (!q->len) continue;
        if (q->flp_magic == FLP_QUEUE_MAGIC &&
            q->flp_selection_score > best) {
          best   = q->flp_selection_score;
          best_q = q;
        }
      }

      if (best_q) {
        u8 was_favored    = best_q->favored;
        best_q->flp_preserved = 1;
        if (!best_q->favored) {
          best_q->favored = 1;
          queued_favored++;
        }
        if (!was_favored && !best_q->was_fuzzed)
          pending_favored++;
      }
    }
  }

#else
  (void)queue;
  (void)queued_paths;
  (void)queue_cycle;
#endif /* FLP_DISABLE_TOPK */
}

static inline u8 flp_is_edge_active(u8 v) {
  return v >= 2;
}

void flp_collect_active_edges(void) {

  u32 i;
  volatile u8* ptr;

  flp_active_cnt = 0;

  if (!trace_bits)
    return;

  ptr = trace_bits;

  for (i = 0; i < MAP_SIZE; i++) {

    if (!flp_is_edge_active(ptr[i]))
      continue;

    flp_active_edges[flp_active_cnt++] = i;

    if (flp_active_cnt >= MAP_SIZE)
      break;
  }
}
void flp_update_edge_stats(u64 total_execs) {

  u64 cur_exec;
  u32 i;

  if (!flp_active_cnt)
    return;

  if (flp_active_cnt > MAP_SIZE)
    return;

  cur_exec = total_execs ? total_execs : 1;

  for (i = 0; i < flp_active_cnt; i++) {

    u32 idx = flp_active_edges[i];

    if (idx >= MAP_SIZE)
      continue;

    if (!flp_edge_hit_cnt[idx]) {
      flp_edge_first_discovered[idx] = cur_exec;
      if (flp_total_edges_found < UINT32_MAX)
        flp_total_edges_found++;
    }

    if (flp_edge_hit_cnt[idx] < UINT32_MAX)
      flp_edge_hit_cnt[idx]++;

    flp_edge_last_seen[idx] = cur_exec;
  }

  if (cur_exec / FLP_EPOCH_EVERY > flp_global_epoch)
    flp_global_epoch = (u32)(cur_exec / FLP_EPOCH_EVERY);
}

void flp_decay_edge_stats(void) {

  u32 i;

  for (i = 0; i < MAP_SIZE; i++) {

    u32 h   = flp_edge_hit_cnt[i];
    u32 dec = 0;

    if (!h)
      continue;

    /* frontier 边（hit_cnt <= 2）不 decay，保护探索信号 */
    if (h <= 3)
      continue;

    if      (h > 100000) dec = h / 64;
    else if (h > 10000)  dec = h / 128;
    else if (h > 1000)   dec = h / 256;
    else if (h > 100)    dec = 1;

    else if (h > 3)      dec = 1;
    else                 continue;

    if (!dec)
      continue;

    if (h <= 3 + dec)
      flp_edge_hit_cnt[i] = 3;
    else
      flp_edge_hit_cnt[i] = h - dec;
  }
}

static inline double flp_edge_recency(u32 idx, u64 total_execs) {

  u32 hits;
  u64 age;
  double recency;

  if (idx >= MAP_SIZE)
    return 0.0;

  hits = flp_edge_hit_cnt[idx];

  if (!hits)
    return 0.0;

  if (!flp_edge_last_seen[idx])
    return 0.8;

  age = total_execs > flp_edge_last_seen[idx] ?
        total_execs - flp_edge_last_seen[idx] : 0;


  recency = exp(-2e-7 * (double)age);

  if (hits == 1)
    recency = fmax(recency, 0.3);

  if (recency < 0.0 || isnan(recency) || isinf(recency))
    recency = 0.0;

  if (recency > 1.0)
    recency = 1.0;

  return recency;
}

static double flp_compute_seed_score(struct queue_entry* q,
                                      u8* trace_bits,
                                      u64 total_execs) {

  u32 i;
  u32 path_edges     = 0;
  u32 frontier_edges = 0;
  u32 fresh_edges    = 0;

  double rarity_sum   = 0.0;
  double recency_sum  = 0.0;
  double sparsity_sum = 0.0;

  double density, rarity, recency, sparsity;
  double fresh_ratio;
  double frontier_score, potential;
  double lineage, success;
  double edge_bonus, early_boost;

  if (!q || !trace_bits)
    return 0.0;

  for (i = 0; i < MAP_SIZE; i++) {

    u32 hits, local_hits = 0;
    s32 k;

    if (!flp_is_edge_active(trace_bits[i]))
      continue;

    path_edges++;

    hits = flp_edge_hit_cnt[i];

    if (hits == 1)
      fresh_edges++;

    /* 动态 frontier 阈值：早期收紧，避免所有边都满足条件导致分数趋同 */
    {
      u32 frontier_thresh;
      if (flp_total_edges_found < 500)
        frontier_thresh = 1;
      else if (flp_total_edges_found < 2000)
        frontier_thresh = 2;
      else
        frontier_thresh = 3;

      if (hits > 0 && hits <= frontier_thresh)
        frontier_edges++;
    }

    rarity_sum  += 1.0 / sqrt((double)hits + 1.0);
    recency_sum += flp_edge_recency(i, total_execs);

    for (k = -FLP_SPARSITY_RADIUS; k <= FLP_SPARSITY_RADIUS; k++) {

      s32 nidx;

      if (!k) continue;

      nidx = (s32)i + k;

      if (nidx < 0 || nidx >= MAP_SIZE)
        continue;

      if (flp_edge_hit_cnt[nidx] >= 1 && flp_edge_hit_cnt[nidx] <= 3)
        local_hits++;
    }

    sparsity_sum += 1.0 / (double)(local_hits + 1);
  }

  if (!path_edges) {

    q->flp_path_edges     = 0;
    q->flp_frontier_edges = 0;

    q->flp_density  = 0.0;
    q->flp_rarity   = 0.0;
    q->flp_recency  = 0.0;
    q->flp_sparsity = 0.0;

    q->flp_base_frontier_score = 0.0;
    q->flp_frontier_score      = 0.0;
    q->flp_frontier_potential  = 0.0;
    q->flp_selection_score     = 0.0;

    q->flp_score_epoch         = flp_global_epoch;
    q->flp_last_recompute_exec = total_execs;

    return 0.0;
  }

  density     = (double)frontier_edges / (double)path_edges;
  rarity      = rarity_sum             / (double)path_edges;
  recency     = recency_sum            / (double)path_edges;
  sparsity    = sparsity_sum           / (double)path_edges;
  fresh_ratio = (double)fresh_edges    / (double)path_edges;

  if (density     < 0.0 || isnan(density)     || isinf(density))     density     = 0.0;
  if (rarity      < 0.0 || isnan(rarity)      || isinf(rarity))      rarity      = 0.0;
  if (recency     < 0.0 || isnan(recency)     || isinf(recency))     recency     = 0.0;
  if (sparsity    < 0.0 || isnan(sparsity)    || isinf(sparsity))    sparsity    = 0.0;
  if (fresh_ratio < 0.0 || isnan(fresh_ratio) || isinf(fresh_ratio)) fresh_ratio = 0.0;

  if (density     > 1.0) density     = 1.0;
  if (rarity      > 1.0) rarity      = 1.0;
  if (recency     > 1.0) recency     = 1.0;
  if (sparsity    > 1.0) sparsity    = 1.0;
  if (fresh_ratio > 1.0) fresh_ratio = 1.0;

  frontier_score = 8.0 * density + 2.0 * fresh_ratio;

  if (frontier_score > 10.0) frontier_score = 10.0;
  if (frontier_score < 0.0 || isnan(frontier_score) || isinf(frontier_score))
    frontier_score = 0.0;

  /* 后期 frontier 崩塌保底：用 rarity/recency/sparsity 托住 */
  {
    double late_floor = (0.45 * rarity + 0.35 * recency + 0.20 * sparsity) * 10.0;

    if (late_floor < 0.0 || isnan(late_floor) || isinf(late_floor))
      late_floor = 0.0;
    if (late_floor > 7.0)
      late_floor = 7.0;

    if (frontier_score < late_floor)
      frontier_score = late_floor;
  }

  potential =3.5 * rarity +3.5 * recency +1.0 * sparsity +2.0 * fresh_ratio;   /* FIX 1: 同步替换 unknown_ratio */

  if (potential > 10.0) potential = 10.0;
  if (potential < 0.0 || isnan(potential) || isinf(potential))
    potential = 0.0;
    
  /* -------------------------------------------------------
     消融守卫：flp_no_lineage 下强制归零，
     防止字段残留历史值通过宏权重=0之外的路径渗入。
     ------------------------------------------------------- */
#ifdef FLP_DISABLE_LINEAGE
  lineage = 0.0;
  success = 0.0;
#else
  lineage = q->flp_lineage_score;
  success = q->flp_success_credit;

  if (lineage < 0.0 || isnan(lineage) || isinf(lineage)) lineage = 0.0;
  if (success < 0.0 || isnan(success) || isinf(success)) success = 0.0;

  if (lineage > 10.0)                    lineage = 10.0;
  if (success > FLP_MAX_SUCCESS_CREDIT) success = FLP_MAX_SUCCESS_CREDIT;
#endif

  edge_bonus = 0.0;
  if (q->flp_new_edges_found > 0)
    edge_bonus = 10.0 * tanh((double)q->flp_new_edges_found / 3.0);

  if (edge_bonus < 0.0 || isnan(edge_bonus) || isinf(edge_bonus))
    edge_bonus = 0.0;
  if (edge_bonus > 10.0)
    edge_bonus = 10.0;

  early_boost = 0.0;
  if (flp_total_edges_found < FLP_EARLY_EDGE_THRESHOLD) {
    double progress = (double)flp_total_edges_found / (double)FLP_EARLY_EDGE_THRESHOLD;
    early_boost = 1.0 * (1.0 - progress);
  }

  if (early_boost < 0.0 || isnan(early_boost) || isinf(early_boost))
    early_boost = 0.0;

  q->flp_path_edges     = path_edges;
  q->flp_frontier_edges = frontier_edges;

  q->flp_density  = density;
  q->flp_rarity   = rarity;
  q->flp_recency  = recency;
  q->flp_sparsity = sparsity;

  q->flp_base_frontier_score = frontier_score;
  q->flp_frontier_score      = frontier_score;
  q->flp_frontier_potential  = potential;

  q->flp_selection_score =
      0.40 * frontier_score +
      0.35 * potential      +
      0.10 * lineage        +
      0.05 * success        +
      0.10 * edge_bonus     +
      early_boost;

  if (q->flp_selection_score > 10.0)
    q->flp_selection_score = 10.0;

  if (q->flp_selection_score < 0.0  ||
      isnan(q->flp_selection_score)  ||
      isinf(q->flp_selection_score))
    q->flp_selection_score = 0.0;

  q->flp_score_epoch         = flp_global_epoch;
  q->flp_last_recompute_exec = total_execs;

  return q->flp_selection_score;
}

void flp_on_new_seed(struct queue_entry* q, u8* trace_bits,u64 total_execs) {

  if (!q || !trace_bits) return;

  q->flp_new_edges_found        = 0;
  q->flp_times_fuzzed           = 0;
  q->flp_child_finds            = 0;
  q->flp_frontier_child_finds   = 0;

  q->flp_multiplier             = 1.0;
  q->flp_adaptive_boost         = 1.0;

  q->flp_last_recompute_exec    = total_execs;
  q->flp_score_epoch            = flp_global_epoch;

  if (q->flp_lineage_score < 0.0 ||
      isnan(q->flp_lineage_score) ||
      isinf(q->flp_lineage_score))
    q->flp_lineage_score = 0.0;

  if (q->flp_lineage_score > 10.0)
    q->flp_lineage_score = 10.0;

  if (q->flp_success_credit < 0.0 ||
      isnan(q->flp_success_credit) ||
      isinf(q->flp_success_credit))
    q->flp_success_credit = 0.0;

  if (q->flp_success_credit > FLP_MAX_SUCCESS_CREDIT)
    q->flp_success_credit = FLP_MAX_SUCCESS_CREDIT;

  flp_compute_seed_score(q, trace_bits, total_execs);
}

static double flp_seed_penalty(struct queue_entry* q,
                                u64                 avg_exec_us,
                                u32                 avg_len) {

  double mult = 1.0;
  double norm_score;

  if (!q) return 1.0;

  norm_score = q->flp_selection_score / 10.0;

  if (norm_score < 0.0 || isnan(norm_score) || isinf(norm_score))
    norm_score = 0.0;
  if (norm_score > 1.0) norm_score = 1.0;

  /* -------------------------------------------------------
     速度惩罚：慢种子消耗太多 exec 预算
     ------------------------------------------------------- */
  if (avg_exec_us > 0 && q->exec_us > 0) {
    double speed_ratio = (double)q->exec_us / (double)avg_exec_us;
    if (speed_ratio > FLP_SLOW_PENALTY_THRES)
      mult *= 1.0 / sqrt(speed_ratio);
  }

  /* -------------------------------------------------------
     体积惩罚：过大种子变异效率低
     ------------------------------------------------------- */
  if (avg_len > 0 && q->len > 0) {
    double len_ratio = (double)q->len / (double)avg_len;
    if (len_ratio > FLP_LARGE_PENALTY_THRES)
      mult *= 1.0 / sqrt(len_ratio);
  }

  if (q->was_fuzzed &&
      q->flp_times_fuzzed > 500 &&
      q->flp_child_finds == 0 &&
      q->flp_frontier_child_finds == 0) {
    mult *= 0.55;
  } else if (norm_score < FLP_LOW_SCORE_THRES &&
             q->flp_times_fuzzed > 150 &&
             q->flp_child_finds == 0) {
    mult *= 0.80;
  }

  if (mult < 0.10) mult = 0.10;
  if (mult > 1.00) mult = 1.00;  /* FIX 3: 上限收紧到 1.0，语义清晰 */
  if (isnan(mult) || isinf(mult)) mult = 1.0;

  return mult;
}

u32 flp_apply_perf_score(struct queue_entry* q,
                          u32  base_perf_score,
                          u64  total_execs,
                          u64  avg_exec_us,
                          u32  avg_len) {

  double frontier, potential, lineage, success, edge_component;
  double penalty, flp_score, soft_boost, mult;
  double noise, early_mult_boost;
  double ucb_exploration;
  u32    new_score;

  if (!q) return base_perf_score;

  if (q->flp_score_epoch != flp_global_epoch) {
    q->flp_frontier_score     *= 0.985;
    q->flp_frontier_potential *= 0.985;
    /* FIX 2: flp_no_lineage 下 lineage 字段不应被触碰 */
#ifndef FLP_DISABLE_LINEAGE
    q->flp_lineage_score      *= 0.995;
#endif
    q->flp_score_epoch         = flp_global_epoch;
  }

  frontier  = q->flp_frontier_score;
  potential = q->flp_frontier_potential;
  lineage   = q->flp_lineage_score;
  success   = q->flp_success_credit;

  if (frontier  < 0.0 || isnan(frontier)  || isinf(frontier))  frontier  = 0.0;
  if (potential < 0.0 || isnan(potential) || isinf(potential)) potential = 0.0;
  if (lineage   < 0.0 || isnan(lineage)   || isinf(lineage))   lineage   = 0.0;
  if (success   < 0.0 || isnan(success)   || isinf(success))   success   = 0.0;

  if (frontier  > 10.0)                  frontier  = 10.0;
  if (potential > 10.0)                  potential = 10.0;
  if (lineage   > 10.0)                  lineage   = 10.0;
  if (success > FLP_MAX_SUCCESS_CREDIT) success   = FLP_MAX_SUCCESS_CREDIT;

  edge_component = 0.0;
  if (q->flp_new_edges_found > 0)
    edge_component = 10.0 * tanh((double)q->flp_new_edges_found / 3.0);

  if (edge_component < 0.0 || isnan(edge_component) || isinf(edge_component))
    edge_component = 0.0;
  if (edge_component > 10.0) edge_component = 10.0;

  flp_score =
      FLP_FRONTIER_WEIGHT  * frontier      +
      FLP_POTENTIAL_WEIGHT * potential     +
      FLP_LINEAGE_WEIGHT   * lineage       +
      FLP_SUCCESS_WEIGHT   * success       +
      0.10 * edge_component;

  if (flp_score < 0.0 || isnan(flp_score) || isinf(flp_score)) flp_score = 0.0;
  if (flp_score > 10.0) flp_score = 10.0;

  soft_boost = tanh(flp_score / 10.0);

  early_mult_boost = 0.0;
  if (flp_total_edges_found < FLP_EARLY_EDGE_THRESHOLD) {
    double progress = (double)flp_total_edges_found
                      / (double)FLP_EARLY_EDGE_THRESHOLD;
    early_mult_boost = 0.4 * (1.0 - progress);
  }

  ucb_exploration = 0.0;
  if (total_execs > 0) {
    double ln_n    = log((double)(total_execs + 1));
    double visits  = (double)(q->flp_times_fuzzed + 1);
    double ucb_raw = sqrt(2.0 * ln_n / visits);
    double ucb_max = sqrt(2.0 * ln_n);

    if (ucb_max > 0.0)
      ucb_exploration = ucb_raw / ucb_max;
    else
      ucb_exploration = 0.0;

    if (ucb_exploration > 1.0) ucb_exploration = 1.0;
    if (isnan(ucb_exploration) || isinf(ucb_exploration)) ucb_exploration = 0.0;
  }

  noise = 0.85 + ((double)(rand() % 300) / 1000.0);

  mult  = 1.0 + 1.50 * soft_boost + early_mult_boost + ucb_exploration;
  mult *= noise;

  penalty = flp_seed_penalty(q, avg_exec_us, avg_len);
  mult   *= penalty;

  if (mult > 3.5)              mult = 3.5;
  if (mult < 0.6)              mult = 0.6;
  if (isnan(mult) || isinf(mult)) mult = 1.0;

  q->flp_multiplier     = mult;
  q->flp_adaptive_boost = mult;
  q->flp_times_fuzzed++;

  new_score = (u32)((double)base_perf_score * mult);

  if (new_score < 1)                    new_score = 1;
  if (new_score > HAVOC_MAX_MULT * 100) new_score = HAVOC_MAX_MULT * 100;

  return new_score;
}

void flp_reward_parent_ancestor(struct queue_entry* parent,
                                 struct queue_entry* child,
                                 u8                  hnb,
                                 u64                 total_execs) {


#ifdef FLP_DISABLE_LINEAGE
  (void)parent; (void)child; (void)hnb; (void)total_execs;
  return;
#endif

  double reward, child_score, child_potential;
  struct queue_entry* p;
  u32 depth;

  if (!parent || !child)   return;
  if (parent == child)     return;
  if (parent->flp_magic != FLP_QUEUE_MAGIC) return;
  if (child->flp_magic  != FLP_QUEUE_MAGIC) return;

  child_score     = child->flp_frontier_score;
  child_potential = child->flp_frontier_potential;

  if (child_score     < 0.0 || isnan(child_score)     || isinf(child_score))     child_score     = 0.0;
  if (child_potential < 0.0 || isnan(child_potential) || isinf(child_potential)) child_potential = 0.0;

  if (child_score     > 10.0) child_score     = 10.0;
  if (child_potential > 10.0) child_potential = 10.0;

  reward = 0.0;

  if      (hnb == 2) reward += 4.0;
  else if (hnb == 1) reward += 2.0;

  reward += 0.45 * child_score;
  reward += 0.35 * child_potential;

  if (child->hits_frontier) reward += 1.5;

  if (reward < 0.0 || isnan(reward) || isinf(reward)) reward = 0.0;
  if (reward > 10.0) reward = 10.0;

  p     = parent;
  depth = 0;

  while (p && depth < 4 && reward > 0.05) {

    double frontier, potential, lineage, success;

    if (p == child) break;
    if (p->flp_magic != FLP_QUEUE_MAGIC) break;

    if (depth == 0)
      p->flp_lineage_score = 0.70 * p->flp_lineage_score + 0.30 * reward;
    else
      p->flp_lineage_score = 0.85 * p->flp_lineage_score + 0.15 * reward;

    if (p->flp_lineage_score > FLP_LINEAGE_CAP)
      p->flp_lineage_score = FLP_LINEAGE_CAP;
    if (p->flp_lineage_score < 0.0 ||
        isnan(p->flp_lineage_score) ||
        isinf(p->flp_lineage_score))
      p->flp_lineage_score = 0.0;

    if (p->flp_child_finds < UINT32_MAX)
      p->flp_child_finds++;

    if (child->hits_frontier &&
        p->flp_frontier_child_finds < UINT32_MAX)
      p->flp_frontier_child_finds++;

    p->flp_last_lineage_exec = total_execs;

    if (hnb || child->hits_frontier)
      p->flp_last_success_exec = total_execs;

    frontier  = p->flp_frontier_score;
    potential = p->flp_frontier_potential;
    lineage   = p->flp_lineage_score;
    success   = p->flp_success_credit;

    if (frontier  < 0.0 || isnan(frontier)  || isinf(frontier))  frontier  = 0.0;
    if (potential < 0.0 || isnan(potential) || isinf(potential)) potential = 0.0;
    if (lineage   < 0.0 || isnan(lineage)   || isinf(lineage))   lineage   = 0.0;
    if (success   < 0.0 || isnan(success)   || isinf(success))   success   = 0.0;

    if (frontier  > 10.0)                  frontier  = 10.0;
    if (potential > 10.0)                  potential = 10.0;
    if (lineage   > 10.0)                  lineage   = 10.0;
    if (success > FLP_MAX_SUCCESS_CREDIT) success   = FLP_MAX_SUCCESS_CREDIT;

    p->flp_selection_score =
        FLP_FRONTIER_WEIGHT  * frontier  +
        FLP_POTENTIAL_WEIGHT * potential +
        FLP_LINEAGE_WEIGHT   * lineage   +
        FLP_SUCCESS_WEIGHT   * success;

    if (p->flp_selection_score > 10.0)
      p->flp_selection_score = 10.0;
    if (p->flp_selection_score < 0.0 ||
        isnan(p->flp_selection_score) ||
        isinf(p->flp_selection_score))
      p->flp_selection_score = 0.0;

    reward *= FLP_LINEAGE_PARENT_DECAY;
    p       = p->flp_parent;
    depth++;
  }
}

u32 choose_block_len(u32 limit, struct queue_entry* q) {

  u32 r, result;

  /* 概率档位阈值：高 frontier 种子偏小块，低 frontier 偏大块 */
  u32 small_thresh, medium_thresh;

  if (limit == 0) return 1;
  if (limit == 1) return 1;

#ifdef FLP_USE_TOPK

  if (q && q->flp_magic == FLP_QUEUE_MAGIC) {

    double fs = q->flp_frontier_score;

    if (fs < 0.0 || isnan(fs) || isinf(fs)) fs = 0.0;
    if (fs > 10.0) fs = 10.0;

    double t = fs / 10.0;

    small_thresh  = (u32)(30.0 + 30.0 * t);   /* [30, 60] */
    medium_thresh = (u32)(65.0 + 25.0 * t);   /* [65, 90] */

  } else {

    /* FLP 信号不可用，退化为 AFL 标准分布：1/3 each */
    small_thresh  = 33;
    medium_thresh = 66;
  }

#else

  small_thresh  = 33;
  medium_thresh = 66;

#endif /* FLP_USE_TOPK */

  r = rand() % 100;

  if (r < small_thresh) {

    u32 cap = (limit < 32) ? limit : 32;
    result  = 1 + (rand() % cap);

  } else if (r < medium_thresh) {

    u32 cap = (limit < 128) ? limit : 128;
    result  = 1 + (rand() % cap);

  } else {

    u32 cap = (limit < 1500) ? limit : 1500;
    result  = 1 + (rand() % cap);
  }

  if (result < 1)      result = 1;
  if (result > limit)  result = limit;

  return result;
}
