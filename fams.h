#ifndef FAMS_H
#define FAMS_H
#include "types.h"

extern u8 *trace_bits;     // ✔ 指针
extern u8 virgin_bits[];   // ✔ 数组（关键）

/* ================== 对外接口 ================== */

/* 初始化（在 main 中调用） */
void fams_global_init(void);

/* 每次执行后调用 */
void fams_on_exec(void);

/* 在 havoc 中选择算子 */
u32 fams_select_mutator(u32 max_id);

/* 设置当前 mutator（用于统计） */
void fams_set_current_mutator(u32 id);

/* 在执行结束后更新反馈 */
void fams_feedback(void);

#endif
