# MDLM 多 head 蒸馏 - 实验方案

## 1. 目标

在 MDLM（masked diffusion language model，169M，OpenWebText，DiT backbone）之上做多 head 蒸馏，让一次 backbone forward 服务 K 个连续反向 substep。

## 1.1 参考代码

**MDLM 官方代码仓库**：https://github.com/kuleshov-group/mdlm

实现过程中可以也应该借鉴 MDLM 官方代码，特别是以下部分必须严格对齐 MDLM 原版（直接复用其实现或保持行为等价）：

- DiT backbone 的构建和 checkpoint 加载（HuggingFace ID：`kuleshov-group/mdlm-owt`，context length 1024）
- Tokenizer：GPT-2 BPE
- Forward noising 的实现
- Logits 输出后 MASK 列的处理（置 -inf）
- log_softmax 的精度与时机
- Absorbing reverse step 的采样实现（包括 stay/unmask 概率公式、carry-over 行为）
- Linear schedule（alpha_t = 1 - t）相关计算
- Generative PPL 评估的具体协议（用同一份 GPT-2 large 评分器，相同序列长度截断）

teacher 直接加载 MDLM 原始 ckpt 即可。student 是在 MDLM backbone 上挂 LoRA + 加新的 head 模块。

---

## 2. 模型

### 2.0 Logits / log-prob 约定

**全套 log-prob 矩阵的构造方式（teacher 端、student 端、absorbing reverse step 输入端）都严格按照 MDLM 原版的处理方式，包括 MASK 列处理、log_softmax、carry-over 等所有细节。下面是具体规范：**

- Backbone forward 出 logits（不归一化）
- 输出层把 MASK 列的 logits 直接置为 `-inf`
- **任何把 logits 当 log-prob 用的地方（KL loss 的 teacher 端、absorbing reverse 采样的输入），先做一次 `log_softmax(logits, dim=-1)`** —— softmax 后 MASK 列变成 0
- Log-softmax 在 fp32 下计算（即使其他部分 bf16）
- 因为只有 MASK 列是 -inf，其他列有限，log_softmax 数值上稳定（不会出现 -inf - (-inf) 这种）

具体到三处：

| 用途 | logits 来源 | 是否需要 log_softmax |
|---|---|---|
| Teacher 输出作为 KL target | teacher backbone forward 出 logits → MASK 列置 -inf | 是，得到 `log_pT` |
| Student head 输出算 KL loss | compute_one_head 出 logits（已置 MASK 列 -inf） | 是，得到 `log_pS` |
| Absorbing reverse step 的输入 | 同上 | 是，得到 log_p；reverse step 内部再 exp 一次拿 probs 采样 |

Teacher 端建议封装一个 `teacher.forward_log_probs(z, t)` 函数，内部就是 `log_softmax(teacher.forward(z, t), dim=-1)` 加 MASK 列处理，返回 [B, L, V] 的 log-prob。后续都按它返回的是 log-prob 来用。

### 2.1 起点

从 MDLM 原始预训练 checkpoint 加载 backbone。同一份 checkpoint 也作为 teacher（冻结）。

### 2.2 Backbone

冻结所有原始 backbone 参数。在 backbone 的以下位置注入 LoRA：

- Attention 的 Q、K、V、O 投影
- FFN 的 up_proj 和 down_proj

Backbone LoRA rank：128。

### 2.3 MultiHead

`K = 4` 个 head，共享同一个 backbone 输出的 hidden。

每个 head 的设计：

- 共享一个 W_out（直接复用 MDLM 的 lm_head，冻结）
- 每个 head 用一组 per-head LoRA（rank 64）在 hidden 上做修正
- 通过 sinusoidal head-index embedding（dim = d_model）经过一个 2 层 MLP 后加到 hidden 上作为 head index 信号

Head h 的计算：
1. 取 hidden（[B, L, d]）
2. 加上 head-index embedding 的 MLP 输出（broadcast）
3. 过 head h 的 LoRA（rank 64）得到 hidden 的修正量
4. residual：hidden + 修正量
5. 过共享的 W_out 得到 logits [B, L, V]
6. 把 MASK 列的 logits 设为 -inf

Per-head LoRA 的 V 因子零初始化，MLP 最后一层零初始化。要求 K=1 时 student 输出等于 raw MDLM（这是必须通过的 sanity check）。

### 2.4 Teacher

加载同一份 MDLM 原始 checkpoint，eval 模式，所有参数 requires_grad=False。

---

## 3. 训练流程

### 3.1 单 step 流程

**核心设计：训练时 segment 大小固定 = 1/T_outer，与推理完全对齐**。这意味着：
- 训练时和推理时每个 substep 的大小 Δ = 1/(T_outer × K) 完全相同
- 训练时 K 个 head 学的是"如何走完一个 1/T_outer 长度的 segment"
- 一个训练 ckpt 对应一个目标 T_outer（不同 T_outer 需要不同 ckpt，跟 PiFlow 的 4-step / 8-step 分别训一致）

输入：一个 batch 的 x_0（ground truth 序列）

1. 把 [0, 1] 划成 T_outer 个等长的 outer segment，segment 边界 = `[0, 1/T_outer, 2/T_outer, ..., 1]`
2. 对每个样本随机选一个 outer segment（共 T_outer 个可选），记其起点为 `t_src`。等价于 `t_src ~ Uniform({1/T_outer, 2/T_outer, ..., 1})`
3. 固定 `Delta = 1 / (T_outer × K)`（**所有样本共享同一个 Delta**，不随 t_src 变）
4. `forward_noise(x_0, t_src)` 得到 z_t（每个 token 以概率 t_src 被替换为 MASK）
5. Student backbone forward 一次得到 hidden
6. 用 detached hidden 和 detached heads 做 K 步 rollout（详见 3.2）
7. 用带梯度的 hidden 和带梯度的 heads 算 loss（详见 3.3）
8. Backward 更新所有可训练参数（仅 LoRA + head 内部模块）

T_outer 是训练时的超参，Step 1 设 **T_outer = 4**。

### 3.2 Target 构造（rollout 部分，no_grad）

```
Delta = 1 / (T_outer * K)    # 固定值，比如 T_outer=4, K=4 时 Delta = 1/16
z = z_t
for h in 0..K-1:
    t_curr = t_src - h * Delta
    t_next = t_src - (h+1) * Delta   # 当 h=K-1 时 t_next 可能为 0，需 clamp 到 eps

    target[h] = teacher.forward_log_probs(z, t_curr)   # [B, L, V] log-prob, MASK 列 -inf
    mask[h] = (z == MASK_TOKEN)                         # [B, L] bool

    # 用 student 自己 detached 的 head h 推进 z
    detached_logits_h = student.heads.compute_one_head(detached_hidden, h)
    detached_log_p_h = log_softmax(detached_logits_h, dim=-1)    # 注意先 log_softmax
    z = absorbing_reverse_step(z, detached_log_p_h, t_curr, max(t_next, eps))

return target, mask
```

边界处理：
- 当 t_src 是最低 segment 起点（1/T_outer）时，最后一个 substep t_next = 0，需 clamp 到 eps
- 某个 substep 所有 mask 都是 False 时，该样本对该 head 的 loss 贡献为 0

### 3.3 Loss

按 head 顺序计算，每算一个 head 立即 backward（用 retain_graph=True 直到最后一个），避免同时持有 K 套 logits。

每个 head h 的 loss：

```
logits_h = student.heads.compute_one_head(hidden_with_grad, h)
log_pS_h = log_softmax(logits_h, dim=-1)   # fp32
log_pT_h = target[h]                        # 已经是 log-prob

p_T = exp(log_pT_h)
term = p_T * (log_pT_h - log_pS_h)
term = where(p_T > 0, term, 0)              # 处理 0 * -inf = NaN
term[..., MASK_TOKEN] = 0
kl_per_pos = term.sum(dim=-1)               # [B, L]

loss_h = (kl_per_pos * mask[h]).sum() / mask[h].sum().clamp(min=1)
weighted_h = (1/K) * loss_h
weighted_h.backward(retain_graph=(h < K-1))
```

### 3.4 Absorbing reverse step

输入：z（当前序列），log_p（已经 log_softmax 过的 log-prob，[B, L, V]，MASK 列为 -inf 即 prob 为 0），t（当前时间），s（目标时间，s < t）

```
对每个 MASK 位置：
  以概率 s/t 保持 MASK
  以概率 (t-s)/t 按 softmax(log_p) 采一个 token
非 MASK 位置 carry over（不做任何改动）
```

注意 t 需要 clamp(min=1e-5) 避免除 0。

由于 log_p 的 MASK 列是 -inf，softmax 后 MASK 列概率为 0，采样得到的 token 一定不是 MASK。

### 3.5 Forward noising

```
对每个 token 位置：
  以概率 t 替换为 MASK_TOKEN
```

### 3.6 Head divergence 诊断

固定一个 val batch（启动时采样并缓存，不重新采）。每 50 步算一次：

对每个 h ∈ {1, ..., K-1}：
- 在 val batch 的 z（已加噪）上 student backbone forward 一次
- 算 head 0 和 head h 的 log-prob
- 算 KL(head_0 || head_h)（注意方向是 head_0 在前），在 z 是 MASK 的位置上 mean
- log 这个值

---

## 4. 超参

| 参数 | 值 |
|---|---|
| K | 4 |
| T_outer（训练目标推理 NFE） | 4 |
| Backbone LoRA rank | 128 |
| Head LoRA rank | 64 |
| Optimizer | AdamW |
| LR | 1e-4 |
| LR schedule | linear warmup 1000 steps → cosine decay 到 0 |
| Betas, eps | (0.9, 0.95), 1e-8 |
| Weight decay | 0.0 |
| Grad clip | 1.0 |
| Batch size | 8（单卡，不做梯度累积） |
| 训练 steps | 30000 |
| 序列长度 | 1024 |
| Head loss 权重 w_h | uniform 1/K |
| 精度 | bf16 混合精度，fp32 master weights，loss 计算用 fp32 |
| 硬件 | 单卡 RTX 4090（24GB） |
| 随机种子 | 42 |

eps（数值下界）= 1e-5
MASK_TOKEN id = 50257
Vocab size = 50258
Hidden dim d = 768

---

## 5. 推理

输入：batch_size，序列长度 L，T_outer，K，device

```
z = 全 MASK 序列
times = linspace(1.0, 1e-5, T_outer * K + 1)

for outer in 0..T_outer-1:
    t_outer = times[outer * K]
    hidden = student.backbone(z, t_outer)        # ONE NFE per outer iteration
    for h in 0..K-1:
        t_curr = times[outer * K + h]
        t_next = times[outer * K + h + 1]
        logits_h = student.heads.compute_one_head(hidden, h)
        log_p_h = log_softmax(logits_h, dim=-1)   # 注意先 log_softmax
        z = absorbing_reverse_step(z, log_p_h, t_curr, t_next)

return z
```

总 backbone NFE = T_outer。
总反向 substep = T_outer * K。

---

## 6. Sanity 测试

实现完后必须先通过以下测试：

| 测试 | 通过标准 |
|---|---|
| K=1，LoRA 全零初始化的 student 输出 | 与 raw MDLM 输出最大绝对误差 < 1e-5 |
| 在随机 z_t, t_start 上构造 target | 无 NaN；MASK 列为 -inf；其余有限 |
| Student 等于 teacher 时 KL loss | 数值约 0（< 1e-4） |
| 随机输入下 KL loss | 有限值，无 NaN |
| K=4, B=8, L=1024 一次训练 step | GPU 内存峰值 < 22 GB（4090 24GB 留余量） |
| K=4, B=4, L=128 跑 1000 步 tiny 训练 | loss 单调下降，无 NaN |
| 推理生成 4 个样本 | 输出无 MASK token |
| 检查可训练参数总量 | ~22M（约占 169M 的 13%），不应接近 169M |

---

## 7. 训练时 logging

每 50 步 log：
- 总 loss
- 每个 head 的 loss（loss_h for h in 0..K-1）
- 每个 head 的 head divergence（KL(head_0 || head_h) for h in 1..K-1）
- 当前 LR

每 5000 步：
- 推理生成 8 个样本，dump 文本到 log
- Save checkpoint

---

## 8. 评估

### 8.1 评估对象

- 训练完的 MH-K4 student
- Baseline 1：MDLM 原始 checkpoint 在低 substep 数下推理
- Baseline 2：公开的 SDTT round-6 checkpoint

### 8.2 评估配置

MH-K4 模型主要在它训练时对齐的 T_outer = 4 上评估（推理时也用 T_outer=4，K=4，总反向 substep = 16）。

**额外做一组 T_outer 扫描诊断**：用同一个 T_outer=4 训出来的 ckpt，分别用 T_outer ∈ {1, 2, 4, 8} 推理，看跨 T_outer 的泛化能力。预期：
- T_outer = 4 性能最优（训练对齐）
- T_outer = 1, 2 时 Δ 大于训练时见过的 Δ，可能 overshoot
- T_outer = 8 时 Δ 小于训练时见过的 Δ，可能 undershoot

这个扫描是**诊断性的**，不是 headline 数字。Headline 数字看 T_outer = 4 上的表现。

MDLM baseline 扫总 substep ∈ {4, 8, 16, 32, 64}。

SDTT round-6 扫总 substep ∈ {4, 8, 16, 32, 64}。

每个 (模型, 配置) 组合生成 1024 个样本。

### 8.3 评估指标

定量：
- **Generative PPL**：按 SDTT 仓库（https://github.com/jdeschena/sdtt）`mode=eval` 分支里 `ppl_with_ar` 的实现来算，照搬即可
- **MAUVE**：按 SDTT 仓库 `mode=eval` 分支里 `mauve` 的实现来算，照搬即可

定性：
- 在 T_outer=4 配置上 dump 64 个样本到 plain text
- 人工读至少 16 个样本
- 按失败模式分类：mode collapse / 局部不通顺 / 长程不连贯 / token salad / subtle drift / clean

### 8.4 速度对比口径

主要比较：**MH-K4（T_outer=4, K=4，16 总 substep, 4 次 backbone NFE） vs SDTT round-6 在 16 substep 推理（16 次 backbone NFE）**。

如果质量接近，则获得 **4× backbone 加速**。

辅助对比：
- MH-K4 在 T_outer=4 下的 PPL vs MDLM 在 16 substep 推理的 PPL
- MH-K4 在 T_outer=4 下的 PPL vs SDTT round-6 在 16 substep 推理的 PPL

---

## 9. 提交清单

训练 + 评估完成后给出：

1. 每个 head 的 loss 曲线（30000 steps）
2. Head divergence 曲线（h ∈ 1, 2, 3）
3. Generative PPL 对比表：MDLM baseline（多 substep 数）、SDTT round-6（多 substep 数）、MH-K4（T_outer=4 主指标 + T_outer ∈ {1, 2, 8} 诊断）
4. MAUVE 对比表：同上结构
5. 64 个定性样本（plain text）+ 失败模式分类统计
6. 最终 step 的 head divergence 数值表
