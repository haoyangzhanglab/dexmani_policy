# GReST：几何区域—关系场景编码器

**Geometric Region–relation Scene Tokenizer (GReST)** 将 GAPT 的空间特征组织为少量区域，再在压缩空间中编码区域关系。代码类为 `GeometricRegionRelationTokenizer`，仍位于 `agents/obs_encoder/interaction/scene_context.py`，不再保留旧 `SceneContextPool` 别名或旧权重兼容逻辑。

研究问题是：在有限 scene token 预算下，区域内容与整体布局能否更好地支持灵巧操作。一个物体可对应多个区域；不声称每个 slot 就是物体、接触点或跨帧跟踪 ID。

## 分工与不变量

```text
GAPT -> 共享 geometry_projection -> [BT,M,D]
                |-> GReST -> [BT,K,D] scene tokens
                |-> 原 HandSceneRelationEncoder -> [BT,6,D]
scene + hand + state -> [B,T*(K+7),D] -> 原 DiTX/RectifiedFlow
```

GReST 只消费 projected patch features、同一米制坐标系的 XYZ 和有效位。hand reader 仍读取全部 M 个 patches 及其原始成员索引。触觉不进入 GReST；`query/late/none`、逐指门控、动作监督、GAPT、动作主干和 NFE 保持原样。`scene.mean(dim=1)` 仍进入原模态门控。

## 最小结构

1. 唯一索引 FPS 生成 K 个覆盖种子。每个 patch 只参与最近 `nearest_slots` 个种子的竞争，种子本身固定归属自己的区域，避免空槽。空间距离平局按输入索引处理；不保证跨帧身份一致性。
2. 对 dense features 只做一次低维投影；两轮原型更新在 `route_dim` 维完成。对每个输入 patch，沿 slot 轴归一化，满足 `sum_k A[k,i]=1`；padding 的分配为零。
3. 使用同一个归一化权重聚合完整特征、质心和协方差。几何描述只包含质心 3 维与协方差上三角 6 维，不默认引入高维交叉矩或重建器。
4. 在 K 个区域上做一层窄关系 attention，加入相对 XYZ/距离的 bias **和 edge value**，随后逐 slot FFN。不同区域可远距离交互；不再回读全部 dense patches，不增加独立 relation tokens。

### 本次微调：较缓的距离先验

令 `u_i` 为归一化的低维 patch 特征、`q_k` 为区域原型、`a_k` 为固定种子：

```text
logit[k,i] = alpha * dot(q_k,u_i) - log1p(||p_i-a_k||^2 / geometry_scale^2)
A[:,i] = softmax(logit[:,i], dim=slot) within allowed candidates
0 < alpha = 4*sigmoid(parameter) < 4
```

相较 v3 原型的 Gaussian 距离项，Cauchy 形式衰减更慢，降低稀疏预算或较大工作空间下在特征评分参与前就数值饱和的风险。最近种子候选仍限制混合范围。该先验不是接触概率，也未被证明在任务成功率上优于 Gaussian。

关系 attention 的 Q/K 使用逐头 LayerNorm，head dimension 必须至少为 2；保留 0.1 初始残差增益。单个 patch/单个 slot 情况仍可运行；K=1、K=M 或 `nearest_slots=1` 等退化情况下，路由学习信号可能消失，这是分配约束的性质，不是物体发现。

### 几何与诊断语义

`W[k,i]=A[k,i]/sum_i A[k,i]`，然后：

```text
pooled[k] = sum_i W[k,i]*z_i
center[k] = sum_i W[k,i]*p_i
cov[k]    = sum_i W[k,i]*(p_i-center[k])*(p_i-center[k])^T
```

距离、softmax、加权和和直接中心化协方差以 FP32 计算。XYZ 始终保留米制；`position_scale`/`geometry_scale` 只缩放特征计算。协方差描述 patch 中心分布，不是物体真实占据体积；`scene_mass` 是分配权重和，不是面积、物理质量、重要性或置信度。

`return_intermediate=True` 时，`InteractionObsEncoder` 在 `aux['scene']` 下返回分配、候选掩码、seed、基础聚合特征与关系 attention。`scene_center` 是基础区域质心；`assignment` 仅解释本层直接聚合，不能作为最终 token 的完整感受野或因果解释：GAPT 及关系层均已传播上下文。

## 配置与 token 预算

唯一预算入口为 `agent.num_scene_tokens`。默认 **12** 是初始折中，不是最优结论，也不需要是 2 的幂。支持任意整数 `1 <= K <= 有效patch数`。`agent.scene_tokenizer` 只包含路由/几何/关系参数，禁止在其中再次配置宽度或预算。

```yaml
agent:
  num_scene_tokens: 12
  scene_tokenizer:
    route_dim: 64
    routing_steps: 2
    nearest_slots: 3
    geometry_scale: 0.10
    position_scale: 1.0
    use_relation: true
    relation_dim: 64
    relation_heads: 4
    mlp_ratio: 2.0
    validate_inputs: true
```

两帧观测时，K=8/12/16/24 分别产生 30/38/46/62 个总条件 tokens。原版已是 192->16，不能把这项既有压缩当成新贡献。K=12 相比原 16 个 scene tokens，scene 数少 25%，总条件数由 46 降到 38；这不是端到端加速比例。

DDP 配方继承 `/interaction_flow`，不另设 scene 参数。参数张量没有 K-sized slot embedding，因此不同 K 的权重形状兼容；**形状兼容不等于免训练切换预算后性能不变**。正式比较宜分别训练或在验证集评估预算迁移。

默认保持输入校验；`validate_inputs=false` 只能用于上游已满足有效点数量/有限值约束的受控性能测试，不能作为跳过坏数据的办法。当前 agent 仍只编译原有动作主干与 GAPT dense blocks；不额外改变全局编译策略。

## 训练、消融和恢复边界

第一阶段仍使用原 flow-matching 目标，不加世界模型、对象监督、语言条件或未来输入。默认模型是动作监督训练的区域聚合器，不是显式 task-conditioned tokenizer。

先固定 K=16，与原版及 v3 原型比较，避免同时改变架构和预算。再比较 K=8/12/16/24；消融 `use_relation=false`、单轮路由、去特征评分和 Gaussian 先验。参数匹配基线应排除单纯轻量化效应；不要把所有机制开关做成长期公共框架。

重点评价干扰物变化、目标/容器相对布局、物体朝向及需要多指协调的任务。执行 scene 屏蔽测试时，要区分只替换动作条件与同时替换门控输入，避免将两条路径的影响混淆。未来几何预测可作为后续训练期辅助任务，但本提交未实现，也不宣称已有世界模型能力。

**旧 SceneContextPool checkpoint、EMA 和 optimizer 不能严格恢复到 GReST。** 使用新实验目录；当前目录和 W&B 命名加入 `_grest${agent.num_scene_tokens}`。同一 K 的不同消融仍需独立 run/name，不将结构不兼容实验当作 resume。历史模型应使用原源码和保存配置。

## 验证

定向测试（仅需要 Python、PyTorch、pytest、PyYAML）：

```bash
python -m pytest -q tests/test_grest_scene_tokenizer.py tests/test_grest_interaction_wiring.py
```

本次 CPU/PyTorch 2.10.0 环境实际结果：**79 passed，1 skipped（CUDA 不可用）**。覆盖预算、候选支撑、分配归一化、padding、重复坐标、梯度、BF16、CPU eager 后端 fullgraph 捕获、配置字段和 DDP 继承。接线测试执行 production encoder/agent 的类体，但 GAPT、hand/tactile 与动作主干使用明确标注的测试替身；不代表这些上游模块或真实策略的联合验证。

完整仓库依赖可用后还需运行：

```bash
python dexmani_policy/smoke_test.py --config-only interaction_flow
python dexmani_policy/smoke_test.py --config-only ddp/interaction_flow
python dexmani_policy/smoke_test.py interaction_flow --max-updates 4
```

本次未运行完整 Hydra smoke、GPU/Inductor、实际训练、DDP、闭环评测或真机动作。容器无法直接 clone GitHub，修改基于 connector 读取的 `main@e240852`；关键调用方与配置的原始 blob SHA 已核对。不得将局部参数/序列长度下降当作已测得的硬件时延或成功率改善。

## 文献来源与采用边界

- [SceneTok](https://arxiv.org/abs/2602.18882)：借鉴固定规模紧凑 scene latent；不引入其生成式渲染/场景扩散系统。
- [HiSC](https://arxiv.org/abs/2608.04610)：借鉴几何与特征共同组织压缩；不复刻多视角图连通/LLM 内层次剪枝，也不宣称实例完整性保证。
- [LiLa-WAM](https://arxiv.org/abs/2608.03701)：借鉴控制目标塑造紧凑表示；暂不移植依赖冻结视觉目标的未来预测系统。
- [SlotVLA](https://arxiv.org/abs/2511.06754)：借鉴内容/关系分工；其物体监督与跟踪前提在本模型中不存在。
- [Object-Centric Representations Improve Policy Generalization in Robot Manipulation](https://arxiv.org/abs/2505.11563)：以结构化表示的泛化作为实验问题，不将预训练 OCR 结果直接归因于从零训练的 GAPT。

以上为设计来源，不是对原论文的复现或对 GReST 性能的证明。
