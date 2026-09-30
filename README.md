# Mage-VL 4.74B 端侧压缩：量化与剪枝实测

面向**端侧部署**的模型压缩工具集与实测记录。

本仓库现在只做一件事：**把 4.74B 的多模态模型压到边缘设备能承受的体积，并如实记录每一步的代价**
（体积 / 显存 / 延迟 / 精度）。

> 原先的「家庭场景跌倒检测」方向**已放弃**（实测结论：跨域泛化不成立，详见文末「历史」）。
> 那批探针脚本仍保留在 `scripts/` 下，不再维护、不再是本项目的目标。

> 本仓库只包含**代码与实测结果**。模型权重、数据集、虚拟环境一律不入库（见 `.gitignore`）。

---

## 一、结论速览

被测模型：`microsoft/Mage-VL`（4,741,793,792 参数 ≈ 4.74B，bf16，Apache-2.0），
结构 = Mage-ViT 视觉塔（24 层）+ 2 层 projector + Qwen3-4B-Instruct-2507 语言主干（36 层）。
全部数字在**本机 RTX 5060 Ti 16GB**上实测跑出，不是引用别人的论文。

| 变体 | 参数量 | 权重体积 | 加载后常驻 | 生成峰值 | 生成速度 | 质量 |
| --- | --- | --- | --- | --- | --- | --- |
| **bf16 基线** | 4.7418B (100%) | 8.83 GiB | 8.83 GB | 10.51 GB | 10.68 tok/s | 参照 |
| **INT8 · group64** | 4.7418B | **5.37 GiB** (61%) | 5.40 GB | 7.08 GB | 5.95 tok/s | 层相对误差 0.0054 |
| **INT4 · group64** ← 推荐 | 4.7418B | **3.53 GiB** (40%) | 3.56 GB | 5.25 GB | 4.37 tok/s | 平均 KL 0.0739 / top-1 一致 88.8% / 逐字 18.8% |
| **INT4 全量**（+词嵌入+lm_head） | 4.7418B | **2.46 GiB** (28%) | 2.50 GB | 4.98 GB | 3.71 tok/s | KL 0.1125 / top-1 82.2% / 逐字 12.5% |
| **剪枝：丢 6/36 层** | 4.1362B (87.2%) | — | 7.71 GB | 9.34 GB | 9.28 tok/s | 抽查 4 项全部答对 |
| **剪枝：FFN 宽度 75%（激活感知）** | 4.0694B (85.8%) | — | 7.59 GB | 9.26 GB | 8.66 tok/s | 抽查 3 项全部答对 |
| **剪枝：丢 3 层 + FFN 90%** | 4.1924B (88.4%) | — | 7.82 GB | 9.47 GB | 8.31 tok/s | 抽查 4 项全部答对 ✅ |
| **剪枝：丢 6 层 + FFN 75%** | 3.5759B (75.4%) | 6.68 GiB | 6.67 GB | 8.29 GB | 13.91 tok/s | ⚠️ **出现重复退化**（见 §2.3） |
| ↳ 再量化 INT4·g64 | 3.5759B (75.4%) | **2.92 GiB** | 2.95 GB | 4.58 GB | 2.3~3.0 tok/s | ❌ 质量已损坏 |
| **↳ 再量化 INT4 全量**（体积最小） | 3.5759B (75.4%) | **1.85 GiB** (21%) | 1.88 GB | 4.33 GB | ~2.3 tok/s | ❌ KL 1.658 / top-1 37.9% / 逐字 0% |

> 「权重体积」= 真正被加载的主干 safetensors 之和（不含 `streammind_gate.safetensors` 等旁支）。
> 「常驻」= 加载完成后 `torch.cuda.memory_allocated`；「峰值」= 一次生成过程中的 `max_memory_allocated`。
> KL / top-1 / 逐字一致 = **与自己的 bf16 原版**在同一批 16 个提示词上的偏离度，**不是**榜单准确率。

### 精度和原版相近吗：分档，而且差得很不平均

同一批提示词下的实际输出：

| 提示 | bf16 原版 | INT4·g64（3.53 GiB） | INT4 全量（2.46 GiB） | 剪枝+INT4（1.85 GiB） |
| --- | --- | --- | --- | --- |
| What is the capital of France? | `Paris` | `Paris` | `Paris` | `France` ❌ |
| What is 17 * 23? | `391` | `391` | `391` | 答错 ❌ |
| Describe this image in detail. | `a dog sitting on a patterned rug… Border Collie…` | `a medium-sized dog with a thick, fluffy coat… ears are large and erect…` | 措辞再变，仍是同一只狗 | `<image>` ❌ |
| 视频帧「发生了什么」 | `A man is holding a microphone and talking.` | `A man is holding a microphone and speaking.` | `A man holding a microphone with the BBC Sport logo…` | `None of the provided options.` ❌ |

**读法**：
- **3.53 GiB 这一档才谈得上"相近"**——意思与细节都在，但**措辞会变**（逐字相同仅 18.8%）。
- **2.46 GiB 仍可用**，代价可测：多花的 1.07 GiB 换来 KL 0.0739→0.1125（**+52%**）。
  → **别小看词嵌入与 `lm_head`**：它们只占参数量的 16%，量化后却贡献了一半以上的质量损失。
- **1.85 GiB 完全不接近**：KL 是原版的 **22 倍**，top-1 一致率掉到 37.9%，16 条**没有一条**逐字相同。
  它是**体积极限**，不是可用模型——省下的 6 GiB 是用模型能力换的，不补蒸馏/恢复训练就到不了"可用"。

---

## 二、三条最重要的实测结论

### 1. 量化省的是**内存**，在 GPU 上反而**更慢**

INT4(组内 64) 把线性层从 **8.11 GiB 压到 2.08 GiB**（1/3.9），整包 **8.83 → 3.53 GiB**，
加载后显存 **8.83 → 3.56 GB**、生成峰值 **10.51 → 5.25 GB**。
但在同一张卡上生成速度从 10.68 掉到 **4.37 tok/s（慢 2.4 倍）**——因为这里的实现是
「还原成浮点再 matmul」，反量化开销全暴露在关键路径上。

**端侧要的整数核（int4×int8 直接乘）在 GPU 上没有对应算子可测**；本机可测到的收益是
**内存与体积**，不是时间。这一条必须在选型时分开算，不能拿 GPU 延迟去推端侧延迟。

### 2. 「KL 很小」不等于「输出没变」——必须看自由生成

INT4 的平均 KL 只有 0.0739、top-1 一致率 88.8%，看上去"几乎无损"；
但同一批 16 个提示词做**自由生成**，与 bf16 逐字相同的只有 **18.75%（3/16）**，平均相似度 0.657。

> 只报 KL / top-1 一致率，会把 4bit 的代价说小。**压缩评估必须包含开放式生成的一致性。**

### 3. 结构化剪枝：丢层很便宜，**宽度裁剪的判据决定生死**

- **丢 6/36 层（均匀丢 5,10,15,21,26,31）**：参数量 87.2%，抽查 4 项（文本常识、文本解释、
  图像描述、图像细节）**全部答对**，质量肉眼几乎不掉。
- **FFN 宽度保留 75%**：结果**完全取决于选哪些神经元**——
  | 判据 | text00（应为 Paris） | img00 |
  | --- | --- | --- |
  | 纯权重范数 `‖gate_j‖·‖down_j‖` | **乱码** ❌ | **乱码** ❌ |
  | 取前 75% | `Sainte-Marie` ❌ | 退化成 "breed of breed of…" |
  | **随机取 75%** | `France` ✅ | `A dog sitting on a carpet.` ✅ |
  | **激活感知（Wanda 式）** | **`Paris`** ✅ | 描述连贯 ✅ |

  纯权重判据**比随机还差**：这类模型存在"权重很小、激活极大"的离群神经元，
  按权重排序恰好把它们整批丢掉。**判据必须含激活**——只用权重会得到比不做还差的结果，
  而这一点在只看参数量/体积的仪表盘上完全看不出来。

**那么安全线在哪？** 逐项抽查（16 项里的 4 项，文本常识 / 列表 / 解释 + 图像描述）：

| 配置 | 参数量 | 抽查表现 |
| --- | --- | --- |
| 丢 3 层 + FFN 90% | **88.4%** | ✅ 全部正常，无退化 |
| 丢 6 层 | **87.2%** | ✅ 全部正常 |
| FFN 75% | **85.8%** | ✅ 全部正常 |
| 丢 6 层 + FFN 75% | **75.4%** | ⚠️ 部分项**陷入重复**（"…and the world is just beginning to stir, and the world…"），
  且常识题答错（`17*23=1511`） |

→ **在不做任何恢复训练的前提下，本模型的可用压缩区间大约在 86~88% 参数量**；
再往下压，退化会从"答错"变成"输出崩坏"，而这**不是靠调阈值能救的**，得靠蒸馏或
用训练数据做恢复（本仓库都没做）。

**叠加要小心**：把 75.4% 的剪枝模型再上 INT4，产物只有 **2.92 GiB** 且能独立加载，
但抽查直接答非所问（`France.` / `None of the options.`）——**两个"还能用"的压缩叠一起
不等于"还能用"**，必须以端到端抽查为准。

⚠️ 抽查 ≠ 评测：上表只是几项肉眼检查，**不能**当作准确率数字。

---

## 三、工具链取舍（为什么自己写量化器）

在目标 venv（`E:\MageVL\venv`，Python 3.12.7）里做过实测探针，结论见
`results/compress_env_probe.txt`：

| 包 | 结论 |
| --- | --- |
| gptqmodel 7.5.0 / autoawq 0.2.9 / torchao 0.18.0 / hqq 0.2.8 | **有可用发行版** |
| llm-compressor | ❌ 无可用发行版（`No matching distribution`） |
| onnx 1.23.1 / onnxruntime 1.30.0 | ✅ 有 Windows 轮子（后续导出候选） |

> ⚠️ 本机到 PyPI 的链路会**间歇性 ConnectionReset(10054)**：同一个包两次探测可能一次 FAIL 一次 OK。
> 第一轮探针就因此得出过「GPTQ/AWQ 装不上」的**错误结论**，重试后推翻。**别信一次探测。**

真正的障碍不是安装，而是**架构适配**：gptqmodel / autoawq 都要求模型类按它们的约定暴露层结构，
而 Mage-VL 是 `trust_remote_code` 的**自定义架构**（`MageVLForConditionalGeneration` + 自定义
`modeling_mage_vl.py`）；能否直接套用**未验证**。torchao 的 int4 核依赖 Triton，Windows 上没有轮子。

因此本轮走**零新依赖**路线：自己写 group-wise 非对称 RTN weight-only 量化（纯 PyTorch），
好处是确定能跑、可复现、产出的 int 权重 + group scale 正是端侧运行时吃的格式；
bitsandbytes 0.50.2 作为对照基线。

---

## 四、用法

模型快照与产物位置（重资产一律放 `E:\`，不进仓库）：

| 路径 | 内容 |
| --- | --- |
| `E:\MageVL\Mage-VL\` | 官方权重与自定义建模代码 |
| `E:\MageVL\venv\` | Python 3.12.7 + torch 2.9.1+cu128 + transformers 5.17.0 |
| `E:\MageVL\compress\` | 量化/剪枝产物（**不入库**） |
| `results/compress_*.json` | 每次实测的原始记录（**入库**） |

```powershell
$py='E:\MageVL\venv\Scripts\python.exe'; $c='D:\program\模型优化\scripts\compress'

# 0) 基线（体积 / 显存 / 延迟）
& $py "$c\bench.py" --tag bf16

# 1) 量化（默认跳过 lm_head：词表 151936，量化它省 0.58GiB 但最伤精度）
& $py "$c\quant_rtn.py" --bits 4 --group-size 64 --tag int4_g64     # 也可 --bits 8
& $py "$c\quant_rtn.py" --load E:\MageVL\compress\int4_g64          # 独立加载验证（不需要原权重）

# 2) 量化前后质量对比（KL / top-1 一致率 / 生成一致率）
& $py "$c\eval_quality.py" --tag int4_g64 --bits 4 --group-size 64

# 3) 结构化剪枝
& $py "$c\prune_structured.py" --tag pruneL30 --llm-drop 6 --no-save
& $py "$c\prune_structured.py" --tag pruneL36_ffn75 --ffn-keep 0.75 --ffn-strategy wanda --no-save
& $py "$c\prune_structured.py" --tag pruneL30_wanda75 --llm-drop 6 --ffn-keep 0.75 --reload-check

# 4) 剪枝产物完整性校验（逐 key 比对，只读）
& $py "$c\verify_pruned_ckpt.py" --ckpt E:\MageVL\compress\pruneL30_wanda75

# 5) 组合：剪枝后再量化
& $py "$c\quant_rtn.py" --model E:\MageVL\compress\pruneL30_wanda75 --bits 4 --group-size 64 --tag pruned_int4
```

⚠️ **同一时刻只跑一个 CUDA 进程**（本机页面文件配置下并发会耗尽提交内存，见坑点 #73）。
⚠️ 含中文的 `.ps1` 必须带 UTF-8 BOM（坑点 #79）。

### 产物是**能被重新打开**的，不是一串数字

量化包用 meta device 建壳 + 灌入 int 权重的方式实现独立加载（自定义架构没法直接
`from_pretrained` 吃量化权重，所以自己写加载器）。已实测：

```
[load] 独立加载成功 {'n_swapped': 350, 'n_rest': 248, 'bits': 4, 'group_size': 64}；GPU 常驻 3.548GB / 15.902GB
[load] text00 → Paris
[load] img00  → The image features a medium-sized dog with a thick, fluffy coat...
```

即 `E:\MageVL\compress\int4_g64` **不需要原始 bf16 权重**就能加载并正常推理。
证据：`results/compress_loadcheck_int4_g64.json`。

剪枝产物同理自带全部权重与建模代码，回读后参数量一致（3.5759B）且能推理。

---

## 五、文件

```
scripts/compress/
├── common.py                 公共加载/输入构造/测量 + 写死的固定评测集（16 项，不依赖外部数据）
├── safetensors_total.py      量「真正被加载的主干权重」体积（别拿目录总大小当分母）
├── bench.py                  统一基准：体积 / 加载 / 常驻 / 峰值 / tok/s
├── quant_rtn.py              group-wise 非对称 RTN 量化（INT8/INT4）+ 存盘 + 独立加载
├── eval_quality.py           量化前后：KL / top-1 一致率 / 生成答案一致率
├── prune_structured.py       结构化剪枝：丢层（LLM/视觉）+ FFN 宽度（激活感知判据）
├── diag_prune.py             剪枝异常时的诊断（真实 traceback + 各层 config 实际取值）
├── verify_pruned_ckpt.py     checkpoint 逐 key 校验（排除"存盘存坏了"这一类原因）
└── desensitize_results.py    结果脱敏：本机绝对路径 → 占位符（先解析 JSON 再处理字符串）

results/compress_*.json|txt   上述每一步的实测输出
```

---

## 六、已知坑点（压缩专章）

1. **纯权重范数做 FFN 宽度剪枝会把模型打崩**，且比随机还差 → 判据必须含激活（Wanda 式）。
2. **丢层后必须重编号 `layer_idx`**：解码层构造时拿的是原始下标，而 KV cache 按新层数建槽，
   不重编号会在 `cache_utils.py` 的 `self.layers[layer_idx]` 处 `IndexError`。
3. **自定义建模代码用 `attention_mask.cumsum(-1)-1` 推 `position_ids`**：手工拼接续写 token 时
   必须同步延长 `attention_mask`，否则 rotary 的 cos/sin 长度对不上（实测报过 58 vs 33）。
4. **别把 `attention_mask` 只当成"补零标记"**——在本模型里它直接决定位置编码。
5. **产物目录不要拷原快照的 `model.safetensors.index.json`**：它指向的是原分片名，
   会让加载器去找不存在的 `model-0000x-of-0000y.safetensors`。
6. **量体积要量主干**：`model.safetensors` 之外还有 `streammind_gate.safetensors`（1.0 GiB）
   与 64 个旁支权重（`cls_net` / `mamba_model` / `pre_net` / `post_net`），把它们算进去会得到错的分母。
7. **PyPI 探测会假失败**（ConnectionReset）→ 同一个包至少探测两次再下结论。
8. **别用 `| Select-String` 包住 python 调用后看 `$LASTEXITCODE`**：管道会让退出码变成 Select-String 的，
   真失败会被掩盖。要看退出码就直接跑。

---

## 七、局限 / 本轮没做的事（照实说）

- **精度评估是"与自己的 bf16 比"，不是"与人类标注比"**：KL / 一致率衡量的是"量化改了多少"，
  不是"模型多聪明"。仓库里**没有**跑标准的 VLM 榜单（MMBench 等），所以不存在"压缩后准确率 X%"这种说法。
- **生成的图像描述只抽查了少数几项**，没有系统评测；剪枝那两行的"应答正确"是抽查结论，不是评分。
- 量化只做了 **weight-only RTN**，没做 GPTQ/AWQ 的误差补偿，也没做 activation 量化；
  group size 只测了 64，没有扫 32/128。
- **没有蒸馏、没有量化感知训练（QAT）、没有用训练数据做恢复（healing）**——剪枝掉的质量是净损失。
- **没有导出到任何端侧运行时**（ONNX / RKNN / NCNN / GGUF 都没做）。
  本轮的产出是"更小的权重 + 可复现的测量"，离"能烧进 IPC SoC"还差一整套导出与算子适配。
- 视觉塔未压缩（24 层原样保留）：它只占 0.32B，但对图像能力的影响没测过。
- `lm_head` 与 `embed_tokens`（合计约 1.45 GiB）**没有量化**，它们占了 INT4 包 3.53 GiB 的 41%。

---

## 八、历史：跌倒检测（已停用，仅存档）

2026-09-28 ~ 09-30 本项目曾用于「家庭场景跌倒检测的边缘可行性研究」。该方向的最终结论是负面的：
在 10 段**未见过的**室内视频上按视频 2 折交叉验证后，Mage-VL 判「有人躺在地上」只有 **27.3%**
（有利测试集上曾是 98.89%），YOLOv10n 姿态检测器 `lie` 召回 **14.4%**（同域曾是 89.2%）。
→ 视觉层不能承担"确认有人躺在地上"，判定权必须交给传感器。**该方向就此终止。**

相关脚本（`scripts/probe_*.py` / `build_*_yolo.py` / `train_yolov10n_posture.py` /
`eval_posture_detector.py` / `batch_*.ps1`）与 `results/probe_*.txt` 原样保留，
作为「同源高分不可外推」这一方法论教训的证据。

---

## 九、实测数据（模型权重不随仓库发布）

### 模型产物为什么不发

6 个压缩包合计 **22.9 GiB**，而 GitHub 的两条硬限制是（均为官方文档口径）：

| 限制 | 对 22.9 GiB 的影响 |
| --- | --- |
| git 仓库内单文件 **100 MiB** | 权重根本无法入库 |
| Release 单资产 **2 GiB** | 必须切片成 15 个分片 |

切片后**技术上能发**（已实测：`gh release download` 下来的目录可直接加载，不需要拼接或改名），
但**上传链路撑不住**：本机上行 19:43 实测 3.65 MiB/s、20:00 掉到 **0.2~0.52 MiB/s**，
且 `gh release upload` 在这个速率下会**挂着不动**——1.85 GiB 的分片传了 21 分钟只走 253 MiB，
既不报错也不退出。按 0.5 MiB/s 算要 10 小时以上，且随时可能卡死。

**结论：模型产物不发布。** 曾经建好又删除的 6 个 Release：
`v0.2.0-int4-g64`、`-int8-g64`、`-int4-all`、`-pruned-int4`、`-pruned-int4-all`、`-prune-wanda75`。
本机仍保有全部产物（`<ASSETS>\compress\`），需要时用 §四 的命令自行生成或就地取用。

### 发布的是实测数据

**数值一字未改**，只把本机绝对路径换成了占位符：

| 占位符 | 原值 |
| --- | --- |
| `<ASSETS>/` | 本机重资产目录（模型快照、量化/剪枝产物） |
| `<REPO>/` | 本仓库路径 |
| `<HOME>/`、`<PAGEFILE>`、`<PATH>/` | 兜底 |

脱敏脚本 `scripts/compress/desensitize_results.py`：**先解析 JSON 再处理字符串**，
避免 JSON 里双重反斜杠转义被改坏（处理后 21 个 JSON 全部仍合法）。

| 文件 | 内容 |
| --- | --- |
| `results/compress_bench_bf16.json` | bf16 基线：体积 / 加载 / 常驻 / 峰值 / tok-s + 16 项固定评测集 |
| `results/compress_quant_int4_g64.json`、`compress_quant_int8_g64.json` | INT4/INT8 量化：层相对误差、量化后权重体积、存盘比 |
| `results/compress_quant_int4_all.json` | INT4 全量（额外量化词嵌入 + `lm_head`） |
| `results/compress_quant_pruned_int4{,_all}.json` | 剪枝组合的两个量化包 |
| `results/compress_quality_*.json` | 逐项 KL / top-1 一致率 / 生成逐字一致率 / 答案相似度（**含两边原始答案**） |
| `results/compress_prune_*.json` | 剪枝：参数量、丢的层号、FFN 保留数、16 项生成结果 |
| `results/compress_loadcheck_*.json` | 独立加载验证（不依赖原始 bf16 权重） |
| `results/compress_env_probe.txt` | 量化工具链在本机 venv 下的可用性探测原始输出 |

### 自己复现 / 自己造产物

```powershell
$py='<VENV>\Scripts\python.exe'; $c='<REPO>\scripts\compress'
& $py "$c\bench.py" --tag bf16
& $py "$c\quant_rtn.py" --bits 4 --group-size 64 --tag int4_g64 --shard-gib 1.9
& $py "$c\quant_rtn.py" --load '<ASSETS>\compress\int4_g64'          # 独立加载验证
& $py "$c\eval_quality.py" --tag int4_g64 --bits 4 --group-size 64
& $py "$c\prune_structured.py" --tag pruneL30_wanda75 --llm-drop 6 --ffn-keep 0.75 --ffn-strategy wanda --reload-check
& $py "$c\desensitize_results.py"                                     # 数据脱敏
```

⚠️ 同一时刻只跑一个 CUDA 进程。所有 KL / 一致率都是**与自己的 bf16 原版比**，不是榜单准确率（见 §一 表下注）。

**来源与许可**：全部数字来自 `microsoft/Mage-VL`（**Apache-2.0**）；派生产物遵循同一许可，
请一并遵守其模型卡条款并保留署名。

---

## 许可

代码与文档：见仓库设置。模型与数据遵循各自原仓库许可（Mage-VL 为 Apache-2.0）。
