# 实测环境与机器配置

> 本文件记录**跑出 `README.md` 里那些数字的那台机器**的软硬件配置与性能锚点，供复现与选型参考。
> **已脱敏**：不含用户名、主机名、邮箱、内网 IP、令牌；本机路径统一写成 `E:\MageVL\...` 这类形式。
> 表中每一个值都是在本机实测/查询得到的，采集命令附在每节末尾。

---

## 一、主机

| 项目 | 实测值 |
| --- | --- |
| 操作系统 | Windows 11 专业版 10.0.26200（build 26200，64-bit） |
| CPU | Intel Core i5-14600KF，**14 核 / 20 线程**，3500 MHz |
| 内存 | **31.8 GB**（2 × 16 GB DDR4-3600，双通道 —— 模组级细节见下） |
| GPU | NVIDIA GeForce RTX 5060 Ti，显存 **16311 MiB（15.90 GiB）** |
| 显卡驱动 | **616.56**（`nvidia-smi` 报告）/ 32.0.16.1656（Windows 设备侧） |
| 计算能力 | **sm_120（12.0）** —— Blackwell 架构，**必须 CUDA 12.8 及以上的 torch**（cu126 及以下不可用） |
| 页面文件 | `C:\pagefile.sys` **1024 MB** + `D:\pagefile.sys` **32768 MB**，且**自动管理已关闭** |
| 磁盘容量（采集时刻剩余） | C: 24 GB / D: 406 GB / E: 1932 GB |

```powershell
Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber,OSArchitecture
Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
Get-CimInstance Win32_PageFileUsage | Select-Object Name,AllocatedBaseSize
Get-PSDrive -PSProvider FileSystem | Select-Object Name,@{n='FreeGB';e={[math]::Round($_.Free/1GB,1)}}
```

### 内存（模组级）

> 频率/插槽来自 WMI；**时序与 XMP 档位来自 SPD 转储（Thaiphoon Burner），且只是其中一条** ——
> 两条 WMI 报告的 `PartNumber` / `Speed` 相同，但**另一条没有独立读取**。
> SPD 原始转储里的 `Serial Number` **已被剔除**，脱敏副本只存本地
> （`E:\MageVL\baseline\spd-JHD3600U1616JG-redacted.txt`），**不入库**。

| 项目 | 实测值 |
| --- | --- |
| 容量 / 条数 | 2 × 16 GB = 32 GB（两条同型号） |
| 型号 | JUHOR **JHD3600U1616JG**，DDR4 UDIMM；JEDEC DIMM 标签 `16GB 2Rx8 PC4-2400T-UB1-11` |
| 频率 | 标称 `Speed` **3600 MT/s** = 实际运行 `ConfiguredClockSpeed` **3600 MT/s** |
| 通道 / 插槽 | **双通道**：`Controller0-DIMMA2` + `Controller1-DIMMB2`（主板 4 槽，占用 2） |
| 组织 | 2 Rank ×8（2048M x64） |
| DRAM 颗粒 | **SK hynix H5AN8G8NCJR-UHC**，8 Gb **C-die（18 nm）** |
| JEDEC 档（SPD） | 1200 MHz（DDR4-**2400T** downbin）/ 时序 **17-17-17-39-55** / 1.20 V |
| XMP 档（SPD，XMP 2.0） | **XMP Certified**：1800 MHz（DDR4-**3600**）/ **16-20-20-38-80** / **1.35 V**；另有 XMP Extreme：1901 MHz（≈3802 MT/s）/ 18-22-22-42 |
| 本机实际档位 | OS 报告运行在 **3600 MT/s** → 已启用 1800 MHz 的 XMP 档；**由此推断**实际生效时序为 **16-20-20-38**（*推断*：OS 侧读不到实际 tCL/tRCD/tRP/tRAS） |
| 电压疑点 | WMI `ConfiguredVoltage` 报 **1200 mV**，与 XMP 档要求的 **1.35 V** 不一致 → 该字段**不代表实际 DRAM 电压**；实际时序/电压请以 BIOS 或 CPU-Z 的 *Memory* 页为准（**我未独立验证**） |
| 制造周次 | 2025 年第 22 周（单条 SPD 数据） |
| SPD / XMP 版本 | SPD 1.1 / XMP 2.0 |

```powershell
Get-CimInstance Win32_PhysicalMemory |
  Select-Object DeviceLocator,Capacity,Speed,ConfiguredClockSpeed,Manufacturer,PartNumber,SMBIOSMemoryType,ConfiguredVoltage
Get-CimInstance Win32_PhysicalMemoryArray | Select-Object MemoryDevices,MaxCapacityEx
```

## 二、软件栈

全部装在**独立 venv**里（`E:\MageVL\venv`），系统 Python 未被改动。

| 组件 | 版本 |
| --- | --- |
| Python | **3.12.7** |
| torch | **2.9.1+cu128**（CUDA 12.8；`torch.cuda.is_available() == True`） |
| transformers | **5.17.0**（Mage-VL 要求 ≥ 5.7） |
| ultralytics | **8.4.165**（YOLOv10n 训练/评估） |
| opencv-python | 5.0.0.93（`cv2.__version__` 报 `5.0.0`） |
| numpy | 2.5.2 |
| ffmpeg / ffprobe | Gyan.FFmpeg **9.0.2**（winget 安装，用户级 PATH） |

```powershell
E:\MageVL\venv\Scripts\python.exe -c "import sys,torch,transformers,ultralytics,cv2,numpy; print(sys.version.split()[0], torch.__version__, torch.version.cuda, transformers.__version__, ultralytics.__version__, cv2.__version__, numpy.__version__)"
ffmpeg -version | Select-Object -First 1
```

**缓存重定向**（避免把几 GB 缓存写进系统盘，属用户级环境变量）：

| 变量 | 指向 |
| --- | --- |
| `HF_HOME` | `E:\MageVL\hf` |
| `PIP_CACHE_DIR` | `E:\MageVL\pipcache` |
| `TORCH_HOME` | `E:\MageVL\torchcache` |

## 三、这台机器的三个硬约束（都是踩出来的）

1. **sm_120（Blackwell）** → torch 必须 **cu128+**；装 CUDA 版 torch 会在系统盘留下数 GB 缓存，务必先设 `PIP_CACHE_DIR`。
2. **页面文件被手工固定（C 盘仅 1 GB）** → **不要并发跑两个 CUDA 训练/评估进程**：会抛
   `OSError: [WinError 1455] 页面文件太小，无法完成操作 Error loading "cublas64_12.dll"`，
   而且**两个进程一起死**；放大器是 ultralytics 默认开 8 个 dataloader 子进程 → 训练与评估**严格串行**、评估显式 `--workers 0`。
3. **未安装 PowerShell 7（没有 `pwsh`）** → 只有 Windows PowerShell **5.1**。5.1 处理 UTF-8 文本有两个坑：
   含中文的 `.ps1` **必须带 BOM**；读 UTF-8 JSON 必须显式 `-Encoding UTF8`（否则中文键的字节对会吞掉后面的引号，`ConvertFrom-Json` 直接报错）。
   同理，含中文的 `.bat` **不要存成 UTF-8 无 BOM**（cmd 在 936 代码页下会按 GBK 解码，把残片当命令执行）。

## 四、本机性能锚点（同一个硬件上的实测值）

| 场景 | 实测 |
| --- | --- |
| Mage-VL（4.74B）图像推理 | 峰值 **10.51 GB / 15.90 GB**，约 **13.5 tok/s**，加载 8 ~ 80 s（取决于文件缓存） |
| Mage-VL 静态判「有人躺在地上」 | **0.66 s/张**（n=180，准确率 98.89%） |
| Mage-VL frames 视频 | 出厂像素预算下**上限 8 帧**；`max_pixels=64000` 后可跑 **64 帧** —— 瓶颈是注意力的 O(L²)，不是权重 |
| YOLOv10n 姿态检测器（640） | 推理 **0.7 ms/帧**；训练 batch=32 时显存 2.8 ~ 8.8 GB，约 **38 s/epoch**（7,098 张训练图） |
| bitsandbytes 4bit 加载 Mage-VL | 权重 8.83 GB → **3.39 GB**，但**解决不了视频 OOM** |

> 这些数字对应的脚本与原始输出都在本仓库 `scripts/` 与 `results/` 里，可逐条重跑。
