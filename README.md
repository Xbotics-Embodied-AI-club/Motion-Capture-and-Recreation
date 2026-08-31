# Motion-Capture-and-Recreation（身外化身 · 动作捕捉与复刻）

> **项目简介**：基于单目 RGB 视频的人体动作捕捉与机器人复刻系统。通过 WHAM 从视频重建 3D 人体（SMPL-X），经 GMR 运动学重映射到机器人（默认灵龙 2.0），在 MuJoCo 中渲染演示视频，并输出关节角度序列（CSV）。
>
> **技术链路**：`输入视频 → 人体检测 → WHAM 3D 重建 → GMR IK 重映射 → MuJoCo 渲染 → CSV + 演示视频`
>
> **最终产出**：`live_motion.csv`（37 列关节角序列）+ MuJoCo 演示视频（mp4）

---

## 仓库内容（已就绪，可直接运行）

> ✅ **灵龙 2.0 的资产与代码改动已全部导入本仓库**：模型资产、GMR 注册、IK 映射配置、贴地修正等均已完成。**环境配置好之后，直接跑即可，无需再手动接入灵龙。**

```
├── assets/LingLong2.0/       # 灵龙 2.0 模型资产（URDF/MJCF/场景/网格，已导入）
├── assets/<其他机器人>/       # 参考机器人模型（unitree_g1、openloong 等）
├── general_motion_retargeting/  # GMR 重映射核心（params.py 注册、ik_configs 映射表）
├── lib/                      # WHAM 相关库（检测/提取/重建）
├── scripts/                  # 辅助脚本（smplx_to_robot、pkl 转换、批处理等）
├── examples/                 # 示例：linglong2_demo.mp4（灵龙演示视频）
├── video_input/              # 示例输入视频 dateset_vedio.mp4
├── run.ps1 / run.sh          # 运行入口（Windows / Linux）
├── run_test.bat              # 默认机器人验证脚本（unitree_g1）
├── run_linglong.bat          # 灵龙 2.0 运行脚本
└── demo.py                   # 离线串行入口（含 DPVO 全局 SLAM 支持）
```

> **需要自行下载（不入库）**：模型权重（`checkpoints/`、`assets/body_models/`，从 ModelScope 下载）；第三方库（`third-party/`，clone 后按 `.gitmodules` 拉取 DPVO / ViTPose）。

---

# 第一部分 · 环境配置

## 1. 下载模型权重

从 ModelScope 下载权重，放入仓库根目录：

```
https://modelscope.cn/models/Xbotics_Embodied_AI_Club/Robot-imitation-learning/files
```

```
Robot-imitation-learning-wham_gmr_win/     ← 仓库根目录
├── checkpoints/              ← WHAM 权重、vitpose_base_coco_aic_mpii.pth、yolov8n.pt 等 9 个文件
├── assets/
│   └── body_models/          ← SMPL-X 模型（smplx/ 下 SMPLX_FEMALE/MALE/NEUTRAL.npz）
└── dataset/                  ← 训练数据（初赛离线推理非必需，可先不放）
```

> ⚠️ 权重是运行必需，缺失会报 `Failed to load` / 模型加载错误。路径避免中文。

## 2. 系统要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10 / 11（也可用 Linux） |
| 显卡 | NVIDIA 独立显卡，建议显存 ≥ 8GB（4GB 需半精度 + 降分辨率） |
| 显卡驱动 | 支持 CUDA 11.3+（572.83 / CUDA 12.8 已验证） |
| 必备软件 | Anaconda 或 Miniconda |

**不需要安装**：❌ Visual Studio / VS Build Tools（不编译 C++）；❌ CUDA 11.3 Toolkit（不编译 pytorch3d/DPVO 就不需要 nvcc）。

## 3. 可选模块（不影响核心链路）

| 模块 | 作用 | 不装的后果 |
|------|------|-----------|
| pytorch3d | WHAM 重建的 3D 可视化窗口 | 少一个预览窗口，不影响任何输出文件 |
| DPVO | 全局 SLAM（相机自身运动，适合手持移动拍摄） | 人体在局部坐标重建，**30 个关节角完全不受影响**；建议 Linux 上安装 |

**核心链路**：视频 → WHAM 重建 → GMR 重映射 → MuJoCo 渲染 → CSV/视频，全程 pip 预编译包，零编译零 VS。

## 4. 安装步骤

> 所有命令在 **Anaconda Prompt** 中执行。VSCode PowerShell 需先 `conda init powershell` 并重开终端。

### 4.1 创建 Python 环境

```bash
conda create -n wham_gmr python=3.10 -y
conda activate wham_gmr
```

### 4.2 安装 PyTorch 1.11（CUDA 11.3）

```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

期望输出：`1.11.0 True`（False 则检查显卡驱动）。

### 4.3 降级 numpy（⚠️ 关键）

```bash
pip install numpy==1.22.3
```

> conda 默认装 numpy 2.x，与 torch 1.11 ABI 不兼容，运行时会崩溃。必须降到 1.22.3。

### 4.4 固定 setuptools + 安装 mmcv（预编译 wheel）

```bash
pip install setuptools==59.5.0
pip install https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/mmcv_full-1.5.0-cp310-cp310-win_amd64.whl
```

> mmcv **必须装 1.5.0**（ViTPose/mmpose 0.24.0 要求 1.3.8~1.5.0，装 1.7.1 报 AssertionError）。直接装 wheel 绕过 mim。

### 4.5 安装 WHAM 依赖

```bash
pip install -r requirements.txt
python -c "import numpy; assert numpy.__version__=='1.22.3'; print('numpy OK')"
```

> ultralytics 可能把 numpy 顶回 2.x，被改则 `pip install numpy==1.22.3 --force-reinstall`。

### 4.6 安装 ViTPose（仓库根目录执行）

```bash
pip install --no-build-isolation chumpy==0.70 json-tricks
pip install -v -e third-party/ViTPose
```

### 4.7 修改 setup.py（两处必改）

用 VSCode/记事本打开根目录 `setup.py`：

**① 修复编码**（否则 `pip install -e .` 报 GBK 错误）：

```python
long_description=open("README.md").read(),
# 改为
long_description=open("README.md", encoding="utf-8").read(),
```

**② 去掉 proxqp 依赖**（GMR 实际用 daqp，proxqp 在无 VS 环境编译必失败）：

```python
"qpsolvers[proxqp]",
# 改为
"qpsolvers",
```

### 4.8 安装 GMR 本体

```bash
pip install daqp
pip install -e .
```

> daqp 是 GMR 的 IK 求解器，PyPI 有 Windows 预编译包；失败可 `conda install -c conda-forge daqp`。

### 4.9 环境验证（7 条全绿即就绪）

```bash
python -c "import numpy; assert numpy.__version__=='1.22.3'; print('numpy OK:', numpy.__version__)"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import smplx; print('SMPL-X OK')"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import mujoco; print('MuJoCo OK')"
python -c "import mink; print('mink IK OK')"
python -c "from general_motion_retargeting import GeneralMotionRetargeting; print('GMR OK')"
```

> `xrobotoolkit_sdk not found, skip for now` 为无害警告。

## 5. 常见问题速查

| 问题 | 症状 | 解决 |
|------|------|------|
| numpy 版本不对 | import torch 报 NumPy 1.x/2.x 兼容错误 | `pip install numpy==1.22.3 --force-reinstall` |
| mmcv 版本不兼容 | `AssertionError: MMCV==1.7.1 ... <=1.5.0` | 重装 1.5.0（见 4.4 wheel 地址） |
| proxsuite 编译失败 | `pip install -e .` 报 CMake `'nmake' failed` | 按 4.7② 去掉 proxqp，用 daqp |
| `pip install -e .` GBK 报错 | `UnicodeDecodeError: 'gbk'` | 按 4.7① 加 `encoding="utf-8"` |
| 视频打不开 | `Failed to load video` | 视频复制为英文路径 |
| conda 命令找不到 | 普通终端输入 conda 提示不是内部命令 | 用 Anaconda Prompt；VSCode 先挂 conda-hook.ps1 |
| cmd 跨盘 cd 失败 | `cd D:\xxx` 后提示符没变 | 用 `cd /d D:\xxx` |
| PowerShell set 无效 | `set OUTPUT_ROOT=...` 脚本仍用默认值 | PowerShell 用 `$env:OUTPUT_ROOT='...'` |

---

# 第二部分 · 跑通流程（环境配置好之后）

> 本仓库代码与灵龙 2.0 资产已导入完成，环境就绪后按下面两步即可跑通。

## 1. 先验证环境：默认机器人 unitree_g1

**① 准备视频**：把测试视频复制为英文路径 `examples/dataset_video.mp4`。

**② 运行** `run_test.bat`（已内置在仓库，编码 ANSI）：

```bat
@echo off
call D:\anaconda\Scripts\activate.bat wham_gmr
set OUTPUT_ROOT=output/test_run
set ROBOT=unitree_g1
set RECORD_GMRVIDEO=1
set RECORD_WHAMVIDEO=0
set VIDEO=examples/dataset_video.mp4
powershell -ExecutionPolicy Bypass -File run.ps1
```

Anaconda Prompt 中进入仓库根目录执行 `.\run_test.bat`。

**③ 预期结果**：模型加载（1~3 分钟）→ 逐帧处理 → MuJoCo 窗口弹出机器人跟随动作 → 自动停止。

输出（`output/test_run/` 下）：

```
pkl/my_motion.pkl            # 动作序列（pkl）
csv/live_motion.csv          # 关节角度序列（37 列 = 根位姿 7 + 30 关节角）
video/live_stream_robot.mp4  # MuJoCo 演示视频
```

> 目的：确认环境与链路通畅。显存不足 8GB 追加 `set WHAM_USE_AMP=1`、`set WHAM_INPUT_SCALE=0.5`。

## 2. 直接运行灵龙 2.0

环境验证通过后，直接运行 `run_linglong.bat`（灵龙已导入，无需任何额外配置）：

```bat
@echo off
call D:\anaconda\Scripts\activate.bat wham_gmr
set OUTPUT_ROOT=output/linglong2_run
set ROBOT=linglong2
set RECORD_GMRVIDEO=1
set RECORD_WHAMVIDEO=0
set VIDEO=examples/dataset_video.mp4

:: 灵龙 2.0 专用优化参数
set GMR_MAX_ITER=20
set GMR_WORKER_QUEUE_SIZE=64

powershell -ExecutionPolicy Bypass -File run.ps1
```

> `GMR_MAX_ITER=20` 控制 IK 最大迭代次数；`GMR_WORKER_QUEUE_SIZE=64` 控制 worker 队列深度。

**完整数据流**：

```
输入视频
  → reader 逐帧读取
  → detector 检测人体
  → extractor 裁剪 ROI
  → WHAM 重建 3D 人体（SMPL-X 姿态）
  → GMR IK 重映射（灵龙 30 关节）
  → 输出 live_motion.csv（37 列）+ MuJoCo 演示视频
```

**验证输出**：

| 输出文件 | 检查项 | 合格标准 |
|---------|--------|---------|
| `csv/live_motion.csv` | 行数 = 帧数，每行 37 列 | `pos3 + quat_xyzw4 + dof30` |
| `video/live_stream_robot.mp4` | 视频正常生成 | 文件大小 > 0，能正常播放 |
| 观感检查 | 身体朝向正确 | 与输入视频一致 |
| | 脚是否贴地 | 脚底不陷入地面 |
| | 深蹲时手臂是否自然 | 小臂不异常弯曲 |

> ✅ 本仓库已内置灵龙的两处已知修复：脚底离地间隙 `GROUND_CLEARANCE_DICT` 与手臂旋转主导权重（详见第三部分 5 节），直接运行即为调优后的效果。

---

# 第三部分 · 如何引入新的机器人资产（参考）

> 本仓库以**灵龙 2.0** 为例，已经走通"新机器人接入"的完整流程。如果之后要接入其他机器人（或替换为新的 URDF），按下面 5 步操作即可。

## 1. URDF 接入

把新机器人官方 URDF + meshes 放入 `assets/<机器人名>/`，最终形态：

```
assets/LingLong2.0/  （示例：灵龙 2.0）
├── LingLong2.0.urdf      # 官方 SolidWorks 导出（原始）
├── LingLong2.0.xml       # 转好的 MJCF（手工修正浮动基座 + yaw 轴）
├── scene.xml             # 场景包装（include + 地面 + 灯光 + 相机）
└── meshes/               # 全部 STL 网格（与 urdf 同级）
```

**官方 URDF 通常不能直接用**，常见三个原因：

| 问题 | 原因 | 后果 |
|------|------|------|
| ❌ 没有浮动基座 | 官方 URDF 只有 30 个关节 | GMR 的 `KinematicsModel` 需要 `nq = 37`（7 浮动基座 + 30 关节），报 nq 不匹配 |
| ❌ yaw 关节缺 `axis` | 部分 yaw 关节没有 `axis` 属性 | MuJoCo 解析直接崩溃 |
| ❌ mesh 路径问题 | 官方 URDF 有 `meshdir="meshes/"` 指令 | MuJoCo 拼出 `meshes/meshes/` 双前缀，找不到网格 |

## 2. URDF → MJCF 转换与修正

**① 转换脚本**（仓库根目录新建 `.py` 文件运行）：

```python
import mujoco
import os

# ========== 修改这里 ==========
urdf_path = '你的URDF路径/LingLong2.0.urdf'
out_path = '你想保存的 MJCF 路径/LingLong2.0.xml'
# =============================

# 1. 读取原始 URDF
with open(urdf_path, 'r') as f:
    urdf_text = f.read()

# 2. 去掉 meshdir 指令，避免 meshes/meshes/ 双前缀
urdf_text = urdf_text.replace('meshdir="meshes/"', '')

# 3. 写临时文件
urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
tmp_path = os.path.join(urdf_dir, '_tmp_LingLong2.urdf')
with open(tmp_path, 'w') as f:
    f.write(urdf_text)

# 4. MuJoCo 解析并保存 MJCF
try:
    model = mujoco.MjModel.from_xml_path(tmp_path)
finally:
    os.remove(tmp_path)  # 临时文件用完自动删除

mujoco.mj_saveLastXML(out_path, model)
print('✅ 转换成功！nq=', model.nq, 'nu=', model.nu)
```

> ⚠️ 运行前确保输出目录已存在。期望输出：`nq=30`、`nu=30`（此时尚无浮动基座，正常）。

**② 手工修正 MJCF（三处）**：

- **补浮动基座**：在 `<body name="base_link" ...>` 下加 `<freejoint name="base_free"/>`，`nq` 从 30 → 37；
- **补 yaw 关节轴**：给缺 `axis` 的 yaw 关节补 `axis="0 0 1"`（灵龙需补 6 个：`waist_yaw`、`head_yaw`、左右 `shoulder_yaw`、左右 `wrist_yaw`）；
- **保留零位姿特性**：确认 body 只有 `pos` 没有 `quat`（零位姿全局旋转 = 单位阵，配 IK offset 时关键，不要手动加 quat）。

**③ 场景包装**：在资产目录新建 `scene.xml`，复制 `assets/openloong/scene.xml` 模板，把 include 文件名改成新机器人的 MJCF。

## 3. 注册到 GMR

**① `general_motion_retargeting/params.py` 四处注册**（照 openloong 条目格式，各加一行）：

| 字典名 | 含义 | 示例值（灵龙） |
|--------|------|--------|
| `ROBOT_XML_DICT` | 机器人 MJCF 路径 | `"linglong2": "assets/LingLong2.0/scene.xml"` |
| `IK_CONFIG_DICT["smplx"]` | IK 映射配置文件 | `"linglong2": "smplx_to_linglong2.json"` |
| `ROBOT_BASE_DICT` | 根 body 名 | `"linglong2": "base_link"` |
| `VIEWER_CAM_DISTANCE_DICT` | 查看器相机距离 | 参考其他机器人，如 `3.0` |

**② 创建 IK 映射配置**（`general_motion_retargeting/ik_configs/` 下新建 `<机器人名>.json`）。每条记录格式：

```
[human_joint, pos_weight, rot_weight, pos_offset(xyz), rot_offset(wxyz)]
```

灵龙经验证有效的配置（可作模板）：

| 部位 | 映射（human → robot link） | table1 权重 pos/rot | table2 权重 pos/rot |
|------|---------------------------|--------------------|--------------------|
| 根 | `pelvis` → `base_link` | 100 / 10 | 100 / 5 |
| 腿 | `left_hip` → `left_hip_roll_link` 等 | 位置主导 | 位置主导 |
| 手臂 | `left_shoulder` → `left_shoulder_yaw_link` | **0 / 10** | **10 / 5** |

> ⚠️ 关键调优：手臂权重「位置主导 → 旋转主导」是最关键的一步（原理见第 5 节）；肩映射对象用 `yaw_link`（上臂方向的最后一个 link），不是 `pitch_link`。

## 4. 跑通与验收

新建运行脚本（参照第二部分 `run_linglong.bat`，改 `ROBOT` 与输出目录），运行后按第二部分的"验证输出"表验收。

## 5. 已知问题与修复（通用经验）

### 5.1 脚底陷入地面

- **症状**：机器人脚底低于地面，陷进地板。
- **根因**：贴地逻辑 `q[2] -= lowest_height` 用 `forward_kinematics` 返回的 **body 帧原点**最低 z（脚踝处），不是 mesh 几何最低点。灵龙脚底 mesh/圆柱在 `pos="0.02 0 -0.069"`，比脚踝帧低约 7cm；脚踝贴到 z=0 后脚底实际在 z≈-0.07。（g1 有独立 `toe_link` body 在脚底，不受影响。）
- **改法（三处联动）**：

  第 1 处：`params.py` 加配置
  ```python
  GROUND_CLEARANCE_DICT = {"linglong2": 0.075}
  ```

  第 2 处：`scripts/smplx_to_robot_stream.py` 的 `OnlineQposPostprocessor`，给 `init` 加 `ground_clearance` 参数，贴地逻辑改一行：
  ```python
  # 原来
  q[2] -= lowest_height
  # 改成
  q[2] -= (lowest_height - self.ground_clearance)
  ```

  第 3 处：`handle_wham_gmr.py` 加 `_ground_clearance_for_robot()` 方法，把值传进两处 postprocessor 构造。

- **原理**：把最低 body 帧抬到 z=+0.075 而不是 z=0，脚底刚好落 z≈0。0.075 单点可调，其余机器人默认 0.0，行为不变。

### 5.2 深蹲时小臂弯曲

- **症状**：深蹲时默认机器人双臂挺直，灵龙小臂弯曲。
- **根因**：灵龙手臂 7 自由度（肩3+肘1+腕3），原配置位置主导、旋转权重≈0，IK 只约束末端位置时解不唯一，收敛到弯肘解；且肩约束对象误用 `pitch_link`。
- **改法**：IK 配置手臂权重改为旋转主导（table1 0/10、table2 10/5），肩映射修正：
  ```
  left_shoulder_pitch_link  →  left_shoulder_yaw_link
  right_shoulder_pitch_link →  right_shoulder_yaw_link
  ```
- **原理**：旋转权重高时 IK 优先对齐每个关节朝向，姿态自然；位置权重高时 IK 为够到末端点牺牲姿态 → 弯肘。

---

# 第四部分 · 赛事提交

## 1. 提交物

初赛提交三样东西（即本流程的最终产物）：

| 提交物 | 来源（本流程输出） | 说明 |
|--------|-------------------|------|
| **动作序列文件** | `csv/live_motion.csv` | 37 列关节角序列 = 根位姿 7 + 30 关节角，逐帧 |
| **MuJoCo 演示视频** | `video/live_stream_robot.mp4` | 机器人跟随动作的渲染视频 |
| **使用说明** | 本文档 | 环境搭建、复现步骤、参数说明等 |

## 2. 作品提交规则

**分阶段提交要求**：

- **初赛**：提交生成的动作序列文件、使用说明及 MuJoCo 演示视频；
- **复赛**：提交可实时运行的系统封装（Docker 或指定格式）、演示视频；
- **总决赛**：现场部署演示，并提交源码和技术报告、演示视频。

**作品格式**：作品以 zip 压缩包形式发送至大赛组委会官方邮箱：`open@openloong.org.cn`，并抄送 `yelinqi@openloong.net`。

**大小限制**：若作品超过 50M，可选择"超大附件发送"，或将作品上传至云盘并把网盘链接及密码在邮件正文中呈现。压缩包大小原则上不超过 500MB。

**命名规范**（邮件主题与压缩包命名一致）：

```
大师赛第一期赛题1——队伍名称——队长姓名——队长手机号
```

示例：`大师赛第一期赛题1——王中王队——吴xx——184xxxxxxxx`

## 3. 评审规则

| 阶段 | 评审维度 |
|------|---------|
| **初赛** | 动作相似度（关节角度 RMSE、动态时间规整分数）及物理可行性得分 |
| **复赛** | 实时帧率、跟踪延迟、稳定性、流畅性、跨演示者泛化成功率 |
| **总决赛** | 任务模仿完成度（30%）、动作自然流畅度（25%）、鲁棒性（遮挡/快速动作下表现，25%）、创新性（20%） |

---

# 附录 · 运行参数与 Docker（原仓库说明）

## 运行参数

- `VIDEO`：默认 `examples/IMG_9732.mov`，指定输入视频路径，为 `0` 时打开真实摄像头输入。
- `TIME`：仅 `VIDEO=0` 时有效，默认 `0`。`TIME>0` 时录制 `TIME` 秒后自动停止；`TIME=0` 时终端输入 `q` / 按 `Esc` 停止。
- `ROBOT`：默认 `unitree_g1`，指定重定向机器人类型。
- `ROBOT_PATH`：默认空，可传机器人名或 xml 路径。
- `RECORD_VIDEO`：默认 `1`。
- `RECORD_WHAMVIDEO` / `RECORD_GMRVIDEO`：默认跟随 `RECORD_VIDEO`，是否录 WHAM 可视化 / MuJoCo 窗口视频。
- `USE_XVFB_GMR`：默认 `0`，`1` 时 GMR 在虚拟显示渲染（无物理弹窗）。
- `OUTPUT_ROOT`：默认空（空时输出到 `output/stream_demo`、`pkl_outputs/my_motion.pkl`、`pkl_outputs/csv/live_motion.csv`、`videos/live_stream_robot.mp4`）。
- `CAMERA_FOLLOW`：默认 `0`，`1` 时镜头固定跟随。
- `ROOT_ORIGIN_OFFSET`：默认 `0`；`1` 时回到"以起点为原点"的相对轨迹。
- `WHAM_USE_AMP`：默认 `0`，半精度推理（省显存提速）。
- `WHAM_DETECT_INTERVAL`：默认 `1`，每 N 帧做一次完整检测。
- `WHAM_INFER_INTERVAL`：默认 `1`，每 N 帧执行一次完整 WHAM 推理。
- `WHAM_STREAM_SEQ_LEN`：默认 `16`，WHAM 时序窗口长度。
- `WHAM_INPUT_SCALE`：默认 `1.0`，输入缩放比例（越小越快）。
- `GMR_TORCH_DEVICE`：默认 `cpu`，GMR 后处理/FK 的 torch 设备。
- `E2E_WARMUP`：默认 `1`，启动前热启动预热。

## Docker（可选）

仓库提供 `docker/Dockerfile`、`docker/compose.yml`、`docker/install_docker_ubuntu.sh`：

```bash
bash docker/install_docker_ubuntu.sh          # 安装 Docker + NVIDIA Container Toolkit（Ubuntu 22.04）
docker compose -f docker/compose.yml build --no-cache wham-gmr   # 构建镜像（耗时较长）
xhost +local:docker                           # 放开 X11（需要 MuJoCo 窗口时）
docker compose -f docker/compose.yml run --rm wham-gmr           # 启动容器
```

无界面服务器运行：

```bash
docker compose -f docker/compose.yml run --rm \
  -e USE_XVFB_GMR=1 -e VIDEO=0 \
  -e RECORD_GMRVIDEO=1 -e RECORD_WHAMVIDEO=1 \
  -e OUTPUT_ROOT=output/my_run wham-gmr bash run.sh
```

打包镜像：`docker save wham-gmr:local | gzip > wham-gmr_local.tar.gz`

> 容器内默认：`WHAM_PYTHON=/opt/conda/envs/wham/bin/python`、`GMR_PYTHON=/opt/conda/envs/gmr/bin/python`；项目目录通过 volume 挂载到 `/workspace`。
