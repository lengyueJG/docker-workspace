# =============================================================================
# 3D Gaussian Splatting — Dockerfile
# 基础镜像: nvidia/cuda:11.8.0-devel-ubuntu22.04 (含 nvcc 编译器)
# Python: 3.9  |  PyTorch: 2.1.2+cu118  |  目标 GPU: RTX 4090 (SM 8.9)
#
# 构建上下文 (build context) 必须是 gaussian-splatting 源码目录，
# 这样 COPY . 才能把 submodules/ 一起拷进镜像用于编译。
# =============================================================================

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# ── 构建参数 ──────────────────────────────────────────────────────────────────
# 允许构建时覆盖 CUDA 架构列表（默认 Ada Lovelace 8.9）
ARG TORCH_CUDA_ARCH_LIST="8.9"

# ── 环境变量 ──────────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    CONDA_AUTO_ACTIVATE_BASE=false \
    # pip 镜像源（国内加速，海外用户注释掉即可）
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn \
    # 让 conda run 可以直接在非交互 shell 中使用
    PATH=/opt/conda/bin:${PATH}

# ── 阶段 1: 系统依赖 ───────────────────────────────────────────────────────────
# build-essential / cmake: 编译 CUDA 扩展必需
# git: 源码管理
# libgl1 / libglib2.0-0 / libsm6: OpenCV 等库的运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ── 阶段 2: 安装 Miniconda ─────────────────────────────────────────────────────
# 不依赖交互式 conda activate，后续统一用 conda run -n <env> 或直接调用
#
# 关键配置说明（都在下面一条 RUN 里执行，不能用 # 内联注释——shell 续行里 # 会吃掉后续命令）:
#   1. channel_priority=flexible —— 新版 Miniconda 默认 strict，禁止跨 channel 混装基础包
#      （python/libgcc 等）→ SolverError exit 1。flexible 允许跨 channel 组合依赖解。
#   2. solver=libmamba —— 新版 Miniconda (≥23.9) 默认带，比 classic solver 快 10x+ 且更能解复杂约束。
#   3. channel 顺序: pytorch → nvidia → conda-forge（不加 defaults）
#      defaults 里的 pytorch 2.1.2.post104 是 CPU-only 构建，flexible 模式下 conda 会优先选它，
#      导致 CUDA 不可用。移除 defaults 后 pytorch 只能从 pytorch channel 取带 cu118 的版本。
#   4. conda tos accept —— 2025 年中 Anaconda 强制要求 ToS 接受，新版 Miniconda 打包了
#      conda-anaconda-tos 插件，非交互 Docker 构建中必须显式 accept 否则 exit code 1。
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm Miniconda3-latest-Linux-x86_64.sh \
    && conda config --set auto_activate_base false \
    && conda config --set channel_priority flexible \
    && conda config --set solver libmamba \
    && conda config --add channels pytorch \
    && conda config --add channels nvidia \
    && conda config --add channels conda-forge \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# ── 阶段 3: 创建 conda 环境（Python 3.9 + PyTorch 2.1.2 + CUDA 11.8）─────────
# ⚠️ 不使用 environment.yml 里的 PyTorch 1.12.1 —— 4090 Ada Lovelace (SM 8.9) 不支持！
# PyTorch 2.1 是支持 CUDA 11.8 的最后一个稳定大版本，同时支持 SM 8.9
# ── 注意: 不要再加 -c pytorch -c nvidia -c conda-forge，
# ── channel 已在阶段 2 的 conda config 统一配好，命令行 -c 会 prepend 造成重复
RUN conda create -n gaussian_splatting -y \
    python=3.9 \
    pytorch=2.1.2 \
    torchvision=0.16.2 \
    torchaudio=2.1.2 \
    pytorch-cuda=11.8 \
    plyfile \
    tqdm \
    && conda clean -afy

# ── 阶段 4: 拷贝 3DGS 源码 ─────────────────────────────────────────────────────
# 构建上下文是 gaussian-splatting 源码目录，所以 COPY . 把整个仓库拷进来
WORKDIR /workspace/gaussian-splatting
COPY . /workspace/gaussian-splatting/

# ── 阶段 5: 安装 pip 依赖 + 编译 CUDA 扩展 ────────────────────────────────────
# 关键 1: 用 "conda run -n gaussian_splatting" 而非 conda activate
#         避免非交互 shell 下 activate 失效、跑错 Python 版本的问题
# 关键 2: --no-build-isolation —— diff-gaussian-rasterization / simple-knn / fused-ssim
#         的 setup.py 里有 "import torch"，PEP 517 隔离构建看不到当前环境的 torch，
#         必须跳过隔离，直接用 conda 环境里已装好的 torch 编译 CUDA 扩展
# 关键 3: 版本固定 —— 与 PyTorch 2.1.2 (2023 编译) 兼容的 NumPy 1.x 生态
#         numpy 1.26.4 / opencv-python 4.10.0.84 / scipy 1.11.4 / scikit-learn 1.3.2
#         Pillow 10.2.0 / matplotlib 3.8.4（都是 2023 末发布、与 NumPy 1.x + PyTorch 2.1 全兼容）
#         2024+ 的 opencv-python 5.x / numpy 2.x / scipy 1.12+ 会与 PyTorch 2.1.2 冲突
# 此时源码已在 WORKDIR 下，submodules/ 路径正确，可直接 pip install
RUN conda run -n gaussian_splatting pip install --no-cache-dir --no-build-isolation \
    "numpy==1.26.4" \
    "opencv-python==4.10.0.84" \
    "scipy==1.11.4" \
    "scikit-learn==1.3.2" \
    "Pillow==10.2.0" \
    "matplotlib==3.8.4" \
    joblib \
    lpips \
    submodules/diff-gaussian-rasterization \
    submodules/simple-knn \
    submodules/fused-ssim

# ── 阶段 6: 验证编译扩展能否 import ────────────────────────────────────────────
# 构建时就验证，避免跑起来才发现 CUDA 扩展没编好
# 注意: fused_ssim 的 CUDA 扩展名不叫 _C，叫 fused_ssim_cuda；API 是 fused_ssim/fused_ssim3d
RUN conda run -n gaussian_splatting python -c "\
    import torch; \
    print('=' * 50); \
    print(f'PyTorch:        {torch.__version__}'); \
    print(f'CUDA available: {torch.cuda.is_available()}'); \
    print(f'CUDA version:   {torch.version.cuda}'); \
    print(f'cuDNN version:  {torch.backends.cudnn.version()}'); \
    print(f'GPU count:      {torch.cuda.device_count()}'); \
    [print(f'  GPU {i}: {torch.cuda.get_device_name(i)} (SM {torch.cuda.get_device_capability(i)})') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None; \
    from diff_gaussian_rasterization import _C; \
    print('diff-gaussian-rasterization: OK'); \
    from simple_knn import _C; \
    print('simple-knn: OK'); \
    import fused_ssim_cuda; \
    from fused_ssim import fused_ssim, fused_ssim3d; \
    print('fused-ssim: OK'); \
    print('=' * 50); \
    print('All 3DGS CUDA extensions compiled successfully!'); \
"

# ── 默认入口 ──────────────────────────────────────────────────────────────────
# 直接启动 bash，容器内默认在 /workspace/gaussian-splatting 目录
# 激活环境: conda activate gaussian_splatting  (或直接用 conda run -n gaussian_splatting <cmd>)
CMD ["/bin/bash"]
