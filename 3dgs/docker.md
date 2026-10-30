# 3D Gaussian Splatting — Docker 使用教程

## 1. 环境概览

| 项目 | 版本 |
|---|---|
| 基础镜像 | `nvidia/cuda:11.8.0-devel-ubuntu22.04` |
| Python | 3.9 |
| PyTorch | 2.1.2 + cu118 |
| 目标 GPU | RTX 4090 (SM 8.9, Ada Lovelace) |
| Conda 环境名 | `gaussian_splatting` |
| 镜像 tag | `3dgs:latest` |

构建完成的镜像已包含：系统依赖 → Miniconda + ToS accept → conda 环境（Python 3.9 + PyTorch 2.1.2+cu118）→ 3DGS 源码。
**pip 依赖和 CUDA 扩展已在 Dockerfile 阶段 5、6 自动安装并验证**。

---

## 2. 容器生命周期

```bash
cd /home/lengyue/workspace/trae/docker-3gds

# 构建镜像
sudo docker compose build

# 启动（后台运行）
sudo docker compose up -d

# 进入容器 shell
sudo docker compose exec gaussian-splatting bash
# 或用容器名
sudo docker exec -it 3dgs bash

# 停止并删除容器
sudo docker compose down

# 重启（停掉再起来）
sudo docker compose down && sudo docker compose up -d

# 查看状态
sudo docker compose ps

# 查看日志
sudo docker compose logs --tail=50
```

### 进入容器后的固定操作

```bash
# 激活 conda 环境（每次 exec 进来都要做）
source /opt/conda/etc/profile.d/conda.sh
conda activate gaussian_splatting
```

---

## 3. 验证环境

### 3.1 PyTorch + GPU

```bash
python -c "
import torch
print(f'PyTorch:        {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version:   {torch.version.cuda}')
print(f'cuDNN version:  {torch.backends.cudnn.version()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'SM:  {torch.cuda.get_device_capability(0)}')
"
```

预期输出：

```
PyTorch:        2.1.2+cu118
CUDA available: True
CUDA version:   11.8
cuDNN version:  8700
GPU: NVIDIA GeForce RTX 4090
SM:  (8, 9)
```

### 3.2 CUDA 扩展

```bash
python -c "
from diff_gaussian_rasterization import _C;  print('diff-gaussian-rasterization: OK')
from simple_knn import _C;                    print('simple-knn: OK')
# 注意: fused_ssim 的扩展名不叫 _C，叫 fused_ssim_cuda
import fused_ssim_cuda
from fused_ssim import fused_ssim, fused_ssim3d
print('fused-ssim: OK')
print('All 3DGS CUDA extensions ready')
"
```

三个扩展都打印 OK 即环境就绪。

---

## 4. 手动编译 CUDA 扩展（仅 Dockerfile 阶段 5 被注释时需要）

### 4.1 为什么要 `--no-build-isolation`

`diff-gaussian-rasterization`、`simple-knn`、`fused-ssim` 的 `setup.py` 里有 `import torch`。
pip 的 PEP 517 隔离构建（默认）会在 `/tmp/` 创建临时虚拟环境，只装 `pyproject.toml` 里声明的依赖（setuptools / wheel），**完全看不到当前 conda 环境里已装好的 torch** → `ModuleNotFoundError: No module named 'torch'`。

`--no-build-isolation` 让 pip 直接在当前 conda 环境里执行 `setup.py`，复用已有的 torch。

### 4.2 编译命令

```bash
cd /workspace/gaussian-splatting
ls submodules/       # diff-gaussian-rasterization  simple-knn  fused-ssim

pip install --no-cache-dir --no-build-isolation \
    "numpy==1.26.4" \
    "opencv-python==4.10.0.84" \
    "scipy==1.11.4" \
    "scikit-learn==1.3.2" \
    "Pillow==10.2.0" \
    "matplotlib==3.8.4" \
    joblib lpips \
    submodules/diff-gaussian-rasterization \
    submodules/simple-knn \
    submodules/fused-ssim
```

编译约 3-5 分钟，全部 `Successfully built ...` 即成功。

---

## 5. 训练

```bash
cd /workspace/gaussian-splatting

# 基本训练
python train.py \
    -s data/mydataset \
    -m output/mydataset \
    --iterations 30000

# 带网络 viewer（浏览器打开 http://localhost:6009 实时渲染）
python train.py \
    -s data/mydataset \
    -m output/mydataset \
    --iterations 30000 \
    --port 6009
```

---

## 6. docker-compose.yml 配置说明

[docker-compose.yml](docker-compose.yml) 是容器的编排配置，把构建参数、GPU 挂载、端口、数据卷等一次性定义好。

| 配置项 | 作用 |
|---|---|
| `build.context` | 构建上下文指向 3DGS 源码目录，`COPY .` 能把子模块 `submodules/` 拷进镜像用于编译 |
| `build.dockerfile` | 指定使用本目录的 `Dockerfile` |
| `build.args.TORCH_CUDA_ARCH_LIST` | 编译 CUDA 扩展的目标架构，默认 `8.9`（RTX 4090） |
| `image: 3dgs:latest` | 构建产物的镜像名 |
| `container_name: 3dgs` | 容器名，方便 `sudo docker exec 3dgs ...` |
| `deploy.resources.reservations.devices` | GPU 挂载声明，等价于 `--gpus all` |
| `environment.NVIDIA_*` | 让容器内能看到 GPU 设备 |
| `ports: "6009:6009"` | 3DGS 训练网络 viewer 端口，浏览器 `http://localhost:6009` |
| `volumes` | 源码 + output + data 三个挂载，详见第 7 节 |
| `shm_size: "16g"` | 共享内存 16GB，PyTorch DataLoader 多进程需要 |
| `stdin_open` / `tty` | 保持容器运行，方便 `exec` 进入交互 |
| `restart: unless-stopped` | 除非手动停止，否则自动重启 |

---

## 7. Volume 挂载：bind mount vs named volume

### 7.1 当前配置（bind mount）

```yaml
volumes:
  # 源码目录：容器修改实时同步到宿主
  - /home/lengyue/workspace/3D-Reconstruction/gaussian-splatting:/workspace/gaussian-splatting
  # 训练输出：直接出现在宿主目录
  - /home/lengyue/workspace/trae/docker-3gds/output:/workspace/gaussian-splatting/output
  # 数据集：直接出现在宿主目录
  - /home/lengyue/workspace/trae/docker-3gds/data:/workspace/gaussian-splatting/data
```

### 7.2 两种挂载的区别

| 类型 | 写法 | 数据存在哪 | 容器修改→宿主 |
|---|---|---|---|
| **bind mount** | `宿主路径:容器路径` | 宿主的真实目录 | ✅ 实时同步 |
| **named volume** | `卷名:容器路径` | Docker 内部（`/var/lib/docker/volumes/<卷名>/_data/`） | ❌ 宿主不可见 |

之前用的是 named volume（`3dgs_output`、`3dgs_data`），容器里改 `output/` 或 `data/`，宿主看不到变化——因为数据存在 Docker 管理的隐藏目录里。现在改成 bind mount，容器里的修改直接反映到宿主目录。

### 7.3 切回 named volume

docker-compose.yml 里保留了注释行，取消注释即可切回：

```yaml
# named volume 版本:
# - 3dgs_output:/workspace/gaussian-splatting/output
# - 3dgs_data:/workspace/gaussian-splatting/data
```

同时取消文件末尾的 volume 声明：

```yaml
volumes:
  3dgs_output:
  3dgs_data:
```

---

## 8. named volume 数据拷贝

如果之前 named volume 里已有数据，需要迁移到新的 bind mount 目录：

### 方式 1：从运行中的容器拷（最简单）

```bash
sudo docker cp 3dgs:/workspace/gaussian-splatting/output /home/lengyue/workspace/trae/docker-3gds/output
sudo docker cp 3dgs:/workspace/gaussian-splatting/data /home/lengyue/workspace/trae/docker-3gds/data
```

### 方式 2：用临时容器挂载卷拷贝（容器没在跑也行）

```bash
# named volume → 宿主目录
sudo docker run --rm \
    -v 3dgs_output:/from \
    -v /home/lengyue/workspace/trae/docker-3gds/output:/to \
    alpine cp -a /from/. /to/

sudo docker run --rm \
    -v 3dgs_data:/from \
    -v /home/lengyue/workspace/trae/docker-3gds/data:/to \
    alpine cp -a /from/. /to/
```

### 方式 3：直接访问宿主真实路径（需 root）

```bash
sudo docker volume inspect 3dgs_output
# "Mountpoint": "/var/lib/docker/volumes/3dgs_output/_data"

sudo cp -a /var/lib/docker/volumes/3dgs_output/_data/. \
    /home/lengyue/workspace/trae/docker-3gds/output/
```

### 方式 4：打包成 tar 迁移

```bash
# 导出
sudo docker run --rm -v 3dgs_output:/data -v /home/lengyue/backup:/backup alpine \
    tar czf /backup/3dgs_output.tar.gz -C /data .

# 恢复
sudo docker run --rm -v 3dgs_output:/data -v /home/lengyue/backup:/backup alpine \
    tar xzf /backup/3dgs_output.tar.gz -C /data
```

---

## 9. 保存容器为正式镜像

手动编译完的修改只在当前容器里，容器删了就没了。固化成镜像：

```bash
# 宿主终端执行（容器里别退出）
sudo docker ps                        # 找到 container_id
sudo docker commit <container_id> 3dgs:latest
```

之后 `sudo docker compose up -d` 就是一键启动，进容器后只需 `conda activate` 即可。

---

## 10. 全量重建（验证 Dockerfile 所有修复）

```bash
cd /home/lengyue/workspace/trae/docker-3gds
sudo docker compose build --no-cache 2>&1 | tee /tmp/build-full.log
```

`--no-cache` 无视缓存全部重来，能验证所有修复（ToS accept、channel 配置、版本固定、CUDA 编译、fused_ssim 验证）都到位。预计 10-15 分钟。

---

## 11. 快速参考卡

| 操作 | 命令 |
|---|---|
| 构建镜像 | `sudo docker compose build` |
| 全量重建 | `sudo docker compose build --no-cache` |
| 启动容器 | `sudo docker compose up -d` |
| 进容器 shell | `sudo docker compose exec gaussian-splatting bash` |
| 激活 conda | `source /opt/conda/etc/profile.d/conda.sh && conda activate gaussian_splatting` |
| 查看 GPU | `nvidia-smi` |
| 训练 | `python train.py -s data/<数据集> -m output/<名称>` |
| 退出容器 | `exit`（容器保持运行） |
| 停容器 | `sudo docker compose down` |
| 重启 | `sudo docker compose down && sudo docker compose up -d` |
| 查看状态 | `sudo docker compose ps` |
| 查看日志 | `sudo docker compose logs --tail=50` |
| 保存为镜像 | `sudo docker commit <container_id> 3dgs:latest` |

---

## 12. 常见问题

### 构建时报 ToS 错误

新版 Miniconda（2025 年中起）强制要求对 `repo.anaconda.com` 显式同意服务条款。Dockerfile 阶段 2 已预置：

```dockerfile
&& conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
&& conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### 构建时报 `EnvironmentLocationNotFound`

原 Dockerfile 有 `CONDA_DEFAULT_ENV=gaussian_splatting`，但 conda create 之前环境不存在。已删除该 ENV 变量。

### PyTorch 是 CPU 版（`2.1.2.post104`，CUDA available: False）

`defaults` channel 里的 pytorch 是 CPU-only 构建。Dockerfile 已移除 `defaults` channel，pytorch 只能从 `pytorch` channel 取带 cu118 的版本。

### pip 安装 CUDA 扩展时报 `ModuleNotFoundError: No module named 'torch'`

PEP 517 隔离构建看不到当前 conda 环境的 torch。加 `--no-build-isolation` 即可解决（见第 4.1 节）。

### `from fused_ssim import _C` 报 ImportError

fused_ssim 的 CUDA 扩展名不叫 `_C`，叫 `fused_ssim_cuda`。正确验证：

```python
import fused_ssim_cuda
from fused_ssim import fused_ssim, fused_ssim3d
```

### NumPy 2.x 与 PyTorch 2.1.2 不兼容

PyTorch 2.1.2 编译于 NumPy 1.x。已固定 `numpy==1.26.4`，同时 `opencv-python==4.10.0.84`（5.x 要求 numpy>=2）。

### Dockerfile 里 `# ──` 注释导致后续命令不执行

**Shell 续行链中不能用 `#` 内联注释**——第一个 `#` 会注释掉后续所有 `&&` 命令。所有说明必须放在 `RUN` 指令之前的 Dockerfile 注释里，RUN 内只留纯命令链。

### NVIDIA Container Toolkit 没装，GPU 不可见

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
