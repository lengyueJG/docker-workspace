# 3DGS Docker 快速使用

镜像 `3dgs:latest` 已内置 Miniconda、conda 环境 `gaussian_splatting`（Python 3.9 + PyTorch 2.1.2+cu118）以及编译好的 3DGS CUDA 扩展。下面三步即可跑起来。

---

## 1. 启动 Docker

```bash
cd /home/lengyue/workspace/trae/docker-3dgs

# 首次或修改 Dockerfile 后构建镜像（已构建过可跳过）
sudo docker compose build

# 后台启动容器
sudo docker compose up -d

# 进入容器 shell
sudo docker compose exec gaussian-splatting bash
# 等价写法：sudo docker exec -it 3dgs bash
```

说明：

- `up -d` 之后容器常驻后台，`exit` 退出 shell 不会停止容器。
- 容器名固定为 `3dgs`，服务名为 `gaussian-splatting`，两者都能用来 exec。
- 已把 GPU（RTX 4090）、端口 `6009`、源码目录、`output/`、`data/` 都挂载好了，无需额外参数。

---

## 2. 激活容器内的 conda 虚拟环境

每次新开一个 `exec` shell 都要激活一次：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate gaussian_splatting
```

激活成功的标志是命令行提示符前出现 `(gaussian_splatting)`。验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望输出: 2.1.2+cu118 True
```

> 不想手动 activate 时，可以用 `conda run -n gaussian_splatting <命令>`，效果等价。

---

## 3. 运行 3DGS 指令

以下命令都在容器的 `/workspace/gaussian-splatting` 下执行（进入容器后默认就在该目录）。

### 3.1 训练

```bash
# 无 viewer
python train.py \
    -s data/NRGBD/complete_kitchen \
    -m output/NRGBD/complete_kitchen \
    --iterations 30000

# 带网络 viewer，宿主浏览器打开 http://localhost:6009
python train.py \
    -s data/NRGBD/complete_kitchen \
    -m output/NRGBD/complete_kitchen \
    --iterations 30000 \
    --port 6009
```

- `-s`：数据集目录（容器内对应宿主 `/home/lengyue/data/3d-reconstruction`）。
- `-m`：模型/输出目录（对应宿主 `output/NRGBD/complete_kitchen`），训练完的 `point_cloud/iteration_30000/point_cloud.ply` 在这里。

### 3.2 渲染与指标

```bash
python render.py -m output/NRGBD/complete_kitchen
python metrics.py -m output/NRGBD/complete_kitchen
```

### 3.3 不进入容器，一条命令直接跑

```bash
sudo docker compose exec gaussian-splatting bash -lc \
  'source /opt/conda/etc/profile.d/conda.sh && conda activate gaussian_splatting && cd /workspace/gaussian-splatting && python train.py -s data/NRGBD/complete_kitchen -m output/NRGBD/complete_kitchen --iterations 30000'
```

---

## 4. 常用运维命令

```bash
sudo docker compose ps                 # 查看状态
sudo docker compose logs --tail=50     # 查看日志
sudo docker compose restart            # 重启容器
sudo docker compose down               # 停止并删除容器（数据在宿主目录，不受影响）
```

| 操作 | 命令 |
|---|---|
| 启动 | `sudo docker compose up -d` |
| 进入 | `sudo docker compose exec gaussian-splatting bash` |
| 激活环境 | `source /opt/conda/etc/profile.d/conda.sh && conda activate gaussian_splatting` |
| 训练 | `python train.py -s <数据集> -m <输出目录> --iterations 30000` |
| 渲染 | `python render.py -m <输出目录>` |
| 退出 | `exit`（容器继续运行） |
