---
name: docker-env-ops
description: Operate this host's 3dgs Docker environment including sg docker access, compose recreation, conda training runs and GitHub repo sync. Use when asked to 查看/启动/停止容器, 运行 gaussian-splatting 训练, or 提交并推送 docker-workspace 仓库.
---

# Docker / 3dgs 环境操作手册

## 环境事实

- 占位符约定：`$HOME` / `$USER` 为当前用户环境变量；`<github-user>` 表示 GitHub 账号，实际值可用 `git remote get-url origin` 查询。
- 主机用户 `$USER`；`/var/run/docker.sock` 属 `root:docker`，但当前 shell 未加载 docker 组（`id -nG` 里没有 docker），直接执行 `docker ...` 会报 `permission denied`。
- `sudo` 需要密码且无法非交互使用。因此**所有 docker 命令都用 `sg docker -c "<命令>"` 包装**。
- 仓库根目录 `$HOME/workspace/docker`，远程 `https://github.com/<github-user>/docker-workspace`（public，分支 `main`）。
- 凭据存放在 `~/.git-credentials`（`credential.helper=store`，classic PAT，账号 `<github-user>`）。**只读取使用，绝不在输出或文件中打印 token**。
- 环境里没有 `gh` CLI，也没有 SSH key；GitHub 操作走 HTTPS + 上述凭据，建库用 API。
- Compose 文件 `3dgs/docker-compose.yml`：容器名 `3dgs`，镜像 `3dgs:latest`，GPU `all`，端口 `6009`，`shm_size 16g`。
- 容器内 Python 环境是 conda 的 `gaussian_splatting`（Python 3.9 + PyTorch 2.1.2 + CUDA 11.8）。**默认 `python` 没有 torch**。
- 挂载关系：
  - `$HOME/workspace/3D-Reconstruction/gaussian-splatting` → `/workspace/gaussian-splatting`
  - `$HOME/workspace/docker/3dgs/data` → `/workspace/gaussian-splatting/data`（宿主机软链到 `$HOME/data/3d-reconstruction`）
  - `$HOME/workspace/docker/3dgs/output` → `/workspace/gaussian-splatting/output`
- 数据集示例：`data/NRGBD/complete_kitchen`（31 张图像）。

## 容器操作

查看运行中的容器：

```bash
sg docker -c "docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'"
```

查看全部（含已退出）与 GPU 状态：

```bash
sg docker -c "docker ps -a --format 'table {{.ID}}\t{{.Names}}\t{{.Status}}'"
sg docker -c "docker exec 3dgs nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv"
```

启动 / 停止 / 删除：

```bash
sg docker -c "docker start 3dgs"
sg docker -c "docker stop 3dgs"
sg docker -c "docker rm -f 3dgs"
```

按 compose 重建（挂载配置变更后必须重建才生效）：

```bash
sg docker -c "cd $HOME/workspace/docker/3dgs && docker compose up -d"
```

核对实际挂载，确认数据可见：

```bash
sg docker -c "docker inspect 3dgs --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{println}}{{end}}'"
sg docker -c "docker exec 3dgs bash -lc 'ls data/NRGBD/complete_kitchen'"
```

## 训练（gaussian-splatting）

长任务用 `docker exec -d` 丢到后台，日志写到 output 目录：

```bash
sg docker -c "docker exec -d 3dgs bash -lc 'cd /workspace/gaussian-splatting && conda run --no-capture-output -n gaussian_splatting python train.py -s data/NRGBD/complete_kitchen -m output/NRGBD/complete_kitchen --iterations 300 > output/NRGBD/complete_kitchen/train_300.log 2>&1'"
```

看进度与产物：

```bash
sg docker -c "docker exec 3dgs bash -lc 'tail -n 20 output/NRGBD/complete_kitchen/train_300.log'"
sg docker -c "docker exec 3dgs bash -lc 'ls -lh output/NRGBD/complete_kitchen/point_cloud/iteration_300/'"
```

交互式进入容器：

```bash
sg docker -c "docker exec -it 3dgs bash"
```

## Git / GitHub

提交并推送（凭据由 store helper 自动提供，不要手写 token）：

```bash
cd $HOME/workspace/docker
git add -A
git commit -m "<说明变更动机的中文信息>"
git push
```

新建远程仓库（API，token 从 `~/.git-credentials` 读取，输出中不要包含 token）：

```bash
cred=$(cat ~/.git-credentials); pass=${cred##*:}; pass=${pass%@*}
curl -sS -X POST -H "Authorization: Bearer $pass" -H "Accept: application/vnd.github+json" \
  https://api.github.com/user/repos \
  -d '{"name":"<repo>","description":"<desc>","private":false,"auto_init":false}'
git remote add origin https://github.com/<github-user>/<repo>.git
git push -u origin main
```

## 坑与约束

- 旧容器可能挂在 `$HOME/workspace/trae/docker-3dgs/{data,output}`（空目录），会遮蔽真实数据集，表现为 `data/NRGBD/complete_kitchen` 不存在。先 `docker inspect` 核对挂载，不符就 `docker rm -f 3dgs` 后用正确 compose 重建。
- `conda run` 不加 `--no-capture-output` 会缓存输出，日志文件长时间为空。
- `sg docker -c "..."` 用外层双引号、内层单引号；命令里出现 `"` 时改用 `'` 或转义，避免被提前拆解。
- `3dgs/data` 是绝对路径软链接，已在 `.gitignore` 中排除（`/data`），不要提交；`output/` 同样不进版本控制。
- 训练会覆盖同名 `point_cloud/iteration_N` 目录，重跑前确认是否需要保留旧结果。
- 仓库是 public，`.trae/` 本地技能目录已加入根 `.gitignore`，不要把它推上去。
