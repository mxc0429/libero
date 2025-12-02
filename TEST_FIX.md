# 修复完成

## 已修改的文件

1. ✅ `libero/lifelong/algos/base.py` - 修改 persistent_workers
2. ✅ `libero/lifelong/algos/multitask.py` - 修改 persistent_workers

## 修改内容

将 `persistent_workers=True` 改为 `persistent_workers=(self.cfg.train.num_workers > 0)`

这样当 `num_workers=0` 时，`persistent_workers` 会自动设置为 `False`。

## 现在可以训练了

```bash
conda activate mxc_libero
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0

python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base \
    train.num_workers=0
```

或使用脚本：

```bash
./train_direct.sh 0 42 bc_rnn_policy base
```
