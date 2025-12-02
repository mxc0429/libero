# 快速修复 - h5py pickle 错误

## ✅ 问题已解决！

所有训练脚本已更新，添加了 `train.num_workers=0` 参数。

**注意**: 只需要设置 `train.num_workers=0`，不要同时设置 `eval.num_workers=0`。

---

## 🚀 现在可以直接使用

### 方法 1: 使用脚本（推荐）

```bash
./train_direct.sh 0 42 bc_rnn_policy base
```

### 方法 2: 使用命令行

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

### 方法 3: 8 GPU 并行

```bash
./train_all_gpus_fixed.sh
```

---

## 📝 快速测试（5 分钟）

```bash
conda activate mxc_libero
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0

python libero/lifelong/main.py \
    seed=42 \
    benchmark_name=LIBERO_10 \
    policy=bc_rnn_policy \
    lifelong=base \
    train.num_workers=0 \
    train.n_epochs=5 \
    eval.n_eval=5
```

---

## ✨ 所有脚本已更新

- ✅ `train_direct.sh`
- ✅ `train_simple.sh`
- ✅ `train_libero10.sh`
- ✅ `train_all_gpus_fixed.sh` (新版本)

现在可以直接使用，不会再出现 pickle 或 persistent_workers 错误！🎉
