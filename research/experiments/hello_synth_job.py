"""hello_synth 端到端验证任务代码（在 sandbox worker 内执行）。

流程：GPU 自检 -> 读取 DATA_DIR 数据 -> 训练阈值基线 -> 写出 submission.csv。
submission.csv 必须落在代码文件所在目录（dispatcher 契约）。
"""
import os

import pandas as pd

print("=== GPU self-check ===")
try:
    import torch

    print("torch_version:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu_name:", torch.cuda.get_device_name(0))
        x = torch.rand(512, 512, device="cuda")
        print("gpu_matmul_finite:", bool(torch.isfinite(x @ x).all()))
except Exception as exc:  # noqa: BLE001
    print("torch_check_failed:", exc)

print("=== Train threshold baseline ===")
data_dir = os.environ["DATA_DIR"]
print("DATA_DIR:", data_dir)
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))
print("train_rows:", len(train), "test_rows:", len(test))

scores = train["f1"] + train["f2"]
best_t, best_acc = 0.0, -1.0
for t in sorted(scores.unique()):
    pred = (scores > t).astype(int)
    acc = float((pred == train["label"]).mean())
    if acc > best_acc:
        best_acc, best_t = acc, float(t)
print(f"best_threshold: {best_t:.4f} train_acc: {best_acc:.4f}")

pred = (test["f1"] + test["f2"] > best_t).astype(int)
submission = pd.DataFrame({"id": test["id"], "label": pred})
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission.csv")
submission.to_csv(out_path, index=False)
print("submission_written:", out_path, len(submission))
