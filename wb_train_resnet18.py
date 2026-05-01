# wb_pipeline_repeat20.py —— 同一训练配置重复运行 N 次并记录准确率
import os, re, argparse, subprocess, shlex
import wandb

def run_cmd(cmd, cwd=None, env=None):
    print(f"[RUN] {cmd}")
    proc = subprocess.Popen(
        shlex.split(cmd), cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, universal_newlines=True
    )
    lines = []
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed ({ret}): {cmd}")
    return "".join(lines)

def parse_top1_from_train(log_text: str) -> float:
    m = re.search(r"WANDB_TOP1_BEST=([0-9.]+)", log_text)
    if m: return float(m.group(1))
    m = re.search(r"Best,\s+last acc:\s+([0-9.]+)\s+[0-9.]+", log_text)
    if m: return float(m.group(1))
    m = re.findall(r"Top1\s+([0-9.]+)\s+Top5", log_text)
    if m: return float(m[-1])
    raise RuntimeError("Failed to parse Top-1 accuracy from train logs")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipc", type=int, required=True)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--train-script", type=str, default="train2.py")
    ap.add_argument("--dataset", type=str, default="imagenet")
    ap.add_argument("--nclass", type=int, default=10)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--net-type", type=str, default="resnet")
    ap.add_argument("--norm-type", type=str, default="instance")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--mixup", type=str, default="cut")
    ap.add_argument("--randaug", type=str, default="true")
    ap.add_argument("--randaug-n", type=int, default=1)
    ap.add_argument("--randaug-m", type=int, default=6)
    ap.add_argument("--spec-train", type=str, default="woof")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--project", type=str, default="MinimaxDiffusion")

    # 数据路径
    ap.add_argument("--distilled-dir", type=str, required=True)
    ap.add_argument("--imagewoof-root", type=str, required=True)
    ap.add_argument("--tag", type=str, default="repeat_train")

    args = ap.parse_args()

    # ==== 初始化 W&B ====
    wandb.init(project=args.project, config=vars(args))
    wandb.run.name = f"{args.tag}_repeat{args.repeat}"

    def normalize_root(p: str) -> str:
        p = p.rstrip("/")
        if p.endswith("/train") or p.endswith("/val"):
            return os.path.dirname(p)
        return p

    distilled_root = normalize_root(args.distilled_dir)
    imagewoof_root = normalize_root(args.imagewoof_root)
    imagenet_dirs_str = f"{os.path.join(distilled_root, 'final_distilled/train/')}  {imagewoof_root}"

    top1_list = []

    # ==== 重复训练 ====
    for run_idx in range(args.repeat):
        print(f"\n===== [RUN {run_idx+1}/{args.repeat}] =====")

        # 可设定随机种子（或每次随机）
        seed = run_idx * 10 + 42

        train_cmd = f"""
        python {args.train_script}
          -d {args.dataset}
          --imagenet_dir {imagenet_dirs_str}
          -n {args.net_type}
          --norm_type {args.norm_type}
          --depth {args.depth}
          --width {args.width}
          --size {args.image_size}
          --mixup {args.mixup}
          --spec {args.spec_train}
          --randaug {args.randaug}
          --randaug_n {args.randaug_n}
          --randaug_m {args.randaug_m}
          --ipc {args.ipc}
          --tag {args.tag}_r{run_idx}
          --seed {seed}
          {"--verbose" if args.verbose else ""}
        """.replace("\n", " ")

        train_logs = run_cmd(train_cmd)
        top1 = parse_top1_from_train(train_logs)
        top1_list.append(top1)
        wandb.log({f"top1_acc_run{run_idx}": top1})
        print(f"[Run {run_idx}] Top1 acc = {top1:.4f}")

    # ==== 汇总结果 ====
    import numpy as np
    mean_acc = np.mean(top1_list)
    std_acc = np.std(top1_list)
    print(f"\n[Summary] mean={mean_acc:.3f}, std={std_acc:.3f}")
    wandb.log({"top1_mean": mean_acc, "top1_std": std_acc})
    print("[W&B] Logged summary metrics")

if __name__ == "__main__":
    main()
