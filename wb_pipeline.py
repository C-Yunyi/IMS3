# wb_pipeline.py  —— 单卡可直接跑
import os, re, argparse, subprocess, shlex
import wandb

def _tag(x: float) -> str:
    # 0.8 -> "0p8"
    return str(x).replace(".", "p")

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
    # ==== Sweep 传进来的关键参数 ====
    ap.add_argument("--groups", type=int, required=True)
    ap.add_argument("--ipc", type=int, required=True)
    ap.add_argument("--w_real", type=float, required=True)
    ap.add_argument("--w_sep",  type=float, required=True)
    ap.add_argument("--feature-backbone", type=str, default="resnet18")
    ap.add_argument("--real-train-dir", type=str, required=True)  # 真实训练集（估计质心）

    # ==== sample 阶段其他参数（按需保留默认） ====
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--num-sampling-steps", type=int, default=50)
    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument("--sel-eps", type=float, default=1e-6)
    ap.add_argument("--sel-max-iters", type=int, default=5)
    ap.add_argument("--nclass", type=int, default=10)
    ap.add_argument("--phase", type=int, default=0)
    ap.add_argument("--spec", type=str, default="none")
    ap.add_argument("--base-save-dir", type=str, default="../autodl-tmp/results/dit-distillation")
    ap.add_argument("--sample-script", type=str, default="centroid.py")  # 你现在用的是 centroid.py

    # ==== 训练脚本参数（对齐你贴的 argument 文件） ====
    ap.add_argument("--train-script", type=str, default="train2.py")
    ap.add_argument("--dataset", type=str, default="imagenet")
    ap.add_argument("--nclass-train", type=int, default=10)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--net-type", type=str, default="resnet_ap")
    ap.add_argument("--norm-type", type=str, default="instance")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--mixup", type=str, default="cut", choices=("vanilla","cut"))
    ap.add_argument("--randaug", type=str, default="true")  # 你的 str2bool 支持 'true'
    ap.add_argument("--randaug-n", type=int, default=1)
    ap.add_argument("--randaug-m", type=int, default=6)
    ap.add_argument("--spec-train", type=str, default="woof")  # 你训练侧默认用了 woof
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--imagewoof-root", type=str, required=True, # 原始数据根（包含 train/ 和 val/）
                help="original dataset root, e.g. /root/autodl-tmp/imagewoof2")

    # W&B
    ap.add_argument("--project", type=str, default="MinimaxDiffusion")

    args = ap.parse_args()  # <<< 先解析再使用

    # ==== W&B init ====
    wandb.init(project=args.project, config=vars(args))
    run_name = f"{args.w_real}x{args.w_sep}"
    wandb.run.name = run_name

    # ==== 路径命名（基于超参 + 唯一 id）====
    unique = wandb.run.id
    file_name = f"{run_name}_{unique}"
    save_dir_sample = os.path.join(args.base_save_dir, file_name)
    os.makedirs(save_dir_sample, exist_ok=True)

    # ==== 1) 采样+子集选择（你的 centroid.py）====
    sample_cmd = f"""
    python {args.sample_script}
      --groups {args.groups}
      --ipc {args.ipc}
      --real-train-dir {args.real_train_dir}
      --save-dir {save_dir_sample}
      --feature-backbone {args.feature_backbone}
      --image-size {args.image_size}
      --num-sampling-steps {args.num_sampling_steps}
      --cfg-scale {args.cfg_scale}
      --w-real {args.w_real}
      --w-sep {args.w_sep}
      --sel-eps {args.sel_eps}
      --sel-max-iters {args.sel_max_iters}
      --nclass {args.nclass}
      --phase {args.phase}
      --spec {args.spec}
    """.replace("\n", " ")
    run_cmd(sample_cmd)

    # 采样完成后的训练集目录
    # distilled_train = os.path.join(save_dir_sample, "final_distilled", "train")
    distilled_root = os.path.join(save_dir_sample, "final_distilled","train")
    if not os.path.isdir(distilled_root):
        raise RuntimeError(f"Distilled train dir not found: {distilled_root}")

    # imagewoof_root：原始数据根（必须是根目录，不是 /val 或 /train）
    def normalize_root(p: str) -> str:
        p = p.rstrip("/")
        if p.endswith("/train") or p.endswith("/val"):
            return os.path.dirname(p)
        return p

    imagewoof_root = normalize_root(args.imagewoof_root)

    # 用两个空格拼接
    imagenet_dirs_str = f"{distilled_root}  {imagewoof_root}"

    # 把 w_real/w_sep/G/K 写进 --tag，训练侧会用它生成保存目录
    tag_for_train = f"{run_name}"

    train_cmd = f"""
    python {args.train_script}
      -d {args.dataset}
      --imagenet_dir {imagenet_dirs_str}
      -n {args.net_type}
      --norm_type {args.norm_type}
      --depth {args.depth}
      --width {args.width}
      --size {args.size}
      --mixup {args.mixup}
      --spec {args.spec_train}
      --randaug {args.randaug}
      --randaug_n {args.randaug_n}
      --randaug_m {args.randaug_m}
      --ipc {args.ipc}    
      --tag {tag_for_train}
      {"--verbose" if args.verbose else ""}
    """.replace("\n", " ")
    # 纠正 argparse 名称中的连字符：--nclass-train 在命令里应写成 --nclass
    train_cmd = train_cmd.replace("--nclass-train", "--nclass")

    train_logs = run_cmd(train_cmd)

    # ==== 3) 记录指标 ====
    top1 = parse_top1_from_train(train_logs)
    wandb.log({"top1_acc": top1})
    print(f"[W&B] Logged top1_acc={top1:.4f}")

if __name__ == "__main__":
    main()
