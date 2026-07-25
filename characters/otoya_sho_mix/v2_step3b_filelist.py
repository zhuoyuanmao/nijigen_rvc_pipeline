"""Generate logs/<exp>/filelist.txt — the training manifest train.py reads.

The step*.sh scripts drive RVC's modules directly and therefore skip the
WebUI, but filelist.txt is written by infer-web.py's click_train (line ~545),
not by train.py. Without it training dies on a missing training_files path.

This replicates that logic exactly: intersect the four per-stage directories
by basename, emit `gt|feature|f0|f0nsf|spk_id`, append the two `mute` padding
entries RVC expects, shuffle.

    python v2_step3b_filelist.py --exp otoya_sho_mix_v2
"""
from __future__ import annotations

import argparse
import os
from random import shuffle

# project root = two levels up from this script (characters/<name>/ -> root)
_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RVC = os.environ.get("RVC_DIR",
                     os.path.join(_PROJECT, "Retrieval-based-Voice-Conversion-WebUI"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="otoya_sho_mix_v2")
    ap.add_argument("--sr", default="40k")
    ap.add_argument("--spk-id", type=int, default=0)
    ap.add_argument("--version", default="v2")
    args = ap.parse_args()

    exp_dir = f"{RVC}/logs/{args.exp}"
    gt = f"{exp_dir}/0_gt_wavs"
    feat = f"{exp_dir}/3_feature768" if args.version == "v2" else f"{exp_dir}/3_feature256"
    f0d = f"{exp_dir}/2a_f0"
    f0nsf = f"{exp_dir}/2b-f0nsf"

    for d in (gt, feat, f0d, f0nsf):
        if not os.path.isdir(d):
            raise SystemExit(f"missing stage dir: {d}")

    names = (set(n.split(".")[0] for n in os.listdir(gt))
             & set(n.split(".")[0] for n in os.listdir(feat))
             & set(n.split(".")[0] for n in os.listdir(f0d))
             & set(n.split(".")[0] for n in os.listdir(f0nsf)))
    print(f"{len(names)} complete samples "
          f"(gt {len(os.listdir(gt))}, feat {len(os.listdir(feat))}, "
          f"f0 {len(os.listdir(f0d))}, f0nsf {len(os.listdir(f0nsf))})")
    if not names:
        raise SystemExit("no samples survived the intersection")

    opt = [f"{gt}/{n}.wav|{feat}/{n}.npy|{f0d}/{n}.wav.npy|"
           f"{f0nsf}/{n}.wav.npy|{args.spk_id}" for n in names]

    fea_dim = 768 if args.version == "v2" else 256
    mute_gt = f"{RVC}/logs/mute/0_gt_wavs/mute{args.sr}.wav"
    if not os.path.isfile(mute_gt):
        raise SystemExit(f"missing mute sample: {mute_gt}")
    for _ in range(2):
        opt.append(
            f"{mute_gt}|{RVC}/logs/mute/3_feature{fea_dim}/mute.npy|"
            f"{RVC}/logs/mute/2a_f0/mute.wav.npy|"
            f"{RVC}/logs/mute/2b-f0nsf/mute.wav.npy|{args.spk_id}")

    shuffle(opt)
    out = f"{exp_dir}/filelist.txt"
    with open(out, "w") as f:
        f.write("\n".join(opt))
    print(f"-> {out}  ({len(opt)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
