"""Generate logs/<exp>/filelist.txt (training manifest) — kotori version.

Same as otoya's (WHY: filelist is written by the WebUI's click_train, which
our scripts bypass — see METHODOLOGY §2), plus `--keep-list` so only slices
chosen by _score_slices.py enter training. F0/features stay extracted for
ALL slices (cheap, allows re-selection without re-extraction).

    python v2_step3b_filelist.py --exp kotori_v2 --keep-list <selected_slices.txt>
"""
from __future__ import annotations

import argparse
import os
from random import shuffle

RVC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Retrieval-based-Voice-Conversion-WebUI")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="kotori_v2")
    ap.add_argument("--sr", default="40k")
    ap.add_argument("--spk-id", type=int, default=0)
    ap.add_argument("--keep-list", default=None,
                    help="file of slice basenames to keep (one per line)")
    args = ap.parse_args()

    exp_dir = f"{RVC}/logs/{args.exp}"
    gt = f"{exp_dir}/0_gt_wavs"
    feat = f"{exp_dir}/3_feature768"
    f0d = f"{exp_dir}/2a_f0"
    f0nsf = f"{exp_dir}/2b-f0nsf"
    for d in (gt, feat, f0d, f0nsf):
        if not os.path.isdir(d):
            raise SystemExit(f"missing stage dir: {d}")

    names = (set(n.split(".")[0] for n in os.listdir(gt))
             & set(n.split(".")[0] for n in os.listdir(feat))
             & set(n.split(".")[0] for n in os.listdir(f0d))
             & set(n.split(".")[0] for n in os.listdir(f0nsf)))
    print(f"{len(names)} complete samples")

    if args.keep_list:
        keep = {l.strip() for l in open(args.keep_list, encoding="utf-8")
                if l.strip()}
        before = len(names)
        names &= keep
        print(f"keep-list: {len(keep)} listed -> {len(names)} kept "
              f"(dropped {before - len(names)})")
    if not names:
        raise SystemExit("no samples left")

    opt = [f"{gt}/{n}.wav|{feat}/{n}.npy|{f0d}/{n}.wav.npy|"
           f"{f0nsf}/{n}.wav.npy|{args.spk_id}" for n in names]
    mute_gt = f"{RVC}/logs/mute/0_gt_wavs/mute{args.sr}.wav"
    if not os.path.isfile(mute_gt):
        raise SystemExit(f"missing mute sample: {mute_gt}")
    for _ in range(2):
        opt.append(f"{mute_gt}|{RVC}/logs/mute/3_feature768/mute.npy|"
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
