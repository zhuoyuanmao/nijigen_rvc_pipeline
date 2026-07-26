#!/bin/bash
# Download the TITAN 40k community pretrain (G+D) for RVC v2 and sanity-check
# the checkpoints. TITAN = f0G40k/f0D40k further pretrained on 11.15h Expresso;
# drop-in replacement for the official pretrains via -pg/-pd.
#
# Primary source: https://huggingface.co/blaise-tk/TITAN
# Mirror:         https://huggingface.co/Politrees/RVC_resources
set -u
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="${1:-$PROJECT/Retrieval-based-Voice-Conversion-WebUI/assets/pretrained_v2_titan}"
mkdir -p "$DST"

try_dl() {  # url, out
  [ -s "$2" ] && { echo "[have] $(basename "$2")"; return 0; }
  echo "trying: $1"
  curl -fL --retry 3 --connect-timeout 20 -o "$2.part" "$1" && mv "$2.part" "$2" && return 0
  rm -f "$2.part"; return 1
}

G_OUT="$DST/G-f040k-TITAN.pth"
D_OUT="$DST/D-f040k-TITAN.pth"

G_URLS=(
  "https://huggingface.co/blaise-tk/TITAN/resolve/main/models/medium/40k/pretrained/G-f040k-TITAN-Medium.pth"
  "https://huggingface.co/Politrees/RVC_resources/resolve/main/pretrained/v2/40k/TITAN/G-f040k-TITAN-Medium.pth"
  "https://huggingface.co/blaise-tk/TITAN/resolve/main/G-f040k-TITAN-Medium.pth"
)
D_URLS=(
  "https://huggingface.co/blaise-tk/TITAN/resolve/main/models/medium/40k/pretrained/D-f040k-TITAN-Medium.pth"
  "https://huggingface.co/Politrees/RVC_resources/resolve/main/pretrained/v2/40k/TITAN/D-f040k-TITAN-Medium.pth"
  "https://huggingface.co/blaise-tk/TITAN/resolve/main/D-f040k-TITAN-Medium.pth"
)

ok_g=1; for u in "${G_URLS[@]}"; do try_dl "$u" "$G_OUT" && ok_g=0 && break; done
ok_d=1; for u in "${D_URLS[@]}"; do try_dl "$u" "$D_OUT" && ok_d=0 && break; done
[ $ok_g -ne 0 ] && { echo "FATAL: could not download TITAN G"; exit 1; }
[ $ok_d -ne 0 ] && { echo "FATAL: could not download TITAN D"; exit 1; }

ls -lh "$DST"

echo ""
echo "=== sanity: load state dicts, check 'model' key + plausible sizes ==="
$PROJECT/.venv/bin/python - "$G_OUT" "$D_OUT" <<'PY'
import sys, torch
for p in sys.argv[1:3]:
    ck = torch.load(p, map_location="cpu")
    keys = list(ck.keys())
    assert "model" in ck, f"{p}: no 'model' key, got {keys[:6]}"
    n = sum(v.numel() for v in ck["model"].values())
    print(f"OK  {p.split('/')[-1]}  top-keys={keys[:4]}  params={n/1e6:.1f}M")
PY
echo "TITAN READY"
