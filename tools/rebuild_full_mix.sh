#!/bin/bash
set -e
SP=/mnt/c/Users/kevin/AppData/Local/Temp/claude/c--Users-kevin-ai-sing-by-ai/5135e3f3-7fe4-43b1-b058-d5c229f65a8e/scratchpad
PY=/mnt/c/Users/kevin/ai_sing_by_ai/ja_tts_explore/.venv/bin/python
TUN=/mnt/c/Users/kevin/Desktop/tokyo_summer_session_au/导出_tuned2
J=/mnt/c/Users/kevin/ai_sing_by_ai/ja_tts_explore/tokyo_summer_session
D=/mnt/c/Users/kevin/Desktop/tokyo_summer_session_au
F=/mnt/c/Users/kevin/ai_sing_by_ai/nijigen_rvc_pipeline/MIX_TABLE.md
V=v16

echo "########## 1. 对新参照重跑诊断 ##########"
sed 's#tokyo_summer_vocals\.wav#tokyo_summer_vocals_6seiyuu.wav#' "$SP/pitch_diag2.py" > "$SP/pitch_diag_v16.py"
$PY -u "$SP/pitch_diag_v16.py" > /tmp/diag_v16.log 2>&1
cp /tmp/diag_v16.log /tmp/diag3.log
grep -A4 "verdict" /tmp/diag_v16.log
echo "-- usable/total --"
grep -oE 'usable comparisons \([0-9]+ of [0-9]+' /tmp/diag_v16.log

echo "########## 2. 清空修音分轨, 曲线修音 (保护清单已排除) ##########"
rm -rf "$TUN"; mkdir -p "$TUN"
$PY -u "$SP/pitch_correct_v3.py" > /tmp/correct_v16.log 2>&1
cp /tmp/correct_v16.log /tmp/correct2.log
grep -viE 'warning|pkg_resources|setuptools|^  import|^reference|^  [a-z-]+ \.\.\.' /tmp/correct_v16.log

echo "########## 3. 保护句: 2:09 台词句 (旧参照曲线, A/B 通过版) ##########"
$PY -u "$SP/fix_dialogue_209.py" 2>/dev/null

echo "########## 4. 表格数据 (先重建 json, 再放耳测句, 其写入不被覆盖) ##########"
$PY -u "$SP/build_pitch_json.py"
$PY -u "$SP/update_pitch_json_v16.py"
echo "-- 2:30.8 綺麗だね +60 / 0:15 合 Uh -60 (耳测) --"
$PY -u "$SP/apply_kirei_p60.py" 2>/dev/null
$PY -u "$SP/apply_uh_m60.py" 2>/dev/null

echo "########## 5. MIX_TABLE ##########"
sed -i "s/lovelive-cover_v[0-9]*\.wav/lovelive-cover_$V.wav/" "$SP/export_line_table.py"
grep -q '修音参照' "$SP/export_line_table.py" || \
  sed -i '/^> 成品:/a > 修音参照: **六声优版**原唱 (v16 起; 逐句选角与本 cast 对应, 参照可用性大幅提升); v15 及以前为二人版。' "$SP/export_line_table.py"
$PY -u "$SP/export_line_table.py" > /tmp/t.log 2>&1; head -1 /tmp/t.log
echo "  歌词未映射行: $(awk -F'|' '/^\| [0-9]+ \|/ { gsub(/ /,"",$7); if ($7=="—") c++ } END {print c+0}' "$F")"
echo "  修音句数: $(grep -cE '\*\*[+-][0-9]+ ct\*\*' "$F")"

echo "########## 6. 混音 -> $V ##########"
sed -e 's#/导出"#/导出_tuned2"#' -e "s#lovelive-cover_v[0-9]*\.wav#lovelive-cover_$V.wav#" \
    "$SP/mix_full_cast.py" > "$SP/mix_$V.py"
$PY -u "$SP/mix_$V.py" 2>&1 | grep -E 'LUFS|-> /mnt'
cp "$J/tokyo-summer-session_lovelive-cover_$V.wav" "$D/"

echo "########## 7. 验收 ##########"
$PY - <<'EOF'
import numpy as np, librosa, soundfile as sf
SR=44100
J="/mnt/c/Users/kevin/ai_sing_by_ai/ja_tts_explore/tokyo_summer_session/"
P=J+"tokyo-summer-session_lovelive-cover_v16.wav"
INST=[(0.5,5.5),(91.0,101.0)]; VOC=[(40.0,120.0),(120.0,175.0)]
def wl(m,a,b):
    s=m[int(a*SR):int(b*SR)]; w=SR
    lv=np.array([20*np.log10(np.sqrt(np.mean(s[i:i+w]**2))+1e-9) for i in range(0,len(s)-w,SR//2)])
    return np.sort(lv)[len(lv)//2:]
def delta(p):
    y,_=librosa.load(p,sr=SR,mono=True)
    return np.concatenate([wl(y,a,b) for a,b in VOC]).mean()-np.concatenate([wl(y,a,b) for a,b in INST]).mean()
do=delta("/mnt/c/Users/kevin/Desktop/AI翻唱/tokyo_summer.wav"); dv=delta(P)
print(f"  人声相对伴奏: 原曲 {do:+.1f} dB | v16 {dv:+.1f} dB -> {'MATCH' if abs(dv-do)<=0.5 else 'off by '+format(dv-do,'+.1f')}")
y,_=sf.read(P)
print(f"  峰值 {20*np.log10(np.max(np.abs(y))+1e-9):.2f} dBFS | 削波样本 {int(np.sum(np.abs(y)>=0.999))}")
EOF
echo "===== V16 DONE ====="
