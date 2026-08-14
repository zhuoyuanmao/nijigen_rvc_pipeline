#!/bin/bash
set -e
SP=/mnt/c/Users/kevin/AppData/Local/Temp/claude/c--Users-kevin-ai-sing-by-ai/5135e3f3-7fe4-43b1-b058-d5c229f65a8e/scratchpad
PY=/mnt/c/Users/kevin/ai_sing_by_ai/ja_tts_explore/.venv/bin/python
EXP=/mnt/c/Users/kevin/Desktop/tokyo_summer_session_au/导出
TUN=/mnt/c/Users/kevin/Desktop/tokyo_summer_session_au/导出_tuned2
J=/mnt/c/Users/kevin/ai_sing_by_ai/ja_tts_explore/tokyo_summer_session
D=/mnt/c/Users/kevin/Desktop/tokyo_summer_session_au
F=/mnt/c/Users/kevin/ai_sing_by_ai/nijigen_rvc_pipeline/MIX_TABLE.md
V=v15

echo "########## 1. 清掉过期的修音分轨 (raw 已重对轨, 拼接点全部失效) ##########"
rm -rf "$TUN"; mkdir -p "$TUN"

echo "########## 2. 对新 raw 重跑音准诊断 ##########"
$PY -u "$SP/pitch_diag2.py" > /tmp/diag_v15.log 2>&1
cp /tmp/diag_v15.log /tmp/diag3.log          # build_pitch_json reads this path
grep -A4 "verdict" /tmp/diag_v15.log || tail -5 /tmp/diag_v15.log

echo "########## 3. 曲线修音 (唱段) ##########"
$PY -u "$SP/pitch_correct_v2.py" > /tmp/correct_v15.log 2>&1
cp /tmp/correct_v15.log /tmp/correct2.log
grep -viE 'warning|pkg_resources|setuptools|^  import|^reference|^  [a-z-]+ \.\.\.' /tmp/correct_v15.log

echo "########## 4. 台词句 2:09 (0:55 保持耳测否决, 不修) ##########"
sed 's/^TARGETS = .*/TARGETS = [(129.4, 133.4)]   # 0:55 stays ear-vetoed/' \
    "$SP/fix_two_lines.py" > "$SP/fix_dialogue_209.py"
$PY -u "$SP/fix_dialogue_209.py" 2>/dev/null

echo "########## 5. 表格数据 (先重建, 再盖上耳测决定) ##########"
$PY -u "$SP/build_pitch_json.py"
$PY -u "$SP/update_pitch_json.py"

echo "########## 6. 两处耳测定值 (从 RAW 单次 PSOLA 重建后拼回) ##########"
$PY -u "$SP/apply_kirei_p60.py" 2>/dev/null    # 2:30.8 海♂「綺麗だね」 +60
$PY -u "$SP/apply_uh_m60.py"    2>/dev/null    # 0:15.0 鸟♂ 合 Uh      -60

echo "########## 7. MIX_TABLE ##########"
sed -i "s/lovelive-cover_v[0-9]*\.wav/lovelive-cover_$V.wav/" "$SP/export_line_table.py"
$PY -u "$SP/export_line_table.py" > /tmp/t.log 2>&1; head -1 /tmp/t.log
echo "  歌词未映射行: $(awk -F'|' '/^\| [0-9]+ \|/ { gsub(/ /,"",$7); if ($7=="—") c++ } END {print c+0}' "$F")"
grep -cE '\*\*[+-][0-9]+ ct\*\*' "$F" | sed 's/^/  修音句数: /'

echo "########## 8. 混音 -> $V ##########"
sed -e 's#/导出"#/导出_tuned2"#' -e "s#lovelive-cover_v[0-9]*\.wav#lovelive-cover_$V.wav#" \
    "$SP/mix_full_cast.py" > "$SP/mix_$V.py"
grep -nE '^D = |^OUT = ' "$SP/mix_$V.py"
$PY -u "$SP/mix_$V.py" 2>&1 | grep -E 'humanize|duck|LUFS|-> /mnt'
cp "$J/tokyo-summer-session_lovelive-cover_$V.wav" "$D/"

echo "########## 9. 验收 (对标原曲的平衡 + 各段电平) ##########"
$PY - <<'EOF'
import numpy as np, librosa, soundfile as sf
SR=44100
J="/mnt/c/Users/kevin/ai_sing_by_ai/ja_tts_explore/tokyo_summer_session/"
P=J+"tokyo-summer-session_lovelive-cover_v15.wav"
INST=[(0.5,5.5),(91.0,101.0)]; VOC=[(40.0,120.0),(120.0,175.0)]
def wl(m,a,b):
    s=m[int(a*SR):int(b*SR)]; w=SR
    lv=np.array([20*np.log10(np.sqrt(np.mean(s[i:i+w]**2))+1e-9) for i in range(0,len(s)-w,SR//2)])
    return np.sort(lv)[len(lv)//2:]
def delta(p):
    y,_=librosa.load(p,sr=SR,mono=True)
    return np.concatenate([wl(y,a,b) for a,b in VOC]).mean()-np.concatenate([wl(y,a,b) for a,b in INST]).mean()
do=delta("/mnt/c/Users/kevin/Desktop/AI翻唱/tokyo_summer.wav"); dv=delta(P)
print(f"  人声相对伴奏: 原曲 {do:+.1f} dB | v15 {dv:+.1f} dB -> {'MATCH' if abs(dv-do)<=0.5 else 'off by '+format(dv-do,'+.1f')}")
y,_=sf.read(P); m=y.mean(1)
def w2(a,b):
    s=m[int(a*SR):int(b*SR)]; w=SR
    lv=np.array([20*np.log10(np.sqrt(np.mean(s[i:i+w]**2))+1e-9) for i in range(0,len(s)-w,SR//2)])
    return float(np.mean(np.sort(lv)[len(lv)//2:]))
s=(w2(40,120)+w2(120,175))/2
print(f"  独唱 {s:.1f} | 三男 {w2(188,200):.1f} ({w2(188,200)-s:+.1f}) | 六人 {w2(200,216):.1f} ({w2(200,216)-s:+.1f}) dB")
print(f"  峰值 {20*np.log10(np.max(np.abs(y))+1e-9):.2f} dBFS | 削波样本 {int(np.sum(np.abs(y)>=0.999))}")
EOF
echo "===== V15 DONE ====="
