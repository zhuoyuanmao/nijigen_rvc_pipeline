#!/bin/bash
set -e
SP=/mnt/c/Users/kevin/AppData/Local/Temp/claude/c--Users-kevin-ai-sing-by-ai/5135e3f3-7fe4-43b1-b058-d5c229f65a8e/scratchpad
PY=/mnt/c/Users/kevin/ai_sing_by_ai/ja_tts_explore/.venv/bin/python
TUN=/mnt/c/Users/kevin/Desktop/tokyo_summer_session_au/导出_tuned2
J=/mnt/c/Users/kevin/ai_sing_by_ai/ja_tts_explore/tokyo_summer_session
D=/mnt/c/Users/kevin/Desktop/tokyo_summer_session_au
F=/mnt/c/Users/kevin/ai_sing_by_ai/nijigen_rvc_pipeline/MIX_TABLE.md
V=v17

echo "########## 1. 对新导出重跑诊断 (参照: 六声优版) ##########"
$PY -u "$SP/pitch_diag_v16.py" > /tmp/diag_v17.log 2>&1
cp /tmp/diag_v17.log /tmp/diag3.log
grep -A4 "verdict" /tmp/diag_v17.log
grep -oE 'usable comparisons \([0-9]+ of [0-9]+' /tmp/diag_v17.log

echo "########## 2. 清空修音分轨, 曲线修音 (保护清单含 2:48.8) ##########"
rm -rf "$TUN"; mkdir -p "$TUN"
$PY -u "$SP/pitch_correct_v3.py" > /tmp/correct_v17.log 2>&1
cp /tmp/correct_v17.log /tmp/correct2.log
grep -viE 'warning|pkg_resources|setuptools|^  import|^reference|^  [a-z-]+ \.\.\.' /tmp/correct_v17.log

echo "########## 3. 台词句 2:09 ##########"
$PY -u "$SP/fix_dialogue_209.py" 2>/dev/null

echo "########## 4. 表格数据 ##########"
$PY -u "$SP/build_pitch_json.py"
$PY -u "$SP/update_pitch_json_v16.py"

echo "########## 5. 耳测定值 + 2:48.8 等律逐音修正 ##########"
$PY -u "$SP/apply_kirei_p60.py" 2>/dev/null
$PY -u "$SP/apply_uh_m60.py" 2>/dev/null
$PY -u "$SP/fine_tune_line41.py" 2>/dev/null

echo "########## 6. MIX_TABLE ##########"
sed -i "s/lovelive-cover_v[0-9]*\.wav/lovelive-cover_$V.wav/" "$SP/export_line_table.py"
$PY -u "$SP/export_line_table.py" > /tmp/t.log 2>&1; head -1 /tmp/t.log
echo "  歌词未映射行: $(awk -F'|' '/^\| [0-9]+ \|/ { gsub(/ /,"",$7); if ($7=="—") c++ } END {print c+0}' "$F")"
echo "  修音句数: $(grep -cE '\*\*[+-][0-9]+ ct\*\*' "$F")"

echo "########## 7. 混音 -> $V (人声目标 -9.0 dB, 按新原曲配平) ##########"
sed -e 's#/导出"#/导出_tuned2"#' -e "s#lovelive-cover_v[0-9]*\.wav#lovelive-cover_$V.wav#" \
    "$SP/mix_full_cast.py" > "$SP/mix_$V.py"
python3 - "$SP" "$V" <<'EOF'
import sys, pathlib
sp, v = pathlib.Path(sys.argv[1]), sys.argv[2]
s = (sp/f"mix_{v}.py").read_text(encoding="utf-8")
add = ('import soundfile as _sf\n'
       '_sf.write("%s/bus17_vocal.wav", vbus, SR, subtype="FLOAT")\n'
       '_sf.write("%s/bus17_bgm.wav", bgm_d, SR, subtype="FLOAT")\n' % (sp, sp))
s = s.replace("mix = bgm_d + vbus", add + "mix = bgm_d + vbus", 1)
(sp/f"mix_{v}.py").write_text(s, encoding="utf-8")
EOF
grep -n 'vocal target\|10\*\*(-9.0' "$SP/mix_$V.py" | head -2
$PY -u "$SP/mix_$V.py" 2>&1 | grep -E 'vocal target|LUFS|-> /mnt'
cp "$J/tokyo-summer-session_lovelive-cover_$V.wav" "$D/"
echo "===== V17 RENDERED ====="
