#!/bin/bash
# Build the relocatable, self-contained engine bundle that the app downloads on
# first launch — so users need NO Homebrew, system Python, or pip (no Terminal).
# Output: _engine-bundle/saglitz-engine-macos-arm64.tar.gz, containing a portable
# CPython + the full ML stack + the GPL CLI tools (draw-things-cli, ffmpeg,
# espeak-ng) with their dylib trees relocated to @executable_path.
#
# GPL note: the three CLI tools are GPL-3.0; we invoke them as separate
# subprocesses (no linking), so the engine/app is not a derivative. We ship their
# license texts + a source offer (stage 5) to satisfy GPL redistribution.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$ROOT/_engine-bundle"
ENG="$WORK/engine"
PYDIR="$ENG/python"

PY_VER="3.11.15"
PY_TAG="20260623"
PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PY_TAG}/cpython-${PY_VER}%2B${PY_TAG}-aarch64-apple-darwin-install_only.tar.gz"
PY_SHA256="d2324bfd1a7b9fc44ccd884c3a2505bcab6691dbfd4f8270e10c50aaa4e19506"

command -v dylibbundler >/dev/null || { echo "✗ need dylibbundler (brew install dylibbundler)"; exit 1; }
for b in draw-things-cli ffmpeg espeak-ng; do command -v "$b" >/dev/null || { echo "✗ need $b on PATH to vendor it"; exit 1; }; done

echo "▶ Stage 1: portable CPython ${PY_VER} (arm64) + integrity check…"
rm -rf "$WORK"; mkdir -p "$ENG"
curl -fsSL "$PY_URL" -o "$WORK/python.tar.gz"
GOT="$(shasum -a 256 "$WORK/python.tar.gz" | awk '{print $1}')"
[ "$GOT" = "$PY_SHA256" ] || { echo "✗ SHA256 mismatch (got $GOT) — refusing to bundle an unverified runtime"; exit 1; }
tar -xzf "$WORK/python.tar.gz" -C "$ENG"; rm -f "$WORK/python.tar.gz"
[ -x "$PYDIR/bin/python3.11" ] || { echo "✗ portable python missing"; exit 1; }
echo "  $("$PYDIR/bin/python3.11" --version)"

echo "▶ Stage 2: copy the venv's site-packages (must match ${PY_VER})…"
SRC_SP="$ROOT/engine-venv/lib/python3.11/site-packages"
[ -d "$SRC_SP" ] || { echo "✗ engine-venv site-packages not found"; exit 1; }
rsync -a --exclude '__pycache__' --exclude '*.pyc' "$SRC_SP/" "$PYDIR/lib/python3.11/site-packages/"

# server.py sits at the engine-dir root so its ROOT (= __file__/../..) is the
# stable App Support dir — user data (dt-models/projects) is a SIBLING of the
# bundle, never inside it, so a bundle update can't clobber it.
echo "▶ Stage 3: engine source…"
cp "$ROOT"/engine/*.py "$ENG/"
[ -d "$ROOT/engine/fonts" ] && cp -R "$ROOT/engine/fonts" "$ENG/fonts"   # OFL fonts for wordmarks

echo "▶ Stage 4: vendor the GPL CLI tools + relocate their dylibs…"
mkdir -p "$ENG/bin" "$ENG/libs" "$ENG/tools"
for b in draw-things-cli ffmpeg espeak-ng; do cp "$(readlink -f "$(command -v "$b")")" "$ENG/bin/$b"; done
lipo "$ENG/bin/draw-things-cli" -thin arm64 -output "$ENG/bin/draw-things-cli.arm64" 2>/dev/null \
  && mv "$ENG/bin/draw-things-cli.arm64" "$ENG/bin/draw-things-cli"   # drop x86_64 half (~85 MB)
# Single dylibbundler pass over all three (so the shared dest dir isn't wiped).
dylibbundler -of -b -x "$ENG/bin/ffmpeg" -x "$ENG/bin/draw-things-cli" -x "$ENG/bin/espeak-ng" \
  -d "$ENG/libs/" -p @executable_path/../libs/ >/dev/null 2>&1
ESPEAK_DATA="$(dirname "$(readlink -f "$(command -v espeak-ng)")")/../share/espeak-ng-data"
[ -d "$ESPEAK_DATA" ] && cp -R "$ESPEAK_DATA" "$ENG/tools/espeak-ng-data"

echo "▶ Stage 5: GPL/LGPL license texts + source offer…"
mkdir -p "$ENG/licenses"
# The GPL/LGPL components (CLI tools draw-things-cli/ffmpeg/espeak-ng + the
# in-process Piper/phonemizer TTS chain) all use these canonical texts; Homebrew
# Cellars and the Python wheels don't reliably ship them, so fetch them directly.
curl -fsSL "https://www.gnu.org/licenses/gpl-3.0.txt"   -o "$ENG/licenses/GPL-3.0.txt"   || echo "  ⚠ couldn't fetch GPL-3.0 text"
curl -fsSL "https://www.gnu.org/licenses/lgpl-3.0.txt"  -o "$ENG/licenses/LGPL-3.0.txt"  || true
curl -fsSL "https://www.gnu.org/licenses/lgpl-2.1.txt"  -o "$ENG/licenses/LGPL-2.1.txt"  || true
# Also keep any license files the tools happen to install (best effort).
for f in draw-things-cli ffmpeg espeak-ng; do
  pref="$(brew --prefix "$f" 2>/dev/null || true)"; [ -z "$pref" ] && continue
  find "$pref" -maxdepth 3 -iregex '.*\(LICENSE\|COPYING\).*' -type f \
    -exec sh -c 'mkdir -p "$1/$2"; cp "$3" "$1/$2/"' _ "$ENG/licenses" "$f" {} \; 2>/dev/null || true
done
cat > "$ENG/licenses/SOURCES.txt" <<'EOF'
This engine is GPL-3.0-or-later. It includes these GPL components — three CLI
tools invoked as separate subprocesses, plus the Piper/phonemizer TTS chain
imported in-process. Complete corresponding source:
  draw-things-cli  : https://github.com/drawthingsai/draw-things-community
  ffmpeg           : https://ffmpeg.org/download.html  (GPL build w/ libx264 etc.)
  espeak-ng        : https://github.com/espeak-ng/espeak-ng
  piper (piper-tts): https://github.com/OHF-Voice/piper1-gpl
  phonemizer       : https://github.com/bootphon/phonemizer
LGPL helpers: num2words (https://github.com/savoirfairelinux/num2words),
frozendict (https://github.com/Marco-Sulla/python-frozendict).
The engine source itself is public at the project's GitHub repositories.
EOF

echo "▶ Stage 6: launcher (sets PATH + espeak data; engine data lives in the parent)…"
cat > "$ENG/start-engine.sh" <<'EOS'
#!/bin/bash
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"    # …/engine
export PATH="$HERE/bin:$PATH"                            # bundled draw-things-cli, ffmpeg, espeak-ng
export ESPEAK_DATA_PATH="$HERE/tools/espeak-ng-data"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export SAGLITZ_QUANTIZE="${SAGLITZ_QUANTIZE:-8}"
PORT="${SAGLITZ_PORT:-8765}"; HOST="${SAGLITZ_HOST:-127.0.0.1}"
exec "$HERE/python/bin/python3.11" -m uvicorn server:app --app-dir "$HERE" --host "$HOST" --port "$PORT"
EOS
chmod +x "$ENG/start-engine.sh"

echo "▶ Stage 7: verify the bundle runs from this location (clean env)…"
"$PYDIR/bin/python3.11" - <<PY
import importlib
for m in ["torch","mlx.core","mflux","kokoro","fastapi"]: importlib.import_module(m)
print("  python ML stack OK")
PY
env -i "$ENG/bin/ffmpeg" -version >/dev/null && echo "  ffmpeg OK"
env -i "$ENG/bin/draw-things-cli" --help >/dev/null && echo "  draw-things-cli OK"
env -i ESPEAK_DATA_PATH="$ENG/tools/espeak-ng-data" "$ENG/bin/espeak-ng" -q "ok" >/dev/null && echo "  espeak-ng OK"

echo "▶ Stage 8: archive…"
TARBALL="$WORK/saglitz-engine-macos-arm64.tar.gz"
( cd "$WORK" && tar -czf "$TARBALL" engine )
echo "✓ $TARBALL  ($(du -sh "$TARBALL" | cut -f1))  — engine dir $(du -sh "$ENG" | cut -f1)"
