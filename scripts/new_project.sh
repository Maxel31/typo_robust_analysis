#!/usr/bin/env bash
# 新規プロジェクトを _sample_project から作成する。
# 使い方: scripts/new_project.sh <project_name>
set -euo pipefail

NAME="${1:?usage: scripts/new_project.sh <project_name>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/_sample_project"
DST="$ROOT/projects/$NAME"

if [ -e "$DST" ]; then
  echo "error: already exists: $DST" >&2
  exit 1
fi

SLUG="${NAME//-/_}"          # パッケージ名はアンダースコア
DIST="${NAME//_/-}"          # 配布名はハイフン

cp -r "$SRC" "$DST"
mv "$DST/src/sample_project" "$DST/src/$SLUG"

# 名前を置換（pyproject / py / md）
grep -rl -e "sample_project" -e "sample-project" "$DST" 2>/dev/null | while read -r f; do
  sed -i "s/sample-project/$DIST/g; s/sample_project/$SLUG/g" "$f"
done

echo "created: $DST"
echo "next:"
echo "  uv sync"
echo "  uv run python projects/$NAME/experiments/reproduction/run.py --config projects/$NAME/configs/repro_baseline.yaml"
