#!/usr/bin/env bash
# Пять исходных критериев преподавателя, с проверкой ошибок запуска
# и восстановлением конфигурации при прерывании.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

tmp=$(mktemp -d) || exit 1
cp params.yaml "$tmp/params.yaml" || exit 1
cleanup() { cp "$tmp/params.yaml" params.yaml; rm -rf "$tmp"; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
fails=0
ok() { printf '  ✓ %s\n' "$1"; }
fail() { printf '  ✗ %s\n' "$1"; fails=$((fails+1)); }

echo '1. Имя модели не прописано в исходниках'
if grep -rEq "[\"'](Qwen|HuggingFaceTB|meta-llama)/" src/ --include='*.py'; then
  fail 'в src/ есть фиксированное имя модели'
else
  ok 'имя берётся из params.yaml'
fi

echo '2. Смена модели через params.yaml'
if uv run --locked python - <<'PY' > "$tmp/switch.log" 2>&1
from pathlib import Path
import yaml
from src.config import load_params
from src.model import load_model
p = load_params()
p['model']['name'] = 'HuggingFaceTB/SmolLM2-135M-Instruct'
Path('params.yaml').write_text(yaml.safe_dump(p, allow_unicode=True), encoding='utf-8')
p = load_params()
_, model = load_model(p)
assert model.config.name_or_path == p['model']['name']
PY
then
  ok 'загружена именно модель из изменённого файла'
else
  fail 'не удалось переключить модель'
  tail -n 8 "$tmp/switch.log"
fi
cp "$tmp/params.yaml" params.yaml || exit 1

echo '3. Два успешных запуска дают одинаковый непустой вывод'
if uv run --locked python -c 'from src.config import load_params; assert load_params()["generate"]["temperature"] == 0' \
  && make -s generate > "$tmp/out1.txt" 2> "$tmp/gen1.log" \
  && make -s generate > "$tmp/out2.txt" 2> "$tmp/gen2.log" \
  && test -s "$tmp/out1.txt" && cmp -s "$tmp/out1.txt" "$tmp/out2.txt"; then
  ok 'temperature=0; оба запуска успешны; выводы совпали'
else
  fail 'ошибка запуска либо выводы различаются'
  tail -n 5 "$tmp"/gen*.log 2>/dev/null || true
fi

echo '4. Зависимости зафиксированы'
if uv lock --check > "$tmp/lock.log" 2>&1 && uv run --locked python - <<'PY'
from pathlib import Path
import tomllib
p = tomllib.loads(Path('pyproject.toml').read_text())
assert all('==' in dep for dep in p['project']['dependencies'])
assert Path('uv.lock').is_file()
PY
then
  ok 'прямые версии указаны; uv.lock соответствует проекту'
else
  fail 'проблема с фиксацией зависимостей'
fi

echo '5. Отдельные таймеры, прогрев, медиана и единицы памяти'
if make -s test > "$tmp/tests.log" 2>&1; then
  cat "$tmp/tests.log"
  ok 'проверки измерений пройдены'
else
  cat "$tmp/tests.log"
  fail 'проверки измерений не прошли'
fi

if [ "$fails" -eq 0 ]; then
  printf '\nВсе 5 пунктов пройдены. Отчёт: docs/hardware.md\n'
else
  printf '\nПровалено пунктов: %s\n' "$fails"
  exit 1
fi
