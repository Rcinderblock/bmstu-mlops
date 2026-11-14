"""Читаемый обзор фактического полного журнала; это не новый запуск тестов."""
from pathlib import Path
import re
path=Path('results/hw11/checks.log');raw=path.read_text()
print('Сводка сохранённого прогона:',path)
assert 'Ran 5 tests' in raw and '\nOK\n' in raw,'Полный набор ещё не прошёл'
names=re.findall(r'^(test_\w+) \(([^\n]+?)\) \.\.\.',raw,re.M)
assert len(names)==5 and len(set(name for name,_ in names))==5
for name,_ in names:print(name,'— ok')
print(re.search(r'Ran 5 tests in [^\n]+',raw).group(0));print('OK')
