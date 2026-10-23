"""Строгий подсчёт по всем заданиям и допуск без разрешённой регрессии."""
import json
import math


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Повторное поле JSON')
        result[key] = value
    return result


def _parse(raw):
    def invalid_constant(value):
        raise ValueError('Недопустимая константа JSON: ' + value)
    parsed = json.loads(raw, object_pairs_hook=_object, parse_constant=invalid_constant)
    if not isinstance(parsed, dict) or set(parsed) != {'answer'} or not isinstance(parsed['answer'], str):
        raise ValueError('Ожидается ровно одно строковое поле answer')
    return parsed


def _ids(items, label):
    if not isinstance(items, list) or not items:
        raise ValueError(label + ': пустой набор')
    result = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(label + ': строка должна быть словарём')
        key = item.get('id')
        if not isinstance(key, str) or not key.strip() or key in result:
            raise ValueError(label + ': пустой или повторный id')
        result[key] = item
    return result


def _counts(items):
    n = len(items)
    correct = sum(item['correct'] for item in items)
    valid = sum(item['valid'] for item in items)
    errors = sum(item['error'] is not None for item in items)
    return dict(n=n, correct=correct, valid=valid, errors=errors,
                accuracy=correct / n, format_rate=valid / n)


def _summarize(items):
    names = sorted({name for item in items for name in item['slices']})
    return dict(overall=_counts(items),
                slices={name: _counts([item for item in items if name in item['slices']]) for name in names},
                per_item=items)


def _slice_names(value):
    if (not isinstance(value, list) or not value or
            any(not isinstance(name, str) or not name.strip() for name in value) or len(value) != len(set(value))):
        raise ValueError('Нужны непустые уникальные имена срезов')
    return value


def score(rows, outputs):
    """JSON с одним answer; неверный формат, пустота и исключения остаются в n."""
    tasks, responses = _ids(rows, 'Задания'), _ids(outputs, 'Ответы')
    if set(tasks) != set(responses):
        raise ValueError('ID заданий и ответов не совпадают')
    items = []
    for key, row in tasks.items():
        choices = row.get('choices')
        if (not isinstance(choices, list) or not choices or
                any(not isinstance(choice, str) or not choice for choice in choices) or
                len(choices) != len(set(choices)) or row.get('expected') not in choices):
            raise ValueError('Неверные choices или expected')
        names = _slice_names(row.get('slices'))
        response = responses[key]
        raw, error = response.get('raw', ''), response.get('error')
        if not isinstance(raw, str):
            raw, error = '', error if error is not None else 'raw должен быть строкой'
        parsed, valid = None, False
        if error is None:
            try:
                value = _parse(raw)
                if value['answer'] in choices:
                    parsed, valid = value, True
            except (ValueError, TypeError, RecursionError):
                pass
        items.append(dict(id=key, raw=raw, parsed=parsed, valid=valid,
                          correct=valid and parsed['answer'] == row['expected'],
                          error=error, slices=list(names)))
    return _summarize(items)


def _check_counts(actual, expected, label):
    if not isinstance(actual, dict):
        raise ValueError(label + ': нет метрик')
    for key in ('n', 'correct', 'valid', 'errors'):
        value = actual.get(key)
        if type(value) is not int or value != expected[key]:
            raise ValueError(label + ': counts не соответствуют per_item: ' + key)
    for key in ('accuracy', 'format_rate'):
        value = actual.get(key)
        if (type(value) not in (int, float) or not math.isfinite(value) or
                not math.isclose(value, expected[key], rel_tol=0, abs_tol=1e-12)):
            raise ValueError(label + ': метрика не соответствует counts: ' + key)


def _validate_score(report, minimum, label):
    if not isinstance(report, dict):
        raise ValueError(label + ': нет отчёта')
    items = _ids(report.get('per_item'), label)
    if len(items) < minimum:
        raise ValueError(f'{label}: требуется минимум {minimum} примеров')
    for item in items.values():
        if any(key not in item for key in ('raw', 'parsed', 'valid', 'correct', 'error', 'slices')):
            raise ValueError(label + ': неполный per_item')
        _slice_names(item['slices'])
        if type(item['valid']) is not bool or type(item['correct']) is not bool:
            raise ValueError(label + ': valid/correct должны быть bool')
        if item['correct'] and not item['valid']:
            raise ValueError(label + ': неверный формат не может быть правильным')
        if item['error'] is not None and (item['correct'] or item['valid']):
            raise ValueError(label + ': ошибка генерации должна давать ноль')
        if item['valid']:
            try:
                parsed = _parse(item['raw'])
            except (ValueError, TypeError, RecursionError) as exc:
                raise ValueError(label + ': invalid raw помечен как valid') from exc
            if item['parsed'] != parsed:
                raise ValueError(label + ': parsed не соответствует raw')
    expected = _summarize(list(items.values()))
    _check_counts(report.get('overall'), expected['overall'], label + '.overall')
    slices = report.get('slices')
    if not isinstance(slices, dict) or set(slices) != set(expected['slices']):
        raise ValueError(label + ': имена срезов не соответствуют per_item')
    for name, counts in expected['slices'].items():
        _check_counts(slices[name], counts, label + '.' + name)
    if expected['overall']['errors']:
        raise ValueError(label + ': есть ошибки генерации')
    return items


def _align(a, b, label):
    if set(a) != set(b) or any(set(a[key]['slices']) != set(b[key]['slices']) for key in a):
        raise ValueError(label + ': различаются id или принадлежность срезам')


def quality_gate(baseline, candidate, critical_slices):
    """Совместимые полные отчёты; ни одного разрешённого ухудшения."""
    critical = _slice_names(critical_slices)
    for role, report in [('baseline', baseline), ('candidate', candidate)]:
        if not isinstance(report, dict) or report.get('status') != 'complete':
            raise ValueError(role + ': оценка не завершена')
        model = report.get('model')
        if not isinstance(model, dict) or not isinstance(model.get('kind'), str) or not model['kind']:
            raise ValueError(role + ': не указана модель')
        if model['kind'] not in ('base', 'adapter', 'fixture'):
            raise ValueError(role + ': неизвестный вид модели')
        if model['kind'] == 'base' and (not isinstance(model.get('name'), str) or not model['name'] or
                not isinstance(model.get('revision'), str) or len(model['revision']) != 40):
            raise ValueError(role + ': база не закреплена точной ревизией')
        if model['kind'] == 'adapter' and any(not isinstance(model.get(k), str) or not model[k] for k in ('name', 'version', 'run_id', 'uri')):
            raise ValueError(role + ': неполное описание версии адаптера')
        for key in ('protocol_sha256', 'golden_sha256', 'baseline_spec_sha256'):
            if not isinstance(report.get(key), str) or not report[key] or report[key] != baseline.get(key):
                raise ValueError(role + ': несовместимый ' + key)
    a = _validate_score(baseline, 50, 'baseline')
    b = _validate_score(candidate, 50, 'candidate')
    _align(a, b, 'golden')
    sa, sb = baseline['slices'], candidate['slices']
    if len(sa) < 3 or set(sa) != set(sb) or not set(critical) <= set(sa):
        raise ValueError('Нужны одинаковые ≥3 среза и все критические группы')
    for name in sa:
        if sa[name]['n'] != sb[name]['n']:
            raise ValueError('Разные размеры среза: ' + name)
    for key in ('correct', 'valid'):
        if candidate['overall'][key] < baseline['overall'][key]:
            raise ValueError('Регрессия общей ' + ('accuracy' if key == 'correct' else 'format_rate'))
    for name in critical:
        if sb[name]['correct'] < sa[name]['correct']:
            raise ValueError('Регрессия критического среза: ' + name)
    ba = _validate_score(baseline.get('behavioral'), 3, 'baseline.behavioral')
    bb = _validate_score(candidate.get('behavioral'), 3, 'candidate.behavioral')
    _align(ba, bb, 'behavioral')
    behavior = candidate['behavioral']['overall']
    if behavior['correct'] != behavior['n']:
        raise ValueError('Не все обязательные behavioral-кейсы пройдены')
    return dict(passed=True, n=len(a), critical_slices=critical,
                baseline_accuracy=baseline['overall']['accuracy'],
                candidate_accuracy=candidate['overall']['accuracy'],
                candidate_format_rate=candidate['overall']['format_rate'],
                behavioral_passed=behavior['correct'],
                protocol_sha256=candidate['protocol_sha256'])
