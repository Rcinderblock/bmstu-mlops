"""Отчёт ДЗ2: только измеренные числа и объяснения условий опыта."""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODE_TITLES = {
    "inference": "Обычный запуск",
    "full_ft": "Полное дообучение",
    "lora": "LoRA r=8, q/v",
}


def thousands(n):
    return f"{n:,}".replace(",", " ")


def plot_activations(a, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), width_ratios=(2, 1))
    for label, values in a["norms"].items():
        axes[0].plot(values, label=f"{label}, блок {a['layers'][label]}")
    axes[0].set(
        xlabel="Позиция токена",
        ylabel="L2-норма, логарифмическая шкала",
        yscale="log",
        title="Значения на выходе трёх блоков",
    )
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    labels = list(a["norms"])
    axes[1].bar(
        labels,
        [sum(a["norms"][x]) / len(a["norms"][x]) for x in labels],
        color=["#507bac", "#d79336", "#44967f"],
    )
    axes[1].set(ylabel="Средняя L2-норма", title="Среднее по токенам")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def markdown_report(r, params):
    c, e, a = r["config"], r["environment"], r["activations"]
    groups = {g["group"]: g for g in r["params_by_group"]}
    modes = {m["mode"]: m for m in r["memory"]}
    base = modes["inference"]
    full = modes["full_ft"]
    lora = modes["lora"]
    weights = base["base_weights_bytes"] / 1024**2
    lines = [
        f"# ДЗ2. Анатомия {r['model']}",
        "",
        "Задание: исправить заготовку, посчитать параметры, снять значения внутри модели,",
        "проверить две конфигурации LoRA и сравнить память трёх режимов.",
        "Выполнено командой `make inspect`; исходные числа — в [report.json](report.json),",
        "ручная арифметика с формулами — в [anatomy-worksheet.xlsx](anatomy-worksheet.xlsx).",
        "",
        f"Время начала разбора: {r['measured_at_utc']}.",
        "",
        "## 1. Устройство модели",
        "",
        "| Поле | Значение |",
        "|---|---:|",
    ]
    for key in [
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "tie_word_embeddings",
    ]:
        lines.append(f"| `{key}` | {c[key]} |")
    lines += [
        "",
        "Один токен представлен вектором hidden_size. В каждом блоке есть механизм",
        "внимания и MLP — преобразование через промежуточное пространство intermediate_size.",
        f"У этой модели {c['num_attention_heads']} голов запроса и {c['num_key_value_heads']} голов ключей/значений.",
        "Несколько голов запроса используют общие ключи и значения — такое устройство называют GQA.",
        "",
        "## 2. Условия замера",
        "",
        "| Условие | Значение |",
        "|---|---|",
        f"| платформа | {e['platform']} |",
        f"| устройство | `{r['device']}` |",
        f"| модель | `{r['model']}` |",
        f"| dtype | `{r['dtype']}` |",
        f"| seq_len × batch_size | {e['seq_len']} × {e['batch_size']} |",
        f"| прогонов на режим | {e['repeats']}; каждый в новом процессе |",
        f"| память ускорителя | `{e['memory_metric']}` |",
        f"| память процесса | `{e['rss_metric']}` |",
        f"| интервал опроса | {e['sample_interval_seconds']} с, плюс границы стадий |",
        "| кэш ключей и значений | `use_cache=False` во всех режимах |",
        "| шаг обучения | forward + backward + AdamW.step; foreach=False |",
        f"| Python | {e['python']} |",
        f"| torch | {e['torch']} |",
        f"| transformers | {e['transformers']} |",
        f"| peft | {e['peft']} |",
        "",
        "Вход — случайные номера токенов с фиксированным seed=42. Собственный датасет здесь",
        "не используется; один шаг проверяет расход памяти, а не качество обучения.",
        "",
        "На MPS считается максимум наблюдений driver_allocated_memory. Это память драйвера,",
        "включая резерв аллокатора и выделения Metal/MPS, а не только живые тензоры.",
        "Очень короткий всплеск между опросами может быть пропущен. RSS сохранён отдельно:",
        "это память процесса и другая метрика. На CUDA используется встроенный счётчик пика.",
        "",
        "## 3. Параметры по частям модели",
        "",
        "| Группа | Тензоров | Размеры | Собственных параметров | Доля | Общих параметров |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for g in r["params_by_group"]:
        lines.append(
            f"| `{g['group']}` | {g['modules']} | {g['shape']} | {thousands(g['params'])} | {100 * g['share']:.3f}% | {thousands(g['tied_params'])} |"
        )
    lines += [
        f"| **Итого** | | | **{thousands(r['params_total'])}** | **100%** | |",
        "",
        f"Прямой подсчёт: **{thousands(r['params_direct'])}**. Разница: **{r['params_total'] - r['params_direct']}**.",
        "Имена и размеры всех отдельных тензоров, их доли и владельцы общих весов находятся",
        "в `report.json → parameter_rows`.",
        "",
        f"`lm_head` повторно использует {thousands(groups['lm_head']['tied_params'])} параметров входной таблицы embed_tokens.",
        "Поэтому его собственный вклад равен нулю. Назначений два, физический набор чисел один.",
        f"Нормировки включают векторы разных размеров: {groups['norm']['shape']}; это учтено отдельно в таблице Excel.",
        f"`k_proj` содержит {groups['k_proj']['params'] / groups['q_proj']['params']:.2f} от числа параметров `q_proj` из-за меньшего числа KV-голов.",
        f"MLP содержит {thousands(sum(groups[x]['params'] for x in ['gate_proj', 'up_proj', 'down_proj']))} параметров:",
        "три большие матрицы перехода между hidden_size и intermediate_size.",
        "",
        "## 4. Значения внутри модели",
        "",
        f"Запрос: {params['hooks']['prompt']} После шаблона диалога — {a['n_tokens']} токенов.",
        "Для каждой позиции измерена длина вектора выхода блока, то есть L2-норма.",
        "",
        "![Нормы активаций](activations.png)",
        "",
        "| Блок | Индекс с нуля | Средняя норма | Максимум | Позиция максимума |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, values in a["norms"].items():
        lines.append(
            f"| {label} | {a['layers'][label]} | {sum(values) / len(values):.2f} | {max(values):.2f} | {values.index(max(values))} |"
        )
    lines += [
        "",
        "Большая норма означает большой масштаб промежуточных чисел, а не высокое качество ответа.",
        "Один такой график не доказывает распределение внимания: для этого пришлось бы отдельно",
        "измерять сами веса внимания. Остаточные связи складывают выход блока с входом,",
        "но сами по себе не гарантируют монотонного роста нормы на каждом токене.",
        "Функции наблюдения снимаются в finally; повторный прогон не накапливает хуки.",
        "",
        "## 5. Дополнительные параметры LoRA",
        "",
        "Исходная матрица остаётся замороженной; обучаются A формы (r, in) и B формы (out, r).",
        "На выбранный линейный слой приходится `r × (in + out)` параметров.",
        "",
        "| Конфигурация | По формуле | Библиотека peft | Разница | Доля от базовой модели |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in r["lora"]:
        lines.append(
            f"| {item['name']} | {thousands(item['formula'])} | {thousands(item['peft'])} | {item['formula'] - item['peft']} | {100 * item['share_of_base']:.3f}% |"
        )
    lines += [
        "",
        "Для r=16 выбраны q/k/v/o/gate/up/down; выходной lm_head не включён.",
        "Процент здесь от базовой модели; peft может печатать процент от модели вместе с добавками.",
        "",
        "## 6. Память обычного запуска и обучения",
        "",
        "Единицы — МиБ (1024² байт); в исходном шаблоне они подписаны МБ.",
        "Память измеряется после загрузки весов и до освобождения результатов шага.",
        "Время включает загрузку модели. Это не сравнение скорости обучения.",
        "",
        "| Режим | Пик драйвера, МиБ | Пик RSS, МиБ | К обычному запуску | Время, с | PID | Опросов |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ["inference", "full_ft", "lora"]:
        m = modes[mode]
        lines.append(
            f"| {MODE_TITLES[mode]} | {m['peak_mb']:.1f} | {m['peak_rss_mb']:.1f} | {m['peak_mb'] / base['peak_mb']:.2f} | {m['seconds']:.1f} | {m['pid']} | {m['sample_count']} |"
        )
    lines += [
        "",
        f"Уникальные базовые веса занимают **{weights:.2f} МиБ** по numel × element_size.",
        f"В полном обучении градиенты занимают {full['gradient_bytes'] / 1024**2:.2f} МиБ, состояния AdamW — {full['optimizer_state_bytes'] / 1024**2:.2f} МиБ.",
        f"Для LoRA это {lora['gradient_bytes'] / 1024**2:.2f} и {lora['optimizer_state_bytes'] / 1024**2:.2f} МиБ соответственно.",
        "Эти размеры измерены по тензорам; итоговый пик также включает промежуточные значения,",
        "буферы вычислений и резерв драйвера. Поэтому он не обязан совпасть с суммой одних весов и градиентов.",
        f"В этом запуске полное обучение требует в {full['peak_mb'] / lora['peak_mb']:.2f} раза больше памяти, чем LoRA.",
        "LoRA экономит градиенты и состояния оптимизатора, но для обратного прохода всё равно нужны активации.",
        "",
        "## 7. Исправления заготовки и проверка",
        "",
        "| Дефект | Проявление | Исправление и наблюдаемый результат |",
        "|---|---|---|",
        f"| Общие веса считались дважды | Сумма была бы завышена на {thousands(groups['lm_head']['tied_params'])} | parameter_rows помечает повторный Parameter; сумма {thousands(r['params_total'])}, разница с прямым подсчётом 0 |",
        "| Хуки не снимались | Каждый вызов добавлял ещё три обработчика | Дескрипторы снимаются в finally; make check проверяет 0 после каждого из двух прогонов, отдельный тест проверяет исключение |",
        f"| Вместо пика снимался остаток в конце | Промежуточные выделения могли быть пропущены | Опрос каждые {e['sample_interval_seconds']} с и на границах стадий; число опросов и история максимумов сохранены в report.json |",
        f"| На ускорителе использовался RSS | Измерялась другая величина | Основная метрика {e['memory_metric']}; RSS отдельной колонкой |",
        "",
        "Дополнительно каждый режим перенесён в отдельный дочерний процесс. Конфигурация передаётся",
        "ему через стандартный ввод; разные PID приведены выше. Шаблон диалога не получает",
        "повторные служебные токены. Настройки градиентов после временной установки LoRA восстанавливаются.",
        "",
        "## 8. Как проверить",
        "",
        "```sh",
        "make install",
        "make inspect",
        "make check",
        "make test",
        "make check-hw1",
        "```",
        "",
        "`make check` выполняет семь проверок ДЗ2, включая новые измерения памяти.",
        "Логи успешного прогона сохраняются в docs/results/hw02-check.txt.",
        "Команды generate и bench, а также отчёт hardware.md относятся к ДЗ1 и сохранены.",
        "Тема будущего датасета пока не согласована; здесь она не требуется.",
        "",
        "Источники: [память MPS](https://docs.pytorch.org/docs/2.14/generated/torch.mps.driver_allocated_memory.html),",
        "[параметры LoRA](https://huggingface.co/docs/peft/package_reference/lora).",
        "",
    ]
    return "\n".join(lines)


def write_report(report, params):
    plot_activations(report["activations"], params["hooks"]["plot"])
    p = Path(params["report"]["markdown"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(markdown_report(report, params), encoding="utf-8")
