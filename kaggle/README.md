# Kaggle: обучение на бесплатной GPU

Пошагово, без лишнего.

## 1. Аккаунт

Зарегистрируйся на kaggle.com, подтверди телефон в Settings. Без этого GPU не дадут.

## 2. Загрузи код

Собери zip с проектом, без данных и чекпоинтов:

```bash
cd ~/Projects
zip -r aotr-agent.zip aotr-agent \
  -x "aotr-agent/.venv/*" \
  -x "aotr-agent/aotr-data/*" \
  -x "aotr-agent/checkpoints/*" \
  -x "aotr-agent/detector/frames/*" \
  -x "aotr-agent/__pycache__/*"
```

На Kaggle: Datasets → New Dataset → загрузи `aotr-agent.zip`, назови `aotr-agent`.

## 3. Загрузи данные

Датасеты лежат в `~/aotr-data/datasets`. Загрузи папки `ds_e1`, `ds_e2`, `ds_dagger` как один Dataset с именем `aotr-datasets`.

Если zip получается больше 20 ГБ, загружай по частям или убери `dg_*`, они для детектора.

## 4. Ноутбук

Создай Notebook, справа подключи оба Dataset, в настройках выбери GPU T4 x2. Вставь ячейки из `train_aotr.ipynb` и поправь пути.

Запуск обучения:

```bash
python -u train.py /kaggle/input/aotr-datasets/ds_e1 \
  /kaggle/input/aotr-datasets/ds_e2 \
  /kaggle/input/aotr-datasets/ds_dagger \
  --epochs 12 --batch 128 --device cuda \
  --out /kaggle/working/policy_v3.pt
```

## 5. Забери результат

Файл появится в Output справа. Скачай `policy_v3.pt`, положи в `~/Projects/aotr-agent/checkpoints`.

## Ограничения

- 30 часов GPU в неделю.
- Сессия до 12 часов, но может оборваться раньше. Поэтому сохраняй чекпоинт каждую эпоху и используй `--resume`.
- Диск 20 ГБ, наши данные влезают.

## Проверка

Первая эпоха должна дать loss примерно как на CPU, но в разы быстрее. Если пишет `cuda False`, проверь Accelerator в настройках.
