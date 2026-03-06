# AI_MIAP_PhD
Методологический прототип по теме: Multi-Index Assignment Problem (MIAP) + Learning to Branch (Gasse-style imitation learning на GNN).
## Что уже реализовано
- Генератор MIAP-инстансов: `generators.py` (random, euclidean, known_optimum в стиле Grundel–Pardalos).
- Загрузчик бенчмарков: `benchmarks.py` (Crama–Spieksma EJOR 1992).
- Сбор разметки branching-состояний через Ecole + Strong Branching: `data_collector.py`.
- Преобразование sample → PyG graph и obs→Data: `dataset.py`.
- Обучение GNN-политики ветвления (с prenorm и sum-агрегацией): `train.py`, `model.py`.
- Оценка: default SCIP, imitation-метрики, **solver-level с обученной GNN-политикой**: `evaluate_baselines.py`.
## Ключевая идея текущей методологии
1. Генерируются MIAP-инстансы (`k=3` как стартовый режим).
2. На каждом branching-шаге в Ecole берутся:
   - двудольный граф (constraints/variables/edges),
   - множество кандидатов на ветвление,
   - оценки Strong Branching.
3. Метка — лучший кандидат из текущего action set.
4. Модель обучается предсказывать выбор эксперта (imitation learning).
## Рекомендуемый воспроизводимый протокол v1
Использовать разные seed для split-ов:
- train seed: `101`
- val seed: `202`
- test seed: `303`
### 1) Сбор train
```powershell
python data_collector.py --num_instances 1000 --save_dir dataset_train --split_name train --seed 101 --n_size 10 --k_dim 3 --max_steps_per_instance 10
```
### 2) Сбор val
```powershell
python data_collector.py --num_instances 200 --save_dir dataset_val --split_name val --seed 202 --n_size 10 --k_dim 3 --max_steps_per_instance 10
```
### 3) Сбор test
```powershell
python data_collector.py --num_instances 200 --save_dir dataset_test --split_name test --seed 303 --n_size 10 --k_dim 3 --max_steps_per_instance 10
```
### 4) Обучение
```powershell
python train.py --train_dir dataset_train --val_dir dataset_val --epochs 100 --batch_size 32 --hidden 128 --lr 0.0005 --seed 101 --save_path best_model.pt
```
### 5) Baseline-оценка solver-level и (опционально) imitation-метрик
```powershell
python evaluate_baselines.py --num_instances 50 --n_size 10 --k_dim 3 --seed 303 --time_limit 60 --output_json baseline_eval_results.json
```
С checkpoint-метриками:
```powershell
python evaluate_baselines.py --num_instances 50 --n_size 10 --k_dim 3 --seed 303 --time_limit 60 --checkpoint_path best_model.pt --dataset_dir dataset_test --device cpu --output_json baseline_eval_results.json
```
## Важные флаги
- `--add_dirty_constraint` и `--dirty_fraction`: стресс-режим генератора. Для базовой научной постановки рекомендуется оставлять выключенным.
- `--max_steps_per_instance`: ограничивает глубину сбора траектории ветвления.
- `--difficulty easy|medium|hard`: подставляет n_size=10/15/20 и max_steps=15/25/40.
- `--target_samples N`: собирать до N сэмплов (Gasse-style, с повторной выборкой инстансов).
## Бенчмарки и генераторы
- **Crama–Spieksma**: скачать инстансы с [instancesEJOR](https://fspieksma.win.tue.nl/instancesEJOR.htm), положить в папку, затем `benchmarks.load_crama_spieksma(path)` или `load_crama_spieksma_directory(dir)` → `(n, cost_tensor)`. Далее `MIAPGenerator(n=n, k=3).build_scip_model(cost_tensor)`.
- **Известный оптимум**: `gen.generate_known_optimum()` → `cost_tensor, assignment`; подходит для проверки решателя.
## Минимальные метрики для отчета
- imitation-level: Acc@1, Acc@5.
- solver-level: solve time, node count, solved ratio under time limit.
