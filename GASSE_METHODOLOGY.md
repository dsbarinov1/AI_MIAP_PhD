# Методология Gasse et al. (2019): Bipartite GCNN для Branching

В данном документе представлено формальное описание методологии, изложенной в статье *Gasse, M., Chételat, D., Ferroni, N., Lodi, A., & Prouvost, J. P. (2019). Exact combinatorial optimization with graph convolutional neural networks.*

## 1. Представление MILP как двудольного графа

Задача смешанно-целочисленного линейного программирования (MILP) в канонической форме:
$$\min \{ \mathbf{c}^\top \mathbf{x} \mid \mathbf{A}\mathbf{x} \leq \mathbf{b}, \mathbf{l} \leq \mathbf{x} \leq \mathbf{u}, \mathbf{x} \in \mathbb{Z}^p \times \mathbb{R}^{n-p} \}$$

Представляется в виде неориентированного двудольного графа $\mathcal{G} = (\mathcal{C} \cup \mathcal{V}, \mathcal{E})$:
- **Узлы ограничений ($\mathcal{C}$):** $m$ узлов, по одному на каждое ограничение (строку матрицы $\mathbf{A}$).
- **Узлы переменных ($\mathcal{V}$):** $n$ узлов, по одному на каждую переменную (столбец матрицы $\mathbf{A}$).
- **Ребра ($\mathcal{E}$):** Ребро $(c_i, v_j) \in \mathcal{E}$ существует, если коэффициент $A_{ij} \neq 0$.

## 2. Исходные признаки (Initial Features)

Каждому узлу сопоставляется вектор признаков:
- **Для ограничений ($c_i \in \mathcal{C}$):** $\mathbf{o}_i \in \mathbb{R}^9$. Основные признаки: правая часть $b_i$, норма строки $\mathbf{A}_{i \cdot}$, тип ограничения.
- **Для переменных ($v_j \in \mathcal{V}$):** $\mathbf{o}_j \in \mathbb{R}^{13}$. Основные признаки: коэффициент целевой функции $c_j$, тип переменной, границы $[l_j, u_j]$, значение переменной в LP-релаксации, статус базиса.
- **Для ребер ($e_{ij} \in \mathcal{E}$):** Скалярное значение коэффициента матрицы $A_{ij} \in \mathbb{R}^1$.

## 3. Архитектура нейронной сети

Архитектура представляет собой двудольную графовую сверточную сеть (Bipartite GCNN) с $K$ слоями.

### 3.1. Начальный эмбеддинг
$$\mathbf{h}_i^{(0)} = \text{MLP}_{\text{init}, \mathcal{C}}(\mathbf{o}_i), \quad \mathbf{h}_j^{(0)} = \text{MLP}_{\text{init}, \mathcal{V}}(\mathbf{o}_j)$$
Размерность скрытого слоя $d$ (обычно 64).

### 3.2. Слои прохода сообщений (Message Passing)
Каждый слой $k \in \{0, \dots, K-1\}$ состоит из двух фаз обновления:

**Фаза 1: Ограничения $\leftarrow$ Переменные**
$$\mathbf{h}_i^{(k+1)} = \mathbf{f}_{\mathcal{C}} \left( \mathbf{h}_i^{(k)}, \sum_{j \in \mathcal{N}(i)} \mathbf{g}_{\mathcal{C}}(\mathbf{h}_i^{(k)}, \mathbf{h}_j^{(k)}, e_{ij}) \right)$$
где:
- $\mathbf{g}_{\mathcal{C}}: \mathbb{R}^d \times \mathbb{R}^d \times \mathbb{R}^1 \to \mathbb{R}^d$ — MLP для формирования сообщения.
- $\mathbf{f}_{\mathcal{C}}: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}^d$ — MLP для обновления состояния узла.

**Фаза 2: Переменные $\leftarrow$ Ограничения**
$$\mathbf{h}_j^{(k+1)} = \mathbf{f}_{\mathcal{V}} \left( \mathbf{h}_j^{(k)}, \sum_{i \in \mathcal{N}(j)} \mathbf{g}_{\mathcal{V}}(\mathbf{h}_j^{(k)}, \mathbf{h}_i^{(k+1)}, e_{ij}) \right)$$
где:
- $\mathbf{g}_{\mathcal{V}}: \mathbb{R}^d \times \mathbb{R}^d \times \mathbb{R}^1 \to \mathbb{R}^d$ — MLP для формирования сообщения.
- $\mathbf{f}_{\mathcal{V}}: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}^d$ — MLP для обновления состояния узла.

*Важно:* На второй фазе используются уже обновленные эмбеддинги ограничений $\mathbf{h}_i^{(k+1)}$.

### 3.3. Размеры тензоров (для одного слоя)
Если $m$ — число ограничений, $n$ — число переменных, $n_e$ — число ребер:
- Эмбеддинги ограничений: $[m, d]$
- Эмбеддинги переменных: $[n, d]$
- Сообщения (до агрегации): $[n_e, d]$
- Индексы ребер: $[2, n_e]$

## 4. Политика ветвления (Policy Head)

После $K$ слоев финальные эмбеддинги переменных $\mathbf{h}_j^{(K)}$ используются для вычисления вероятностей выбора переменной для ветвления среди кандидатов $\mathcal{A} \subseteq \mathcal{V}$:
$$\pi_\theta(v_j \mid s) = \frac{\exp(\text{MLP}_{\text{out}}(\mathbf{h}_j^{(K)}))}{\sum_{l \in \mathcal{A}} \exp(\text{MLP}_{\text{out}}(\mathbf{h}_l^{(K)}))}$$
$\text{MLP}_{\text{out}}$ переводит вектор размерности $d$ в скаляр (logit).

## 5. Обучение (Imitation Learning)

Используется кросс-энтропийная функция потерь для имитации эксперта (Strong Branching):
$$\mathcal{L}(\theta) = - \mathbb{E}_{(\mathcal{G}, \mathcal{A}, a^*) \sim \mathcal{D}} \left[ \log \pi_\theta(a^* \mid \mathcal{G}) \right]$$
где $a^*$ — переменная, выбранная экспертом Strong Branching в состоянии $\mathcal{G}$ с набором кандидатов $\mathcal{A}$.
