import numpy as np
import pyscipopt
from itertools import product
import random

class MIAPGenerator:
    """
    Генератор задач Multi-Index Assignment Problem (MIAP).
    Поддерживает произвольную размерность N и количество индексов K (арность).
    """
    
    def __init__(self, n: int, k: int = 3, seed: int = 42):
        self.n = n
        self.k = k
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.py_rng = random.Random(seed)

    def generate_random_uniform(self):
        """
        Тривиальный генератор: Белый шум.
        C_ijk... ~ Uniform(0, 1)
        """
        shape = tuple([self.n] * self.k)
        cost_tensor = self.rng.random(shape)
        return cost_tensor

    def generate_euclidean(self, dim: int = 2):
        """
        Нетривиальный генератор: Евклидова структура.
        Генерируем K наборов точек в dim-мерном пространстве.
        C_ijk... = Сумма расстояний между точками или периметр.
        
        Для K=3 (i, j, k):
        Cost = Dist(Pi, Pj) + Dist(Pj, Pk) + Dist(Pk, Pi)
        Это моделирует логистику: Работник -> Задача -> Слот.
        """
        # Генерируем координаты для каждого "мира" (работники, задачи, слоты...)
        # sets_of_points shape: [K, N, dim]
        sets_of_points = self.rng.random((self.k, self.n, dim))
        
        # Создаем пустой тензор стоимости
        shape = tuple([self.n] * self.k)
        cost_tensor = np.zeros(shape)
        
        # Это "медленный" способ через циклы, но понятный. 
        # Для N=50 работает мгновенно. Для N>100 надо векторизовать.
        indices_iter = product(range(self.n), repeat=self.k)
        
        for indices in indices_iter:
            # indices - это кортеж (i, j, k, ...)
            # Берем соответствующие точки
            points = [sets_of_points[dim_idx, point_idx] for dim_idx, point_idx in enumerate(indices)]
            
            # Считаем "периметр" или сумму попарных расстояний
            val = 0
            for i in range(len(points)):
                p1 = points[i]
                p2 = points[(i + 1) % len(points)] # Замыкаем круг
                val += np.linalg.norm(p1 - p2)
            
            cost_tensor[indices] = val
            
        return cost_tensor

    def generate_known_optimum(self):
        """
        Generate an axial 3IAP instance with a known unique optimal solution (Grundel-Pardalos style).
        One feasible assignment gets cost 0; all others get positive random cost.
        For 3IAP a feasible solution is (i, pi(i), tau(i)) for i in 0..n-1 with permutations pi, tau.
        Returns cost_tensor and the optimal assignment list [(i, pi(i), tau(i)), ...].
        """
        if self.k != 3:
            raise NotImplementedError("known_optimum generator is for k=3 only")
        pi = self.rng.permutation(self.n)
        tau = self.rng.permutation(self.n)
        shape = (self.n,) * self.k
        cost_tensor = self.rng.random(shape)
        for i in range(self.n):
            cost_tensor[i, pi[i], tau[i]] = 0.0
        optimal_assignment = [(i, int(pi[i]), int(tau[i])) for i in range(self.n)]
        return cost_tensor, optimal_assignment

    def build_scip_model(
        self,
        cost_tensor,
        add_dirty_constraint: bool = False,
        dirty_fraction: float = 0.1,
    ):
        """
        Превращает тензор стоимости в PySCIPOpt модель.
        Добавляет переменные и жесткие ограничения:
        Sum_{j,k} x_ijk = 1 (для каждого i)
        Sum_{i,k} x_ijk = 1 (для каждого j)
        ...
        """
        model = pyscipopt.Model("MIAP")
        
        # 1. Создаем переменные
        # x_ijk... - бинарная переменная
        # Словарь vars: ключ (i,j,k) -> объект переменной SCIP
        vars_dict = {}
        it = np.nditer(cost_tensor, flags=['multi_index'])
        
        for _ in it:
            idx = it.multi_index # Кортеж (i, j, k)
            cost = cost_tensor[idx]
            # Создаем переменную. obj=cost задает целевую функцию (минимизация)
            var_name = f"x_{'_'.join(map(str, idx))}"
            vars_dict[idx] = model.addVar(name=var_name, vtype="B", obj=cost)

        model.setMinimize()

        # 2. Создаем ограничения (Constraints)
        # Для каждого измерения (dimension) d
        # Для каждого индекса i в этом измерении
        # Сумма по всем остальным индексам должна быть равна 1
        
        # Пример для K=3:
        # Dim 0 (i): Fix i, sum over j, k
        # Dim 1 (j): Fix j, sum over i, k
        # Dim 2 (k): Fix k, sum over i, j
        
        for d in range(self.k):
            for i in range(self.n):
                # Собираем все переменные, у которых индекс по измерению d равен i
                vars_in_constraint = []
                
                # Проходим по всем ключам (это не супер эффективно для N>50, но надежно)
                # В будущем оптимизируем через numpy indexing
                for idx, var in vars_dict.items():
                    if idx[d] == i:
                        vars_in_constraint.append(var)
                
                # Добавляем ограничение: sum(x...) == 1
                model.addCons(pyscipopt.quicksum(vars_in_constraint) == 1, name=f"cons_dim{d}_idx{i}")
        
        if add_dirty_constraint:
            all_vars = list(vars_dict.values())
            subset_size = max(1, int(len(all_vars) * dirty_fraction))
            subset_size = min(subset_size, len(all_vars))
            subset_vars = self.py_rng.sample(all_vars, subset_size)
            rhs = subset_size // 2
            model.addCons(pyscipopt.quicksum(subset_vars) <= rhs, name="dirty_constraint")

        return model, vars_dict

# --- Блок тестирования ---
if __name__ == "__main__":
    N = 5 # Размерность
    K = 3 # Арность (3 индекса)
    
    print(f"--- Generating MIAP (N={N}, K={K}) ---")
    
    gen = MIAPGenerator(n=N, k=K, seed=42)
    
    # 1. Тест Random
    print("\n1. Solving Random Uniform Instance...")
    c_random = gen.generate_random_uniform()
    model_rand, _ = gen.build_scip_model(c_random)
    model_rand.optimize()
    print(f"Status: {model_rand.getStatus()}")
    print(f"Optimal Value (Random): {model_rand.getObjVal():.4f}")
    
    # 2. Тест Euclidean
    print("\n2. Solving Euclidean Structured Instance...")
    c_euclid = gen.generate_euclidean()
    model_euc, _ = gen.build_scip_model(c_euclid)
    model_euc.optimize()
    print(f"Status: {model_euc.getStatus()}")
    print(f"Optimal Value (Euclidean): {model_euc.getObjVal():.4f}")
    
    # Проверка на сложность
    # Убедимся, что решение целочисленное
    sol = model_euc.getBestSol()
    # print(sol) # Можно раскомментировать, чтобы увидеть значения переменных