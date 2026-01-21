import numpy as np
import pyscipopt
from itertools import product

class MIAPGenerator:
    def __init__(self, n: int, k: int = 3, seed: int = 42):
        self.n = n
        self.k = k
        self.rng = np.random.default_rng(seed)

    def generate_random_uniform(self):
        return self.rng.random(tuple([self.n] * self.k))

    def generate_euclidean(self, dim: int = 2):
        sets_of_points = self.rng.random((self.k, self.n, dim))
        cost_tensor = np.zeros(tuple([self.n] * self.k))
        indices_iter = product(range(self.n), repeat=self.k)
        for indices in indices_iter:
            val = 0
            points = [sets_of_points[d, idx] for d, idx in enumerate(indices)]
            for p in range(len(points)):
                val += np.linalg.norm(points[p] - points[(p + 1) % len(points)])
            cost_tensor[indices] = val
        return cost_tensor

    def build_scip_model(self, cost_tensor):
        model = pyscipopt.Model("MIAP")
        vars_dict = {}
        it = np.nditer(cost_tensor, flags=['multi_index'])
        for _ in it:
            idx = it.multi_index
            vname = f"x{'_'.join(map(str, idx))}"
            vars_dict[idx] = model.addVar(name=vname, vtype="B", obj=float(cost_tensor[idx]))
        
        model.setMinimize()

        # Стандартные ограничения MIAP (Axial)
        for d in range(self.k):
            for i in range(self.n):
                vars_in_con = [v for idx, v in vars_dict.items() if idx[d] == i]
                model.addCons(pyscipopt.quicksum(vars_in_con) == 1, name=f"c_{d}_{i}")
        
        return model, vars_dict