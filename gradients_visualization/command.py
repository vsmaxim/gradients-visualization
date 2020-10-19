from dataclasses import dataclass
import re
from typing import Dict, Tuple

from gradients_visualization.methods import build_adagrad, build_adam, build_momentum_gd, build_rmsprop, \
    build_vanilla_gd
from gradients_visualization.parser.expression import eval_numpy_expr, eval_pytorch_expr, parser, \
    variable_names
from gradients_visualization.visualizer import visualize


@dataclass
class OptimizationCommand:
    function_expr: str
    method_name: str
    vars: Tuple[str, ...]
    intervals: Dict[str, Tuple[float, float]]

    # TODO: Errors handling
    @classmethod
    def from_string(cls, src: str) -> "OptimizationCommand":
        def remove_spaces(s: str) -> str:
            return re.sub(r"\s*", "", s)

        func, method, intervals = map(remove_spaces, re.split("with|for", src))
        func_expr = func[len("optimize")+1:-1]
        vars = tuple(variable_names(parser.parse(func_expr)))
        interval_parts = intervals.split(",")
        interval_bits = (re.split(r"=|\.\.", part) for part in interval_parts)
        intervals = {var: (float(left), float(right)) for var, left, right in interval_bits}
        return OptimizationCommand(func_expr, method, vars, intervals)

    def run(self):
        def numpy_expr(x: float, y: float):
            var_to_val = {var: val for var, val in zip(self.vars, (x, y))}
            return eval_numpy_expr(self.function_expr, **var_to_val)

        def pytorch_expr(x: float, y: float):
            var_to_val = {var: val for var, val in zip(self.vars, (x, y))}
            return eval_pytorch_expr(self.function_expr, **var_to_val)

        # TODO: Learning rate change
        name_to_method = {
            "vanilla": build_vanilla_gd(numpy_expr, pytorch_expr),
            "momentum": build_momentum_gd(numpy_expr, pytorch_expr),
            "adagrad": build_adagrad(numpy_expr, pytorch_expr),
            "rmsprop": build_rmsprop(numpy_expr, pytorch_expr),
            "adam": build_adam(numpy_expr, pytorch_expr)
        }

        method = name_to_method[self.method_name]
        visualize(method, numpy_expr, *self.intervals.values())
