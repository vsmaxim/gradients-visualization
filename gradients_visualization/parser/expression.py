from pathlib import Path
from typing import Callable, Dict, Iterable, Set

import numpy as np
from parsimonious import Grammar
from parsimonious.nodes import Node
import torch

_grammar_path = Path(__file__).resolve().parent / "optimizer.peg"
parser = Grammar(_grammar_path.read_text())


def atoms(tree: Node) -> Iterable[Node]:
    if tree.expr_name == "atom":
        yield tree

    for child in tree.children:
        yield from atoms(child)


def eval_expr(
        expr: str,
        symbol_to_expr_op: Dict[str, Callable] = None,
        symbol_to_factor_op: Dict[str, Callable] = None,
        symbol_to_term_op: Dict[str, Callable] = None,
        exp_op: Callable = None,
        func_to_op: Dict[str, Callable] = None,
        **kwargs: float):
    def eval_tree(tree: Node, parent_name: str) -> float:
        if tree.expr_name == "atom" and len(tree.children) == 1:
            if tree.text in variable_to_value:
                return variable_to_value[tree.text]
            else:
                try:
                    return int(tree.text)
                except ValueError:
                    try:
                        return float(tree.text)
                    except ValueError:
                        raise ValueError(f"Unknown literal in root node: {tree.text}")

        if tree.expr_name != "":
            parent_name = tree.expr_name

        if len(tree.children) == 1:
            return eval_tree(tree.children[0], parent_name)

        if parent_name == "expression":
            left, symbol, right = tree.children
            operator = symbol_to_expr_op[symbol.text]
            return operator(eval_tree(left, parent_name), eval_tree(right, parent_name))

        elif parent_name == "term":
            left, symbol, right = tree.children
            operator = symbol_to_term_op[symbol.text]
            return operator(eval_tree(left, parent_name), eval_tree(right, parent_name))

        elif parent_name == "term":
            op, target = tree.children
            return symbol_to_factor_op[op.text](eval_tree(target, parent_name))

        elif parent_name == "power":
            base, _, exponent = tree.children
            return exp_op(eval_tree(base, tree.expr_name), eval_tree(exponent, parent_name))

        elif parent_name == "primary":
            func, _, target, _ = tree.children
            return func_to_op[func.text](eval_tree(target, parent_name))

        elif parent_name == "pars":
            _, target, _ = tree.children
            return eval_tree(target, parent_name)

        else:
            raise ValueError(f"Wrong node called, type: {tree.expr_name}")

    variable_to_value = kwargs
    tree = parser.parse(expr)

    defined_vars = set(variable_to_value.keys())
    existing_vars = variable_names(tree)
    undefined_vars = existing_vars.difference(defined_vars)

    if undefined_vars:
        undefined_vars_str = ",".join(f"'{v}" for v in undefined_vars)
        raise ValueError(f"Several values are undefined for expr, define them in call: {undefined_vars_str}")

    return eval_tree(tree, "")


function_names = {"sin", "cos", "tg", "ctg", "exp"}


def variable_names(tree: Node) -> Set[str]:
    return {atom.text for atom in atoms(tree) if atom.text not in function_names and atom.text.isalpha()}


def eval_simple_expr(expr: str, **kwargs: float):
    return eval_expr(expr, **kwargs)


def eval_numpy_expr(expr: str, **kwargs: float):
    symbol_to_term_op = {
        "*": lambda x, y: x * y,
        "/": lambda x, y: x / y,
        "%": lambda x, y: x % y,
    }

    symbol_to_expr_op = {
        "+": lambda x, y: x + y,
        "-": lambda x, y: x - y,
    }

    symbol_to_factor_op = {
        "-": lambda x: -x,
        "+": lambda x: x,
    }

    exp_op = np.power

    func_to_op = {
        "sin": np.sin,
        "cos": np.cos,
        "exp": np.exp,
        "tg": np.tan,
        "ctg": lambda x: 1 / np.tan(x),
    }

    return eval_expr(
        expr,
        symbol_to_expr_op=symbol_to_expr_op,
        symbol_to_factor_op=symbol_to_factor_op,
        symbol_to_term_op=symbol_to_term_op,
        exp_op=exp_op,
        func_to_op=func_to_op,
        **kwargs,
    )


def eval_pytorch_expr(expr: str, **kwargs: float):
    symbol_to_term_op = {
        "*": torch.mul,
        "/": torch.div,
        "%": torch.fmod,
    }

    symbol_to_expr_op = {
        "+": torch.add,
        "-": torch.sub,
    }

    symbol_to_factor_op = {
        "-": torch.neg,
        "+": lambda x: x,
    }

    exp_op = torch.pow

    func_to_op = {
        "sin": torch.sin,
        "cos": torch.cos,
        "exp": torch.exp,
        "tg": torch.tan,
        "ctg": lambda x: torch.div(1, torch.tan(x)),
    }

    return eval_expr(
        expr,
        symbol_to_expr_op = symbol_to_expr_op,
        symbol_to_factor_op = symbol_to_factor_op,
        symbol_to_term_op= symbol_to_term_op,
        exp_op=exp_op,
        func_to_op=func_to_op,
        **kwargs,
    )
