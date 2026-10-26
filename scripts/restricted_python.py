import ast
import copy
import operator
import time


class RestrictedCodeError(ValueError):
    pass


class ReturnValue(Exception):
    def __init__(self, value):
        self.value = value


class LoopBreak(Exception):
    pass


class LoopContinue(Exception):
    pass


BUILTINS = {"abs", "min", "max", "sum", "range", "len", "sorted", "set", "list", "tuple", "reversed", "enumerate", "int", "bool", "any", "all"}
METHODS = {"count", "lower", "split", "find", "rfind"}
NODES = {
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign, ast.AnnAssign, ast.AugAssign,
    ast.If, ast.For, ast.Break, ast.Continue, ast.Pass, ast.Expr, ast.Name, ast.Constant, ast.Load, ast.Store,
    ast.List, ast.Tuple, ast.Set, ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.comprehension,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Call, ast.keyword, ast.Attribute,
    ast.Subscript, ast.Slice, ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod,
    ast.UAdd, ast.USub, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
}


def bound(value, depth=0, remaining=None):
    if remaining is None:
        remaining = [2048]
    remaining[0] -= 1
    if remaining[0] < 0:
        raise RestrictedCodeError("Value traversal limit")
    if depth > 8:
        raise RestrictedCodeError("Value nesting limit")
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        if abs(value) > 10**9:
            raise RestrictedCodeError("Integer magnitude limit")
        return value
    if type(value) in (str, list, tuple, set):
        if len(value) > 128:
            raise RestrictedCodeError("Container length limit")
        if type(value) is not str:
            for item in value:
                bound(item, depth + 1, remaining)
        return value
    raise RestrictedCodeError("Unsupported runtime value")


class RestrictedFunction:
    def __init__(self, source, arguments):
        if not isinstance(source, str) or len(source) > 4096:
            raise RestrictedCodeError("Source length limit")
        try:
            tree = ast.parse(source)
        except (SyntaxError, RecursionError, ValueError) as error:
            raise RestrictedCodeError("Invalid Python syntax") from error
        nodes = list(ast.walk(tree))
        if len(nodes) > 400 or any(type(node) not in NODES for node in nodes):
            raise RestrictedCodeError("Unsupported Python construct")
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise RestrictedCodeError("Expected one function only")
        fn = tree.body[0]
        if fn.name != "solve" or fn.decorator_list or fn.args.defaults or fn.args.kw_defaults or fn.args.vararg or fn.args.kwarg or fn.args.posonlyargs or fn.args.kwonlyargs:
            raise RestrictedCodeError("Expected a plain solve function")
        if [arg.arg for arg in fn.args.args] != list(arguments):
            raise RestrictedCodeError("Function arguments differ from the task")
        parents = {child: node for node in nodes for child in ast.iter_child_nodes(node)}
        local_names = {arg.arg for arg in fn.args.args} | {node.id for node in nodes if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
        for node in nodes:
            if isinstance(node, (ast.Name, ast.arg)):
                name = node.id if isinstance(node, ast.Name) else node.arg
                if name.startswith("_"):
                    raise RestrictedCodeError("Private names are not supported")
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and name not in local_names | BUILTINS:
                    raise RestrictedCodeError("Unsupported global name")
            if isinstance(node, ast.Constant):
                bound(node.value)
            if isinstance(node, ast.Attribute) and node.attr not in METHODS:
                raise RestrictedCodeError("Unsupported attribute")
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id not in BUILTINS:
                    raise RestrictedCodeError("Unsupported function call")
                if not isinstance(node.func, (ast.Name, ast.Attribute)) or any(k.arg is None for k in node.keywords):
                    raise RestrictedCodeError("Indirect calls are not supported")
            if isinstance(node, ast.GeneratorExp):
                parent = parents.get(node)
                if not isinstance(parent, ast.Call) or not isinstance(parent.func, ast.Name) or parent.func.id not in {"sum", "min", "max", "sorted", "set", "list", "tuple"} or len(parent.args) != 1 or parent.keywords or node not in parent.args:
                    raise RestrictedCodeError("Generators are only supported as direct aggregation arguments")
            if isinstance(node, ast.comprehension) and node.is_async:
                raise RestrictedCodeError("Async comprehensions are not supported")
        self.fn = fn
        self.arguments = list(arguments)

    def tick(self):
        self.steps += 1
        if self.steps > 4000 or time.monotonic() > self.deadline:
            raise RestrictedCodeError("Execution step limit")

    def assign(self, target, value, env):
        self.tick()
        if isinstance(target, ast.Name) and target.id not in BUILTINS and target.id != "solve":
            env[target.id] = bound(value)
        elif isinstance(target, (ast.Tuple, ast.List)) and type(value) in (list, tuple) and len(target.elts) == len(value):
            for child, item in zip(target.elts, value, strict=True):
                self.assign(child, item, env)
        else:
            raise RestrictedCodeError("Unsupported assignment target")

    def binary(self, op, left, right):
        if isinstance(op, ast.Add) and type(left) is type(right) and type(left) in (str, list, tuple):
            if len(left) + len(right) > 128:
                raise RestrictedCodeError("Concatenation length limit")
            return left + right
        if isinstance(op, ast.Mult) and ((type(left) in (str, list, tuple) and type(right) is int) or (type(right) in (str, list, tuple) and type(left) is int)):
            sequence, count = (left, right) if type(right) is int else (right, left)
            if len(sequence) * max(0, count) > 128 or abs(count) > 128:
                raise RestrictedCodeError("Repetition length limit")
            return sequence * count
        if type(left) not in (int, bool) or type(right) not in (int, bool):
            raise RestrictedCodeError("Arithmetic requires integers")
        functions = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
        return bound(functions[type(op)](left, right))

    def comprehension(self, node, env):
        if len(node.generators) != 1:
            raise RestrictedCodeError("Only one comprehension iterator is supported")
        generator = node.generators[0]
        result = []
        local = env.copy()
        for value in self.expr(generator.iter, env):
            self.tick()
            self.assign(generator.target, value, local)
            if all(self.expr(condition, local) for condition in generator.ifs):
                result.append(self.expr(node.elt, local))
                bound(result)
        return set(result) if isinstance(node, ast.SetComp) else result

    def call(self, node, env):
        receiver = self.expr(node.func.value, env) if isinstance(node.func, ast.Attribute) else None
        args = [self.expr(item, env) for item in node.args]
        kwargs = {item.arg: self.expr(item.value, env) for item in node.keywords}
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if kwargs and (name != "sorted" or set(kwargs) != {"reverse"} or type(kwargs["reverse"]) is not bool):
                raise RestrictedCodeError("Unsupported keyword arguments")
            if name == "range":
                if not 1 <= len(args) <= 3 or any(type(item) is not int for item in args):
                    raise RestrictedCodeError("range requires 1 to 3 integer arguments")
                value = range(*args)
                if len(value) > 128:
                    raise RestrictedCodeError("Range length limit")
                return list(value)
            if name == "int" and (len(args) != 1 or type(args[0]) not in (int, bool)):
                raise RestrictedCodeError("int conversion is limited to booleans and integers")
            functions = {"abs": abs, "min": min, "max": max, "sum": sum, "len": len, "sorted": sorted,
                         "set": set, "list": list, "tuple": tuple, "bool": bool, "int": int,
                         "reversed": lambda value: list(reversed(value)), "enumerate": lambda value: list(enumerate(value)), "any": any, "all": all}
            return bound(functions[name](*args, **kwargs))
        if kwargs:
            raise RestrictedCodeError("Method keywords are not supported")
        name = node.func.attr
        if name == "count" and type(receiver) in (str, list, tuple):
            return bound(receiver.count(*args))
        if type(receiver) is str and name in {"lower", "split", "find", "rfind"}:
            functions = {"lower": str.lower, "split": str.split, "find": str.find, "rfind": str.rfind}
            return bound(functions[name](receiver, *args))
        raise RestrictedCodeError("Unsupported method receiver")

    def expr(self, node, env):
        self.tick()
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise NameError(node.id)
            return env[node.id]
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            values = [self.expr(item, env) for item in node.elts]
            return bound(tuple(values) if isinstance(node, ast.Tuple) else set(values) if isinstance(node, ast.Set) else values)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            return bound(self.comprehension(node, env))
        if isinstance(node, ast.BinOp):
            return self.binary(node.op, self.expr(node.left, env), self.expr(node.right, env))
        if isinstance(node, ast.UnaryOp):
            value = self.expr(node.operand, env)
            if isinstance(node.op, ast.Not):
                return not value
            if type(value) not in (int, bool):
                raise RestrictedCodeError("Unary arithmetic requires an integer")
            return bound(-value if isinstance(node.op, ast.USub) else +value)
        if isinstance(node, ast.BoolOp):
            for item in node.values:
                value = self.expr(item, env)
                if isinstance(node.op, ast.And) and not value or isinstance(node.op, ast.Or) and value:
                    return value
            return value
        if isinstance(node, ast.IfExp):
            return self.expr(node.body if self.expr(node.test, env) else node.orelse, env)
        if isinstance(node, ast.Compare):
            left = self.expr(node.left, env)
            functions = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
                         ast.Gt: operator.gt, ast.GtE: operator.ge, ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}
            for op, expression in zip(node.ops, node.comparators, strict=True):
                right = self.expr(expression, env)
                if not functions[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.Subscript):
            value = self.expr(node.value, env)
            if type(value) not in (list, tuple, str):
                raise RestrictedCodeError("Unsupported subscript receiver")
            if isinstance(node.slice, ast.Slice):
                index = slice(*(self.expr(item, env) if item is not None else None for item in (node.slice.lower, node.slice.upper, node.slice.step)))
            else:
                index = self.expr(node.slice, env)
            return bound(value[index])
        if isinstance(node, ast.Call):
            return self.call(node, env)
        raise RestrictedCodeError("Unsupported expression")

    def statements(self, body, env):
        for node in body:
            self.tick()
            if isinstance(node, ast.Return):
                raise ReturnValue(self.expr(node.value, env) if node.value is not None else None)
            if isinstance(node, ast.Assign):
                value = self.expr(node.value, env)
                for target in node.targets:
                    self.assign(target, value, env)
            elif isinstance(node, ast.AnnAssign):
                if node.value is not None:
                    self.assign(node.target, self.expr(node.value, env), env)
            elif isinstance(node, ast.AugAssign):
                left, right = self.expr(node.target, env), self.expr(node.value, env)
                if type(left) not in (int, bool) or type(right) not in (int, bool):
                    raise RestrictedCodeError("Augmented assignment is limited to integers")
                self.assign(node.target, self.binary(node.op, left, right), env)
            elif isinstance(node, ast.If):
                self.statements(node.body if self.expr(node.test, env) else node.orelse, env)
            elif isinstance(node, ast.For):
                values = self.expr(node.iter, env)
                if type(values) not in (list, tuple, set, str):
                    raise RestrictedCodeError("Unsupported loop iterable")
                broke = False
                for value in list(values):
                    self.tick()
                    self.assign(node.target, value, env)
                    try:
                        self.statements(node.body, env)
                    except LoopContinue:
                        continue
                    except LoopBreak:
                        broke = True
                        break
                if not broke:
                    self.statements(node.orelse, env)
            elif isinstance(node, ast.Break):
                raise LoopBreak()
            elif isinstance(node, ast.Continue):
                raise LoopContinue()
            elif isinstance(node, ast.Expr):
                self.expr(node.value, env)
            elif not isinstance(node, ast.Pass):
                raise RestrictedCodeError("Unsupported statement")

    def __call__(self, **inputs):
        if set(inputs) != set(self.arguments):
            raise RestrictedCodeError("Input arguments differ")
        for value in inputs.values():
            bound(value)
        self.steps = 0
        self.deadline = time.monotonic() + 0.1
        try:
            self.statements(self.fn.body, copy.deepcopy(inputs))
        except ReturnValue as result:
            return bound(result.value)
        except (LoopBreak, LoopContinue) as error:
            raise RestrictedCodeError("Loop control outside a loop") from error
        return None
