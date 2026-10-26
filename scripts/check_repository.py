#!/usr/bin/env python3
"""Check source syntax, undefined names, internal imports, and release links."""

import ast
import importlib.util
from pathlib import Path
import re
import subprocess
from pyflakes.checker import Checker
from pyflakes.messages import UndefinedName, UndefinedLocal

ROOT = Path(__file__).resolve().parents[1]


def check_internal_imports(files):
    modules = {".".join(p.relative_to(ROOT).with_suffix("").parts).removesuffix(".__init__"): p
               for p in files if p.suffix == ".py" and p.relative_to(ROOT).parts[0] == "verl"}
    exports = {}
    for module, path in modules.items():
        names = set()
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                names.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    names.update(item.id for item in ast.walk(target) if isinstance(item, ast.Name))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update(item.asname or item.name.split(".")[0] for item in node.names)
        exports[module] = names
    errors = []
    for module, path in modules.items():
        package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
        for node in ast.walk(ast.parse(path.read_text())):
            targets = []
            if isinstance(node, ast.Import):
                targets = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                target = "." * node.level + (node.module or "")
                target = importlib.util.resolve_name(target, package) if node.level else target
                targets = [target]
                if target in modules:
                    for item in node.names:
                        if item.name != "*" and item.name not in exports[target] and target + "." + item.name not in modules:
                            errors.append(f"{path.relative_to(ROOT)}:{node.lineno}: missing name {target}.{item.name}")
            for target in targets:
                if (target == "verl" or target.startswith("verl.")) and target not in modules:
                    errors.append(f"{path.relative_to(ROOT)}:{node.lineno}: missing module {target}")
    return errors


def main():
    files = subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT).decode().split("\0")
    files = [ROOT / name for name in files if name and (ROOT / name).is_file()]
    errors = []
    for path in files:
        if path.suffix == ".py":
            try:
                filename = str(path.relative_to(ROOT))
                tree = ast.parse(path.read_text(), filename=filename)
                errors.extend(str(message) for message in Checker(tree, filename=filename).messages
                              if isinstance(message, (UndefinedName, UndefinedLocal)))
            except SyntaxError as error:
                errors.append(str(error))
        elif path.suffix == ".sh":
            result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
            if result.returncode:
                errors.append(result.stderr)
        if path.suffix == ".md":
            content = path.read_text()
            links = re.findall(r"\]\(([^\s)]+)\)", content) + re.findall(r'src="([^"]+)"', content)
            for target in links:
                if target.startswith(("http:", "https:", "mailto:", "#")):
                    continue
                target = target.split("#", 1)[0]
                if not (path.parent / target).exists():
                    errors.append(f"{path.relative_to(ROOT)}: missing link {target}")
        if path.name == ".DS_Store":
            errors.append(f"Unwanted platform artifact: {path}")
    for name in ("README.md", "LICENSE", "CITATION.cff", "CITATION.bib", "configs/models.json", "assets/overview.png", "assets/method.png"):
        if not (ROOT / name).is_file():
            errors.append(f"Missing release file: {name}")
    if not errors:
        errors.extend(check_internal_imports(files))
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Checked {len(files)} files: syntax, undefined names, internal imports, and documentation links passed.")


if __name__ == "__main__":
    main()
