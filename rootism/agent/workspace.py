"""
Repository Understanding, Cross-File Symbol Indexing & Dependency Graph (Step 71).

Provides multi-language repository intelligence across:
- Project layout & manifest discovery (`pyproject.toml`, `package.json`, `Cargo.toml`, `go.mod`, etc.)
- Top-level symbol discovery (classes, functions, structs, interfaces, exports)
- Cross-file import/dependency mapping so the coding agent understands downstream impact
  before editing files.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rootism.agent.safety import SafetyPolicyGuard

_IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "dist",
        "build",
        "target",
        ".next",
    }
)

_EXTENSION_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".html": "html",
    ".css": "css",
    ".sql": "sql",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".cs": "csharp",
    ".php": "php",
    ".sh": "bash",
    ".ps1": "powershell",
    ".lua": "lua",
    ".kt": "kotlin",
    ".swift": "swift",
    ".rb": "ruby",
}

_MANIFEST_FILES: tuple[str, ...] = (
    "pyproject.toml",
    "requirements.txt",
    "package.json",
    "tsconfig.json",
    "Cargo.toml",
    "go.mod",
    "CMakeLists.txt",
    "Makefile",
)

_GENERIC_SYMBOL_PATTERNS: dict[str, re.Pattern[str]] = {
    "javascript": re.compile(r"^\s*(?:export\s+)?(?:async\s+)?(?:function|class|const)\s+([A-Za-z_$][\w$]*)", re.M),
    "typescript": re.compile(
        r"^\s*(?:export\s+)?(?:async\s+)?(?:function|class|interface|type|enum|const)\s+([A-Za-z_$][\w$]*)",
        re.M,
    ),
    "rust": re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:fn|struct|enum|trait|mod)\s+([A-Za-z_]\w*)", re.M),
    "go": re.compile(r"^\s*(?:func(?:\s+\([^)]+\))?|type)\s+([A-Za-z_]\w*)", re.M),
    "c": re.compile(r"^(?:static\s+|inline\s+)?[A-Za-z_][\w\s\*]+\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*\{", re.M),
    "cpp": re.compile(r"^\s*(?:class|struct|namespace)\s+([A-Za-z_]\w*)", re.M),
}

_GENERIC_IMPORT_PATTERNS: dict[str, re.Pattern[str]] = {
    "javascript": re.compile(r"""(?:import\s+.*?from\s+|require\s*\(\s*)['"]([^'"]+)['"]"""),
    "typescript": re.compile(r"""import\s+.*?from\s+['"]([^'"]+)['"]"""),
    "rust": re.compile(r"^\s*use\s+([A-Za-z0-9_:]+)", re.M),
    "go": re.compile(r'"([A-Za-z0-9_./-]+)"'),
    "c": re.compile(r'^\s*#include\s+[<"]([^>"]+)[>"]', re.M),
    "cpp": re.compile(r'^\s*#include\s+[<"]([^>"]+)[>"]', re.M),
}


@dataclass(frozen=True)
class FileSymbolEntry:
    path: str
    language: str
    line_count: int
    symbols: tuple[str, ...]
    imports: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["symbols"] = list(self.symbols)
        d["imports"] = list(self.imports)
        return d


@dataclass(frozen=True)
class RepositoryOverview:
    workspace_root: str
    total_files: int
    language_file_counts: dict[str, int]
    detected_manifests: tuple[str, ...]
    declared_dependencies: dict[str, list[str]]
    indexed_files: tuple[FileSymbolEntry, ...]
    symbol_to_files: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_root": self.workspace_root,
            "total_files": self.total_files,
            "language_file_counts": self.language_file_counts,
            "detected_manifests": list(self.detected_manifests),
            "declared_dependencies": self.declared_dependencies,
            "indexed_files": [f.to_dict() for f in self.indexed_files],
            "symbol_to_files": self.symbol_to_files,
        }


class WorkspaceAnalyzer:
    """
    Scans a software repository inside `SafetyPolicyGuard.workspace_root` to build
    symbol maps, dependency lists, and cross-file navigation context.
    """

    def __init__(self, safety_guard: SafetyPolicyGuard, *, max_indexed_files: int = 400) -> None:
        self.safety_guard = safety_guard
        self.max_indexed_files = max_indexed_files

    def iter_workspace_files(self) -> list[Path]:
        root = self.safety_guard.workspace_root
        discovered: list[Path] = []

        def _walk(cur: Path) -> None:
            if len(discovered) >= self.max_indexed_files:
                return
            try:
                entries = sorted(cur.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except OSError:
                return
            for entry in entries:
                if entry.is_symlink():
                    continue
                if entry.is_dir():
                    if entry.name in _IGNORED_DIRS:
                        continue
                    _walk(entry)
                elif entry.is_file():
                    discovered.append(entry)
                    if len(discovered) >= self.max_indexed_files:
                        return

        _walk(root)
        return discovered

    def inspect_repository(self) -> RepositoryOverview:
        root = self.safety_guard.workspace_root
        files = self.iter_workspace_files()
        lang_counts: dict[str, int] = {}
        manifests: list[str] = []
        indexed: list[FileSymbolEntry] = []
        sym_map: dict[str, list[str]] = {}

        for f in files:
            rel = str(f.relative_to(root))
            if f.name in _MANIFEST_FILES:
                manifests.append(rel)

            lang = _EXTENSION_TO_LANG.get(f.suffix.lower())
            if lang is None:
                continue
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

            if f.stat().st_size > 512_000:
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            symbols, imports = self._extract_symbols_and_imports(text, lang)
            entry = FileSymbolEntry(
                path=rel,
                language=lang,
                line_count=text.count("\n") + (1 if text else 0),
                symbols=tuple(symbols[:60]),
                imports=tuple(imports[:40]),
            )
            indexed.append(entry)
            for sym in entry.symbols:
                sym_map.setdefault(sym, []).append(rel)

        deps = self._parse_dependencies(manifests)
        return RepositoryOverview(
            workspace_root=str(root),
            total_files=len(files),
            language_file_counts=lang_counts,
            detected_manifests=tuple(manifests),
            declared_dependencies=deps,
            indexed_files=tuple(indexed),
            symbol_to_files=sym_map,
        )

    def _extract_symbols_and_imports(self, text: str, lang: str) -> tuple[list[str], list[str]]:
        if lang == "python":
            try:
                tree = ast.parse(text)
                syms: list[str] = []
                imps: list[str] = []
                for node in tree.body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        syms.append(node.name)
                        if isinstance(node, ast.ClassDef):
                            for sub in node.body:
                                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                    syms.append(f"{node.name}.{sub.name}")
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            imps.append(alias.name)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imps.append(node.module)
                return syms, imps
            except SyntaxError:
                return [], []

        syms_out: list[str] = []
        imps_out: list[str] = []
        sym_pat = _GENERIC_SYMBOL_PATTERNS.get(lang)
        if sym_pat is not None:
            syms_out = list(dict.fromkeys(sym_pat.findall(text)))
        imp_pat = _GENERIC_IMPORT_PATTERNS.get(lang)
        if imp_pat is not None:
            imps_out = list(dict.fromkeys(imp_pat.findall(text)))
        return syms_out, imps_out

    def _parse_dependencies(self, manifest_rel_paths: list[str]) -> dict[str, list[str]]:
        deps: dict[str, list[str]] = {}
        root = self.safety_guard.workspace_root
        for rel in manifest_rel_paths:
            p = root / rel
            try:
                raw = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if p.name == "package.json":
                try:
                    pkg = json.loads(raw)
                    combined = sorted(
                        set(pkg.get("dependencies", {}).keys())
                        | set(pkg.get("devDependencies", {}).keys())
                    )
                    deps[rel] = combined
                except Exception:
                    pass
            elif p.name == "pyproject.toml":
                found = re.findall(r'"([A-Za-z0-9_.-]+)\s*(?:>=|==|~=|>|<)', raw)
                if found:
                    deps[rel] = sorted(set(found))
            elif p.name == "requirements.txt":
                pkgs = [
                     re.split(r"[>=<~!;\s]", line.strip())[0]
                    for line in raw.splitlines()
                    if line.strip() and not line.strip().startswith("#")
                ]
                deps[rel] = [x for x in pkgs if x]
        return deps
