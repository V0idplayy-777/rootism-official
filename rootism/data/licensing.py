"""
Commercial-Use License Policy Engine & Deep File-Header Verifier (Steps 35, 37, 38).

Enforces Rootism's absolute commercial-use licensing mandate:
- Explicit SPDX commercial allowlist; unknown/ambiguous/copyleft/non-commercial licenses are rejected.
- Compound SPDX boolean expression parser (`MIT AND Apache-2.0` -> allowed; `MIT AND GPL-3.0-only` -> rejected;
  `MIT OR GPL-2.0-only` -> allowed under MIT election with recorded election notice).
- Deep file-level, submodule-level, and vendored-directory header scanner catching copyleft or
  non-commercial files nested inside permissively licensed parent repositories.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SpdxPolicyRule:
    spdx_id: str
    commercial_allowed: bool
    requires_attribution: bool
    copyleft_or_viral: bool
    summary: str


APPROVED_COMMERCIAL_SPDX: dict[str, SpdxPolicyRule] = {
    "MIT": SpdxPolicyRule("MIT", True, True, False, "Permissive commercial license; preserve copyright & permission notice."),
    "Apache-2.0": SpdxPolicyRule("Apache-2.0", True, True, False, "Permissive commercial license with express patent grant; preserve NOTICE/attribution."),
    "BSD-2-Clause": SpdxPolicyRule("BSD-2-Clause", True, True, False, "Permissive commercial license; preserve copyright notice."),
    "BSD-3-Clause": SpdxPolicyRule("BSD-3-Clause", True, True, False, "Permissive commercial license; no endorsement + preserve copyright notice."),
    "0BSD": SpdxPolicyRule("0BSD", True, False, False, "Zero-Clause BSD public-domain-equivalent commercial license."),
    "ISC": SpdxPolicyRule("ISC", True, True, False, "Permissive commercial license functionally equivalent to simplified BSD/MIT."),
    "CC0-1.0": SpdxPolicyRule("CC0-1.0", True, False, False, "Creative Commons Public Domain Dedication; unrestricted commercial use."),
    "Unlicense": SpdxPolicyRule("Unlicense", True, False, False, "Public domain dedication; unrestricted commercial use."),
    "Python-2.0": SpdxPolicyRule("Python-2.0", True, True, False, "PSF permissive commercial license."),
    "Zlib": SpdxPolicyRule("Zlib", True, True, False, "Permissive commercial license."),
    "BSL-1.0": SpdxPolicyRule("BSL-1.0", True, True, False, "Boost Software License 1.0; permissive commercial use."),
    "CC-BY-4.0": SpdxPolicyRule("CC-BY-4.0", True, True, False, "Creative Commons Attribution 4.0; commercial use permitted with attribution."),
    "Rootism-Proprietary-Original": SpdxPolicyRule(
        "Rootism-Proprietary-Original",
        True,
        False,
        False,
        "First-party original data generated and verified internally by Rootism.",
    ),
}


@dataclass(frozen=True)
class LicenseDecision:
    expression: str
    approved_for_commercial_training: bool
    elected_licenses: tuple[str, ...]
    requires_attribution: bool
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["elected_licenses"] = list(self.elected_licenses)
        return data


_SPDX_TOKEN_RE = re.compile(
    r"\(|\)|\bAND\b|\bOR\b|\bWITH\b|[A-Za-z0-9][A-Za-z0-9.-]*"
)


def _tokenize_spdx(expr: str) -> list[tuple[str, str]]:
    """Tokenize an SPDX expression into ``(type, value)`` pairs.

    Raises ``ValueError`` if any non-whitespace character is not part of a
    recognised token (identifier, operator, or parenthesis).
    """
    tokens: list[tuple[str, str]] = []
    last_end = 0
    for m in _SPDX_TOKEN_RE.finditer(expr):
        gap = expr[last_end : m.start()]
        if gap.strip():
            raise ValueError(f"Unexpected character(s) in SPDX expression: {gap!r}")
        val = m.group(0)
        if val == "(":
            tokens.append(("LPAREN", val))
        elif val == ")":
            tokens.append(("RPAREN", val))
        elif val == "AND":
            tokens.append(("AND", val))
        elif val == "OR":
            tokens.append(("OR", val))
        elif val == "WITH":
            tokens.append(("WITH", val))
        else:
            tokens.append(("LIC", val))
        last_end = m.end()
    tail = expr[last_end:]
    if tail.strip():
        raise ValueError(f"Unexpected trailing character(s) in SPDX expression: {tail!r}")
    if not tokens:
        raise ValueError("Empty SPDX expression")
    return tokens


class _SpdxExpressionParser:
    """Recursive-descent parser for SPDX license expressions.

    Grammar (AND binds tighter than OR, per SPDX spec):
        expression := or_expr
        or_expr    := and_expr (OR and_expr)*
        and_expr   := term (AND term)*
        term       := LPAREN expression RPAREN
                    | LICENSE_ID
    """

    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self._tokens = tokens
        self._pos = 0
        self._error: str | None = None

    def _peek(self) -> tuple[str, str] | None:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _advance(self) -> tuple[str, str] | None:
        tok = self._peek()
        if tok is not None:
            self._pos += 1
        return tok

    @property
    def error(self) -> str | None:
        return self._error

    def parse(self) -> tuple | None:
        node = self._parse_or()
        if self._error is None and self._pos < len(self._tokens):
            self._error = f"Unexpected trailing token '{self._tokens[self._pos][1]}'"
            return None
        return node

    def _parse_or(self) -> tuple | None:
        node = self._parse_and()
        if node is None or self._error is not None:
            return None
        while self._peek() is not None and self._peek()[0] == "OR":
            self._advance()
            rhs = self._parse_and()
            if rhs is None:
                return None
            node = ("OR", node, rhs)
        return node

    def _parse_and(self) -> tuple | None:
        node = self._parse_term()
        if node is None or self._error is not None:
            return None
        while self._peek() is not None and self._peek()[0] == "AND":
            self._advance()
            rhs = self._parse_term()
            if rhs is None:
                return None
            node = ("AND", node, rhs)
        return node

    def _parse_term(self) -> tuple | None:
        tok = self._peek()
        if tok is None:
            self._error = "Unexpected end of expression"
            return None
        if tok[0] == "LPAREN":
            self._advance()
            node = self._parse_or()
            if node is None:
                return None
            tok = self._peek()
            if tok is None or tok[0] != "RPAREN":
                self._error = "Expected ')' to close parenthesised expression"
                return None
            self._advance()
            return node
        if tok[0] == "LIC":
            self._advance()
            lic_val = tok[1]
            # SPDX WITH: `License WITH Exception` (postfix, not binary)
            next_tok = self._peek()
            if next_tok is not None and next_tok[0] == "WITH":
                self._advance()
                exc_tok = self._peek()
                if exc_tok is None or exc_tok[0] != "LIC":
                    self._error = "WITH operator requires a following license identifier"
                    return None
                self._advance()
                return ("WITH", ("LIC", lic_val), ("LIC", exc_tok[1]))
            return ("LIC", lic_val)
        if tok[0] == "WITH":
            self._error = "WITH operator requires a preceding license identifier"
            return None
        self._error = f"Unexpected token '{tok[1]}'"
        return None


def _evaluate_spdx_ast(ast: tuple) -> tuple[bool, list[str], bool, str | None]:
    """Evaluate a parsed SPDX AST.

    Returns ``(satisfiable, elected_licenses, requires_attribution, rejection_reason)``.
    - AND: all sub-expressions must be satisfiable; elected = union of all.
    - OR: at least one sub-expression must be satisfiable; elect first satisfiable branch.
    - LIC: satisfiable iff in APPROVED_COMMERCIAL_SPDX with commercial_allowed=True.
    - WITH: evaluates the left (license) side only.
    """
    kind = ast[0]

    if kind == "LIC":
        lic = ast[1]
        rule = APPROVED_COMMERCIAL_SPDX.get(lic)
        if rule is None:
            return (False, [], False, f"Unknown or unverified license identifier '{lic}'")
        if not rule.commercial_allowed:
            return (False, [], False, f"License '{lic}' is not permitted for commercial use")
        return (True, [lic], rule.requires_attribution, None)

    if kind == "AND":
        left = _evaluate_spdx_ast(ast[1])
        right = _evaluate_spdx_ast(ast[2])
        if left[0] and right[0]:
            return (True, left[1] + right[1], left[2] or right[2], None)
        reasons = [r for _, _, _, r in (left, right) if r is not None]
        return (False, [], False, "; ".join(reasons) if reasons else "AND clause not satisfiable")

    if kind == "OR":
        left = _evaluate_spdx_ast(ast[1])
        if left[0]:
            return (True, left[1], left[2], None)
        right = _evaluate_spdx_ast(ast[2])
        if right[0]:
            return (True, right[1], right[2], None)
        reasons = [r for _, _, _, r in (left, right) if r is not None]
        return (False, [], False, "; ".join(reasons) if reasons else "OR clause not satisfiable")

    if kind == "WITH":
        return _evaluate_spdx_ast(ast[1])

    return (False, [], False, "Internal error evaluating SPDX expression")


def evaluate_spdx_expression(expression: str) -> LicenseDecision:
    """
    Evaluate a single SPDX identifier or compound boolean expression.

    Supports the full SPDX expression grammar:
    - ``MIT`` — single identifier
    - ``MIT AND Apache-2.0`` — both must be approved
    - ``MIT OR GPL-2.0-only`` — at least one must be approved (elects first approved)
    - ``(MIT OR Apache-2.0) AND GPL-3.0-only`` — correct precedence with parentheses

    Operator precedence: AND binds tighter than OR.
    Malformed expressions are rejected.
    """
    cleaned = (expression or "").strip()
    if not cleaned:
        return LicenseDecision(
            expression=expression or "",
            approved_for_commercial_training=False,
            elected_licenses=(),
            requires_attribution=False,
            rejection_reason="Empty or missing SPDX license identifier.",
        )

    try:
        tokens = _tokenize_spdx(cleaned)
    except ValueError as exc:
        return LicenseDecision(
            expression=cleaned,
            approved_for_commercial_training=False,
            elected_licenses=(),
            requires_attribution=False,
            rejection_reason=str(exc),
        )

    parser = _SpdxExpressionParser(tokens)
    ast = parser.parse()
    if ast is None or parser.error is not None:
        return LicenseDecision(
            expression=cleaned,
            approved_for_commercial_training=False,
            elected_licenses=(),
            requires_attribution=False,
            rejection_reason=parser.error or "Malformed SPDX expression.",
        )

    satisfiable, elected, requires_attr, rejection = _evaluate_spdx_ast(ast)
    return LicenseDecision(
        expression=cleaned,
        approved_for_commercial_training=satisfiable,
        elected_licenses=tuple(elected),
        requires_attribution=requires_attr,
        rejection_reason=rejection,
    )


_SPDX_HEADER_RE = re.compile(r"SPDX-License-Identifier:\s*([^\n\r*/#;]+)", flags=re.IGNORECASE)

_PROHIBITED_HEADER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("GPL_COPYLEFT", re.compile(r"GNU\s+(?:Lesser\s+|Affero\s+)?General\s+Public\s+License", re.IGNORECASE)),
    ("AGPL_COPYLEFT", re.compile(r"\bAGPL(?:-|\s+v?[0-9])", re.IGNORECASE)),
    ("GPL_REFERENCE", re.compile(r"licensed\s+under\s+the\s+GPL", re.IGNORECASE)),
    ("CC_NON_COMMERCIAL", re.compile(r"Creative\s+Commons[^\n]{0,60}NonCommercial|\bCC-BY-NC\b", re.IGNORECASE)),
    ("SHARE_ALIKE_VIRAL", re.compile(r"\bCC-BY-SA\b|ShareAlike\s+4\.0", re.IGNORECASE)),
    ("SSPL_BUSL_SOURCE_AVAILABLE", re.compile(r"Server\s+Side\s+Public\s+License|Business\s+Source\s+License|\bCommons\s+Clause\b", re.IGNORECASE)),
    ("RESEARCH_ONLY_RESTRICTION", re.compile(r"for\s+(?:academic|non-commercial|research)\s+(?:use|purposes)\s+only", re.IGNORECASE)),
    ("THIRD_PARTY_ALL_RIGHTS_RESERVED", re.compile(r"All\s+Rights\s+Reserved\.?\s*(?:Proprietary|Confidential|Do\s+not\s+distribute)", re.IGNORECASE)),
)

_VENDORED_OR_SUBMODULE_DIRS: frozenset[str] = frozenset(
    {
        "vendor",
        "vendored",
        "third_party",
        "third-party",
        "3rdparty",
        "external",
        "node_modules",
        "submodules",
    }
)


@dataclass(frozen=True)
class FileLicenseScanResult:
    file_path: str
    parent_repo_license: str
    approved: bool
    effective_license: str
    quarantine_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeepFileLicenseVerifier:
    """
    Verifies both repository-level and individual file-level license provenance
    so nested submodules, vendored code, or copyleft file headers never slip through.
    """

    def scan_file(
        self,
        *,
        file_path: str,
        content: str,
        parent_repo_license: str,
        has_verified_submodule_manifest: bool = False,
    ) -> FileLicenseScanResult:
        parent_decision = evaluate_spdx_expression(parent_repo_license)
        if not parent_decision.approved_for_commercial_training:
            return FileLicenseScanResult(
                file_path=file_path,
                parent_repo_license=parent_repo_license,
                approved=False,
                effective_license=parent_repo_license,
                quarantine_reason=f"Parent repository license rejected: {parent_decision.rejection_reason}",
            )

        norm_parts = [p.lower() for p in file_path.replace("\\", "/").split("/") if p]
        if ".gitmodules" in norm_parts:
            return FileLicenseScanResult(
                file_path=file_path,
                parent_repo_license=parent_repo_license,
                approved=False,
                effective_license="SUBMODULE_MANIFEST",
                quarantine_reason="Submodule reference requires independent license verification.",
            )

        if not has_verified_submodule_manifest:
            vendored_hits = _VENDORED_OR_SUBMODULE_DIRS.intersection(norm_parts[:-1])
            if vendored_hits:
                return FileLicenseScanResult(
                    file_path=file_path,
                    parent_repo_license=parent_repo_license,
                    approved=False,
                    effective_license="UNVERIFIED_VENDORED_TREE",
                    quarantine_reason=(
                        f"File resides inside vendored/third-party directory {sorted(vendored_hits)} "
                        "without an independent license clearance record."
                    ),
                )

        # Inspect top 8 KB and bottom 2 KB where license notices & footers appear
        head_and_tail = content[:8192] + ("\n" + content[-2048:] if len(content) > 8192 else "")

        spdx_match = _SPDX_HEADER_RE.search(head_and_tail)
        if spdx_match:
            file_expr = spdx_match.group(1).strip()
            file_decision = evaluate_spdx_expression(file_expr)
            if not file_decision.approved_for_commercial_training:
                return FileLicenseScanResult(
                    file_path=file_path,
                    parent_repo_license=parent_repo_license,
                    approved=False,
                    effective_license=file_expr,
                    quarantine_reason=(
                        f"File-level SPDX header '{file_expr}' overrides parent '{parent_repo_license}' "
                        f"and is prohibited: {file_decision.rejection_reason}"
                    ),
                )
            effective = "+".join(file_decision.elected_licenses)
        else:
            effective = "+".join(parent_decision.elected_licenses)

        for code, pattern in _PROHIBITED_HEADER_PATTERNS:
            if pattern.search(head_and_tail):
                return FileLicenseScanResult(
                    file_path=file_path,
                    parent_repo_license=parent_repo_license,
                    approved=False,
                    effective_license=code,
                    quarantine_reason=f"Detected prohibited license/restriction notice ({code}) in file header/footer.",
                )

        return FileLicenseScanResult(
            file_path=file_path,
            parent_repo_license=parent_repo_license,
            approved=True,
            effective_license=effective,
            quarantine_reason=None,
        )
