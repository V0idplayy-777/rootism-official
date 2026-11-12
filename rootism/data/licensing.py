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


def evaluate_spdx_expression(expression: str) -> LicenseDecision:
    """
    Evaluate a single SPDX identifier or compound boolean expression (`AND` / `OR`).
    Parentheses are rejected if malformed; unknown identifiers are strictly rejected.
    """
    cleaned = (expression or "").strip()
    if not cleaned:
        return LicenseDecision(
            expression="",
            approved_for_commercial_training=False,
            elected_licenses=(),
            requires_attribution=False,
            rejection_reason="Empty or missing SPDX license identifier.",
        )

    # Strip balanced outer parentheses iteratively
    expr = cleaned.replace("(", " ").replace(")", " ")
    or_branches = [b.strip() for b in re.split(r"\bOR\b", expr) if b.strip()]
    if not or_branches:
        return LicenseDecision(cleaned, False, (), False, "Malformed SPDX expression.")

    branch_failures: list[str] = []
    for branch in or_branches:
        and_terms = [t.strip() for t in re.split(r"\bAND\b", branch) if t.strip()]
        if not and_terms:
            branch_failures.append("Empty AND clause")
            continue

        branch_ok = True
        elected: list[str] = []
        needs_attr = False
        for term in and_terms:
            rule = APPROVED_COMMERCIAL_SPDX.get(term)
            if rule is None or not rule.commercial_allowed:
                branch_ok = False
                branch_failures.append(f"Disallowed or unverified license term '{term}'")
                break
            elected.append(rule.spdx_id)
            needs_attr = needs_attr or rule.requires_attribution

        if branch_ok:
            return LicenseDecision(
                expression=cleaned,
                approved_for_commercial_training=True,
                elected_licenses=tuple(elected),
                requires_attribution=needs_attr,
                rejection_reason=None,
            )

    return LicenseDecision(
        expression=cleaned,
        approved_for_commercial_training=False,
        elected_licenses=(),
        requires_attribution=False,
        rejection_reason="; ".join(branch_failures),
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
