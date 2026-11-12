"""
Ecosystem & Framework Competence Registry (Step 31).

Explicitly models major commercial-friendly frameworks and ecosystems across
Rootism's target languages, plus workspace manifest detection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FrameworkSpec:
    """Specification of a framework ecosystem, license, manifest markers, and idioms."""

    slug: str
    display_name: str
    language: str
    domain: str
    spdx_license: str
    manifest_files: tuple[str, ...]
    package_identifiers: tuple[str, ...]
    core_concepts: tuple[str, ...]
    common_pitfalls: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SUPPORTED_FRAMEWORKS: tuple[FrameworkSpec, ...] = (
    FrameworkSpec(
        slug="react",
        display_name="React",
        language="typescript",
        domain="frontend_ui",
        spdx_license="MIT",
        manifest_files=("package.json",),
        package_identifiers=('"react"', "'react'"),
        core_concepts=("hooks (useState, useEffect, useMemo)", "unidirectional data flow", "JSX reconciliation", "Server/Client Components"),
        common_pitfalls=("stale hook dependency closures", "mutating state objects in place", "missing stable list keys"),
    ),
    FrameworkSpec(
        slug="nextjs",
        display_name="Next.js",
        language="typescript",
        domain="fullstack_web",
        spdx_license="MIT",
        manifest_files=("package.json", "next.config.js", "next.config.mjs", "next.config.ts"),
        package_identifiers=('"next"',),
        core_concepts=("App Router", "Route Handlers", "SSR / SSG / ISR", "middleware"),
        common_pitfalls=("using browser-only window APIs inside Server Components", "leaking server secrets via NEXT_PUBLIC_"),
    ),
    FrameworkSpec(
        slug="nodejs",
        display_name="Node.js",
        language="javascript",
        domain="backend_runtime",
        spdx_license="MIT",
        manifest_files=("package.json",),
        package_identifiers=('"engines"', "node:"),
        core_concepts=("CommonJS vs ESM", "Streams & Buffers", "EventEmitter", "fs/promises", "worker_threads"),
        common_pitfalls=("blocking event loop with synchronous fs/crypto calls", "unbounded stream backpressure"),
    ),
    FrameworkSpec(
        slug="express",
        display_name="Express",
        language="javascript",
        domain="web_api",
        spdx_license="MIT",
        manifest_files=("package.json",),
        package_identifiers=('"express"',),
        core_concepts=("middleware pipeline", "Router", "central error handlers", "request/response lifecycle"),
        common_pitfalls=("forgetting next(err) in Express 4 async routes", "missing body size limits"),
    ),
    FrameworkSpec(
        slug="fastapi",
        display_name="FastAPI",
        language="python",
        domain="web_api",
        spdx_license="MIT",
        manifest_files=("pyproject.toml", "requirements.txt"),
        package_identifiers=("fastapi",),
        core_concepts=("Pydantic v2 validation", "Depends() DI", "ASGI async/sync routing", "OpenAPI schemas", "StreamingResponse"),
        common_pitfalls=("calling blocking CPU/sync-DB code inside `async def` route handlers", "sharing non-threadsafe session state"),
    ),
    FrameworkSpec(
        slug="django",
        display_name="Django",
        language="python",
        domain="fullstack_web",
        spdx_license="BSD-3-Clause",
        manifest_files=("pyproject.toml", "requirements.txt", "manage.py"),
        package_identifiers=("django", "Django"),
        core_concepts=("ORM QuerySets & migrations", "middleware", "CSRF/auth protections", "select_related / prefetch_related"),
        common_pitfalls=("N+1 lazy ORM relation queries in templates/serializers", "DEBUG=True or wildcard ALLOWED_HOSTS in prod"),
    ),
    FrameworkSpec(
        slug="flask",
        display_name="Flask",
        language="python",
        domain="web_api",
        spdx_license="BSD-3-Clause",
        manifest_files=("pyproject.toml", "requirements.txt"),
        package_identifiers=("flask", "Flask"),
        core_concepts=("application factory create_app()", "Blueprints", "request/app context locals", "WSGI deployment"),
        common_pitfalls=("accessing `request` outside active request context", "running Werkzeug dev server in production"),
    ),
    FrameworkSpec(
        slug="pytorch",
        display_name="PyTorch",
        language="python",
        domain="machine_learning",
        spdx_license="BSD-3-Clause",
        manifest_files=("pyproject.toml", "requirements.txt"),
        package_identifiers=("torch",),
        core_concepts=("autograd", "nn.Module", "SDPA/FlashAttention", "DDP/FSDP", "torch.amp autocast"),
        common_pitfalls=("accumulating computation graph via `.item()` omission on logging", "host-device sync stalls", "forgot model.eval()"),
    ),
    FrameworkSpec(
        slug="axum",
        display_name="Axum / Tokio",
        language="rust",
        domain="systems_web",
        spdx_license="MIT",
        manifest_files=("Cargo.toml",),
        package_identifiers=("axum", "tokio"),
        core_concepts=("typed extractors", "Tower Service middleware", "Arc<AppState>", "Tokio work-stealing runtime"),
        common_pitfalls=("blocking Tokio worker threads without spawn_blocking", "non-Send state across await"),
    ),
    FrameworkSpec(
        slug="gin",
        display_name="Gin",
        language="go",
        domain="web_api",
        spdx_license="MIT",
        manifest_files=("go.mod",),
        package_identifiers=("github.com/gin-gonic/gin",),
        core_concepts=("Context handlers", "middleware chains", "struct tag validation"),
        common_pitfalls=("passing *gin.Context pointer into background goroutines without c.Copy()"),
    ),
    FrameworkSpec(
        slug="spring_boot",
        display_name="Spring Boot",
        language="java",
        domain="enterprise_api",
        spdx_license="Apache-2.0",
        manifest_files=("pom.xml", "build.gradle", "build.gradle.kts"),
        package_identifiers=("spring-boot-starter",),
        core_concepts=("constructor DI", "@Transactional", "Spring Data JPA", "Actuator health endpoints"),
        common_pitfalls=("self-invocation bypassing @Transactional AOP proxy", "FetchType.EAGER N+1 explosions"),
    ),
    FrameworkSpec(
        slug="aspnet_core",
        display_name="ASP.NET Core",
        language="csharp",
        domain="enterprise_api",
        spdx_license="MIT",
        manifest_files=(".csproj", "Program.cs"),
        package_identifiers=("Microsoft.AspNetCore",),
        core_concepts=("Minimal APIs / Controllers", "scoped/singleton/transient DI", "EF Core", "Middleware"),
        common_pitfalls=("captive dependency: injecting Scoped DbContext into Singleton service"),
    ),
)

FRAMEWORKS_BY_SLUG: dict[str, FrameworkSpec] = {f.slug: f for f in SUPPORTED_FRAMEWORKS}


def detect_repository_frameworks(workspace_files: dict[str, str]) -> list[FrameworkSpec]:
    """
    Inspect repository manifest filenames and contents (`{relative_path: text}`)
    to identify active frameworks accurately.
    """
    detected: list[FrameworkSpec] = []
    basenames = {p.rsplit("/", 1)[-1]: content for p, content in workspace_files.items()}

    for fw in SUPPORTED_FRAMEWORKS:
        matched = False
        for mf in fw.manifest_files:
            for path, content in workspace_files.items():
                fname = path.rsplit("/", 1)[-1]
                if fname == mf or fname.endswith(mf):
                    if any(pkg in content for pkg in fw.package_identifiers) or mf.startswith("next.config"):
                        matched = True
                        break
            if matched:
                break
        if not matched and fw.slug == "django" and "manage.py" in basenames:
            matched = True
        if matched:
            detected.append(fw)
    return detected
