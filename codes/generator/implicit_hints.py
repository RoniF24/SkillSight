# codes/generator/implicit_hints.py
# SkillSight - Implicit Anchor Bank (v8)
# -------------------------------------
# This file defines:
# - ALL_SKILLS: the closed skill list (loaded from codes/skills/skills_v1.txt)
# - IMPLICIT_ANCHORS_SOFT/STRONG: per-skill anchor phrases for 0.5 (implicit)
# - LEAK_ALIASES + FORBIDDEN_TERMS: used to reject explicit mentions when label=0.5
#
# Notes:
# - Anchors are SHORT (1-5 words) and unique-ish to the skill.
# - For implicit (0.5) you MUST NOT include the skill name itself.
# - For explicit (1.0) the generator requires the skill name verbatim in the paragraph.

from __future__ import annotations

from pathlib import Path
from typing import Dict, List


# -------------------------------------------------------------------
# Single source of truth for the closed skill list:
# data/skills_v1.txt
# -------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]  # repo root
SKILLS_TXT_PATH = ROOT_DIR / "data" / "skills_v1.txt"


def _load_all_skills(path: Path) -> List[str]:
    """
    Load the closed skill list from skills_v1.txt.
    Rules:
    - one skill per line
    - ignore empty lines and comments starting with '#'
    - keep order (stable vector)
    - de-duplicate (first occurrence wins)
    """
    if not path.exists():
        return []

    seen = set()
    out: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line in seen:
            continue
        seen.add(line)
        out.append(line)
    return out


ALL_SKILLS: List[str] = _load_all_skills(SKILLS_TXT_PATH)


# -------------------------------------------------------------------
# Conservative alias set that counts as explicit mention / leakage
# (keep this list small; it is used for both explicit detection and leak rejection)
# -------------------------------------------------------------------

LEAK_ALIASES: Dict[str, List[str]] = {
    "JavaScript": ["js"],
    "TypeScript": ["ts"],
    "C#": ["c sharp"],
    "C++": ["cpp"],
    "Node.js": ["node", "nodejs"],
    "Express.js": ["express"],
    "GitLab CI": ["gitlab ci", "gitlab"],
    "OpenAPI Specification": ["openapi", "swagger"],
    "ASP.NET Core": ["asp.net", "aspnet", ".net", "dotnet"],
    "Ruby on Rails": ["rails"],
    "Gin (Go)": ["gin"],
    "Actix Web (Rust)": ["actix"],
}


# -------------------------------------------------------------------
# Forbidden terms per skill (also treated as explicit/leak for 0.5)
# IMPORTANT: We DO NOT put generic words here (like "deployment"). Only name-like tokens.
#
# Instead of maintaining a huge hardcoded dict, we build it from:
# - the skill string itself (lowercased)
# - common normalization variants
# - LEAK_ALIASES entries
# plus a small manual override set for tricky skills.
# -------------------------------------------------------------------

def _norm_variants(skill: str) -> List[str]:
    s = skill.strip()
    sl = s.lower()
    variants = {sl}

    # Remove dots/spaces for common mention variants
    variants.add(sl.replace(".", ""))
    variants.add(sl.replace(" ", ""))
    variants.add(sl.replace("-", " "))
    variants.add(sl.replace("-", ""))

    # Common parentheses variants: "Gin (Go)" -> "gin", "gin go"
    if "(" in sl and ")" in sl:
        base = sl.replace("(", " ").replace(")", " ")
        base = " ".join(base.split())
        variants.add(base)

    return [v for v in variants if v]


# Manual extra forbidden tokens for “special formatting” skills
_MANUAL_FORBIDDEN_EXTRAS: Dict[str, List[str]] = {
    "OAuth 2.0": ["oauth 2.0", "oauth2", "oauth 2", "oauth2.0"],
    "OpenAPI Specification": ["openapi specification", "openapi", "swagger"],
    "Event-Driven Architecture": ["event-driven architecture", "event driven architecture", "eda"],
    "End-to-End Testing": ["end-to-end testing", "end to end testing", "e2e testing", "e2e"],
    "ASP.NET Core": ["asp.net core", "asp net core", "aspnet core", ".net", "dotnet", "asp.net", "aspnet"],
    "C++": ["c++", "cplusplus", "cpp", "c plus plus"],
    "C#": ["c#", "c sharp", "csharp"],
    "Node.js": ["node.js", "node js", "nodejs", "node"],
    "Express.js": ["express.js", "express js", "express"],
    "Ruby on Rails": ["ruby on rails", "rails"],
    "Gin (Go)": ["gin (go)", "gin go", "gin"],
    "Actix Web (Rust)": ["actix web (rust)", "actix web rust", "actix"],
    "REST API Design": ["rest api design"],
    "gRPC": ["grpc"],
    "BigQuery": ["big query", "google bigquery"],
    "Kubernetes": ["k8s"],
}


def _build_forbidden_terms(all_skills: List[str], leak_aliases: Dict[str, List[str]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for skill in all_skills:
        terms: List[str] = []
        # base variants
        terms.extend(_norm_variants(skill))

        # alias variants
        for a in leak_aliases.get(skill, []):
            al = a.strip().lower()
            if al:
                terms.append(al)

        # manual extras
        for t in _MANUAL_FORBIDDEN_EXTRAS.get(skill, []):
            tl = t.strip().lower()
            if tl:
                terms.append(tl)

        # de-dup keep order
        seen = set()
        uniq: List[str] = []
        for t in terms:
            if t in seen:
                continue
            seen.add(t)
            uniq.append(t)

        out[skill] = uniq
    return out


FORBIDDEN_TERMS: Dict[str, List[str]] = _build_forbidden_terms(ALL_SKILLS, LEAK_ALIASES)


# -------------------------------------------------------------------
# Soft anchors: default anchors used for implicit requirements
# NOTE: This remains your “anchor bank”. More variety = less static dataset.
# -------------------------------------------------------------------

IMPLICIT_ANCHORS_SOFT: Dict[str, List[str]] = {
    "Python": ["pip requirements", "virtual environment", "pytest fixtures", "asyncio event loop", "type hints"],
    "Java": ["JVM tuning", "garbage collector", "bytecode profiling", "Spring DI patterns", "Maven build"],
    "C": ["pointer arithmetic", "manual memory management", "segmentation faults", "header files", "malloc/free"],
    "C++": ["RAII patterns", "template metaprogramming", "smart pointers", "move semantics", "STL containers"],
    "C#": ["LINQ queries", "async/await", "dependency injection container", "NuGet packages", "CLR profiling"],
    "Go": ["goroutines", "channels and select", "static binaries", "gofmt tooling", "context cancellation"],
    "Rust": ["borrow checker", "lifetimes management", "cargo workspace", "ownership rules", "zero-cost abstractions"],
    "JavaScript": ["event loop callbacks", "npm scripts", "DOM manipulation", "promise chains", "bundler config"],
    "TypeScript": ["compile-time types", "interface definitions", "tsconfig settings", "type narrowing", "generics usage"],
    "Kotlin": ["Gradle modules", "null-safety semantics", "coroutines", "Android Studio build", "Jetpack Compose"],
    "Swift": ["Xcode project", "SwiftUI screens", "UIKit views", "CocoaPods setup", "App Store release"],
    "PHP": ["composer packages", "server-side templating", "request lifecycle", "opcache tuning", "session handling"],
    "Ruby": ["gems bundle", "rack middleware", "metaprogramming hooks", "RSpec specs", "bundler config"],
    "R": ["data frames", "tidyverse pipelines", "ggplot charts", "RMarkdown reports", "statistical modeling"],
    "Scala": ["immutable collections", "Akka actors", "sbt build", "functional transforms", "JVM interop"],
    "MATLAB": ["matrix operations", "signal processing scripts", "numerical simulation", "toolbox functions", "plot figures"],
    "Julia": ["multiple dispatch", "package environment", "high-performance numerics", "JIT compilation", "dataframes jl"],
    "Bash": ["shell scripts", "cron jobs", "pipe and grep", "sed/awk filters", "exit codes"],
    "PowerShell": ["cmdlets pipeline", "Windows automation", "module import", "PSObject handling", "execution policy"],
    "Lua": ["embedded scripting", "game logic scripts", "coroutines", "table structures", "runtime hooks"],
    "Elixir": ["OTP supervision", "GenServer patterns", "message passing", "BEAM runtime", "fault tolerance"],
    "Perl": ["regex heavy parsing", "CPAN modules", "text processing", "one-liners", "legacy scripts"],
    "Haskell": ["pure functions", "monads usage", "typeclasses", "lazy evaluation", "stack/cabal build"],
    "SQL": ["joins and aggregations", "window functions", "CTEs", "query optimization", "stored procedures"],
    "PostgreSQL": ["EXPLAIN ANALYZE", "VACUUM routine", "JSONB fields", "btree indexes", "connection pooling", "tuning indexes on large relational tables", "optimizing slow queries in a relational database"],
    "MySQL": ["InnoDB engine", "slow query log", "replication lag", "binlog rotation", "index hints"],
    "SQLite": ["single-file database", "WAL mode", "embedded storage", "pragma settings", "local caching"],
    "MongoDB": ["document collections", "aggregation pipeline", "replica set", "sharded cluster", "BSON documents"],
    "Redis": ["in-memory key store", "TTL eviction", "pub/sub channels", "sorted sets", "cache invalidation"],
    "Elasticsearch": ["inverted index", "query DSL", "shard allocation", "index mappings", "full-text search"],
    "Cassandra": ["wide-column store", "token ring", "eventual consistency", "partition keys", "tunable consistency"],
    "Neo4j": ["graph traversals", "Cypher queries", "node relationships", "property graph", "path queries"],
    "InfluxDB": ["time-series metrics", "retention policies", "measurement tags", "downsampling", "line protocol"],
    "Snowflake": ["virtual warehouses", "time travel", "zero copy clone", "semi-structured ingestion", "warehouse scaling"],
    "BigQuery": ["columnar warehouse", "partitioned tables", "slot usage", "federated queries", "cost controls"],
    "Redshift": ["cluster nodes", "sort keys", "distribution styles", "vacuum analyze", "spectrum queries"],
    "Databricks": ["notebook workflows", "delta tables", "spark clusters", "job runs", "MLflow tracking"],
    "Apache Spark": ["RDD transforms", "spark executors", "shuffle tuning", "structured streaming", "partition pruning"],
    "Apache Kafka": ["topic partitions", "consumer groups", "offset commits", "at-least-once delivery", "broker retention"],
    "Apache Flink": ["watermarks", "checkpointing", "stateful streaming", "event-time windows", "exactly-once"],
    "Apache Airflow": ["DAG scheduling", "operators and hooks", "task retries", "backfill runs", "XCom usage"],
    "dbt": ["model refs", "incremental models", "tests and seeds", "documentation site", "snapshots"],
    "Delta Lake": ["ACID tables", "merge into upserts", "time travel", "schema enforcement", "lakehouse pattern"],
    "Parquet": ["columnar file format", "predicate pushdown", "row groups", "compression codecs", "schema evolution"],
    "Avro": ["schema registry", "binary encoding", "schema evolution", "backward compatible schema", "compact serialization"],
    "Data Warehousing": ["star schema", "ETL pipelines", "fact tables", "dim tables", "warehouse loads"],
    "Data Modeling": ["entity relationships", "normalization rules", "schema design", "access patterns", "constraints"],
    "Dimensional Modeling": ["slowly changing dimensions", "surrogate keys", "conformed dimensions", "grain definition", "snowflake schema"],
    "Node.js": ["non-blocking IO", "npm packages", "express-style middleware", "promise based handlers", "worker threads"],
    "Django": ["querysets", "ORM migrations", "admin interface", "middleware chain", "CSRF protection"],
    "Flask": ["blueprints routing", "WSGI app", "jinja templates", "request context", "gunicorn workers"],
    "FastAPI": ["pydantic models", "async endpoints", "dependency injection", "OpenAPI auto docs", "type-driven validation"],
    "Spring Boot": ["auto configuration", "starter dependencies", "bean lifecycle", "MVC controllers", "application properties"],
    "ASP.NET Core": ["middleware pipeline", "Kestrel server", "attribute routing", "host builder", "DI container"],
    "Express.js": ["route handlers", "middleware stack", "req/res objects", "npm packages", "error middleware"],
    "NestJS": ["decorators controllers", "providers injection", "module boundaries", "guards interceptors", "pipes validation"],
    "Laravel": ["artisan commands", "eloquent models", "migration rollback", "route middleware", "service container"],
    "Ruby on Rails": ["active record", "rails migrations", "controller actions", "asset pipeline", "rails console"],
    "Gin (Go)": ["router groups", "context handlers", "middleware chain", "JSON binding", "http handlers"],
    "Actix Web (Rust)": ["actor model", "extractors", "middleware pipeline", "async handlers", "route scopes"],
    "REST API Design": ["versioned endpoints", "pagination parameters", "resource paths", "idempotent operations", "error envelopes"],
    "GraphQL": ["queries and mutations", "resolver functions", "schema types", "field selection", "n plus one"],
    "gRPC": ["service stubs", "unary calls", "streaming RPC", "IDL definitions", "deadline propagation"],
    "OAuth 2.0": ["authorization code flow", "refresh tokens", "scopes consent", "client secrets", "redirect URI"],
    "JWT": ["bearer token", "claims based auth", "token signing", "issuer validation", "audience checks"],
    "OpenAPI Specification": ["spec-first workflow", "schema-driven contract", "generated client", "component schemas", "endpoint definitions"],
    "Microservices": ["owning data per service", "bounded context", "independent deploys", "API gateway", "service boundaries"],
    "Event-Driven Architecture": ["event consumers", "outbox pattern", "event schema", "idempotent handlers", "async messaging"],
    "RabbitMQ": ["message acknowledgements", "exchange bindings", "dead letter queue", "routing keys", "prefetch count"],
    "NATS": ["pub/sub subjects", "request reply messaging", "lightweight message bus", "durable subscriptions", "stream consumers"],
    "Caching": ["cache invalidation", "write-through cache", "hot key mitigation", "cache stampede", "TTL eviction"],
    "Rate Limiting": ["HTTP 429", "token bucket", "sliding window", "per API key quota", "burst limits"],
    "Docker": ["container images", "Dockerfile builds", "multi-stage build", "image registry", "container runtime"],
    "Kubernetes": ["liveness probe", "readiness probe", "namespace isolation", "resource requests", "pods rescheduled"],
    "Helm": ["values overrides", "templated manifests", "release history", "rollback support", "package templates"],
    "Terraform": ["state file", "plan and apply", "modules reuse", "provider config", "drift detection", "codifying infrastructure changes as code reviews", "updating shared infrastructure modules across environments"],
    "Ansible": ["playbooks runs", "inventory hosts", "idempotent tasks", "role definitions", "yaml automation"],
    "Jenkins": ["pipeline stages", "agent nodes", "build artifacts", "job triggers", "groovy scripts"],
    "GitHub Actions": ["workflow runs", "runner jobs", "matrix builds", "secrets in CI", "actions marketplace"],
    "GitLab CI": ["pipeline yaml", "stages jobs", "runners", "artifacts caching", "merge request pipelines"],
    "Argo CD": ["gitops sync", "desired state", "auto sync", "app of apps", "drift reconciliation"],
    "CircleCI": ["orbs config", "workflows jobs", "caching steps", "docker executor", "config yaml"],
    "Pulumi": ["infrastructure as code", "state backend", "typed resources", "preview updates", "stack config"],
    "Linux": ["systemd services", "file permissions", "package manager", "shell tooling", "proc filesystem"],
    "Nginx": ["reverse proxy", "upstream blocks", "server directives", "TLS termination", "rate limiting zones"],
    "Prometheus": ["time series scraping", "alert rules", "promql queries", "exporter metrics", "label dimensions"],
    "Grafana": ["dashboard panels", "alert notifications", "data sources", "templated variables", "time range"],
    "OpenTelemetry": ["trace context", "span attributes", "OTLP exporter", "metrics instrumentation", "log correlation"],
    "Jaeger": ["distributed traces UI", "trace sampling", "collector agent", "trace search", "span timelines"],
    "ELK Stack": ["log ingestion", "index patterns", "kibana dashboards", "logstash pipelines", "search queries"],
    "Vault": ["secrets engine", "dynamic credentials", "lease renewal", "encryption keys", "policy based access"],
    "AWS": ["IAM roles", "S3 buckets", "VPC networking", "cloudwatch alarms", "autoscaling groups"],
    "Azure": ["resource groups", "managed identities", "ARM templates", "application gateway", "monitor alerts"],
    "Google Cloud": ["service accounts", "cloud storage buckets", "VPC firewall rules", "cloud logging", "managed SQL"],
    "Threat Modeling": ["attack surface", "STRIDE analysis", "abuse cases", "risk scoring", "trust boundaries"],
    "Penetration Testing": ["exploit attempts", "payload crafting", "privilege escalation", "report findings", "proof of concept"],
    "Secure Code Review": ["review checklist", "taint analysis", "security diff review", "dangerous sinks", "patch recommendations"],
    "OWASP Top 10": ["injection risks", "broken auth", "security misconfig", "XSS mitigation", "CSRF issues", "reviewing endpoints for common web app vulnerabilities", "hardening authentication and access control paths"],
    "SAST": ["static scans", "source analysis", "rule sets", "security findings", "CI scan gate"],
    "DAST": ["dynamic scans", "scanner profiles", "crawl and audit", "staging scan", "vuln alerts"],
    "Vulnerability Scanning": ["scan reports", "CVE triage", "severity thresholds", "remediation cycle", "baseline scans"],
    "CVE Analysis": ["CVE advisories", "CVSS score", "patch availability", "exploitability", "vendor bulletin"],
    "Incident Response": ["incident triage", "containment steps", "postmortem writeup", "pager rotation", "runbook execution"],
    "SIEM": ["log correlation", "security alerts", "centralized logging", "rule tuning", "event aggregation"],
    "Splunk": ["SPL queries", "saved searches", "index-time parsing", "alert correlation", "dashboard panels"],
    "Nmap": ["port scanning", "SYN scan", "service version detection", "OS fingerprinting", "CIDR sweep"],
    "Wireshark": ["pcap capture", "packet dissector", "TCP handshake analysis", "filter expressions", "protocol decoding"],
    "Burp Suite": ["intercepting proxy", "request repeater", "intruder payloads", "scanner findings", "parameter tampering"],
    "Metasploit": ["exploit modules", "payload handlers", "meterpreter session", "auxiliary scanners", "post exploitation"],
    "Network Security": ["firewall rules", "network segmentation", "ingress egress policies", "WAF tuning", "zero trust"],
    "Identity and Access Management": ["least privilege", "role based access", "identity federation", "access reviews", "privileged accounts"],
    "Multi-Factor Authentication": ["one-time codes", "authenticator app", "push approval", "step-up auth", "backup codes"],
    "Encryption": ["at-rest encryption", "key rotation", "envelope encryption", "cipher suites", "encrypted backups"],
    "TLS": ["certificate rotation", "encryption in transit", "mTLS handshake", "certificate chain", "cipher suites"],
    "PKI": ["certificate authority", "certificate issuance", "CRL OCSP", "trust chain", "key pairs"],
    "Key Management": ["HSM usage", "key rotation", "access policies", "secret rotation", "key escrow"],
    "Test Planning": ["test strategy", "scope coverage", "risk based testing", "release criteria", "test schedule"],
    "Test Case Design": ["test scenarios", "boundary cases", "equivalence classes", "negative tests", "traceability"],
    "Manual Testing": ["exploratory testing", "repro steps", "bug reports", "smoke checks", "UI flows"],
    "Automated Testing": ["test suites", "CI test runs", "flaky test fixes", "test harness", "mocking"],
    "Unit Testing": ["isolated tests", "mock dependencies", "assertions", "test fixtures", "arrange act assert"],
    "Integration Testing": ["service integration", "test containers", "database fixtures", "contract checks", "environment setup"],
    "End-to-End Testing": ["full user flow", "browser automation", "staging environment", "data seeding", "cross service validation"],
    "Regression Testing": ["release regression", "baseline suite", "known issues checks", "retest bugs", "change impact"],
    "Smoke Testing": ["sanity checks", "build verification", "critical path tests", "quick suite", "post deploy checks"],
    "Contract Testing": ["consumer driven contracts", "provider verification", "contract broker", "schema compatibility", "breaking change detection"],
    "API Testing": ["endpoint tests", "request assertions", "status code validation", "auth headers", "response schema checks"],
    "UI Testing": ["browser tests", "page objects", "visual checks", "DOM assertions", "user journey"],
    "Performance Testing": ["throughput metrics", "response time SLA", "profiling bottlenecks", "load profiles", "capacity testing"],
    "Load Testing": ["ramp-up users", "steady state load", "request rate", "think time", "traffic simulation"],
    "Stress Testing": ["breaking point", "resource saturation", "failure modes", "spike traffic", "graceful degradation"],
    "Selenium": ["webdriver sessions", "page object pattern", "locator strategies", "implicit waits", "browser grid"],
    "Playwright": ["browser contexts", "trace viewer", "auto waiting", "network mocking", "cross browser runs", "visual regression checks on critical user flows", "headless browser tests integrated into CI pipelines"],
    "Cypress": ["component tests", "time travel debugging", "fixtures stubs", "DOM commands", "headless runs"],
    "Postman": ["collections", "collection runner", "environment variables", "pre-request scripts", "saved requests"],
    "JMeter": ["thread groups", "samplers and listeners", "ramp-up schedule", "jtl results", "load generator"],
}


# -------------------------------------------------------------------
# Strong anchors: used after several retries to increase uniqueness/technicality.
# These should be more technical / less common phrasing, still short.
# -------------------------------------------------------------------

IMPLICIT_ANCHORS_STRONG_OVERRIDES: Dict[str, List[str]] = {
    "Kubernetes": ["pod disruption budget", "RBAC rolebinding", "taints and tolerations", "liveness/readiness probes", "cluster autoscaler"],
    "Helm": ["values.yaml overrides", "chart repository", "templated manifests", "release rollback", "semantic chart version"],
    "Terraform": ["remote state locking", "state drift", "provider version pinning", "plan output diff", "workspace separation"],
    "BigQuery": ["slot reservations", "partition pruning", "bytes billed", "federated queries"],
    "Ansible": ["handlers notified", "facts gathering", "vars precedence", "become sudo", "idempotent playbooks"],
    "Caching": ["TTL eviction", "stale-while-revalidate", "request coalescing", "hot key mitigation", "dogpile effect"],
    "Rate Limiting": ["token bucket", "leaky bucket", "HTTP 429", "quota window", "burst budget"],
    "JWT": ["kid header", "issuer validation", "bearer token", "refresh token rotation", "token signing key"],
    "REST API Design": ["versioned endpoints", "idempotent operations", "status code assertions", "pagination parameters", "error envelopes"],
    "OpenAPI Specification": ["schema-driven contract", "generated client stubs", "component schemas", "request/response examples", "contract compliance"],
    "Microservices": ["bounded context", "service boundaries", "independent deploys", "backward compatible schema", "distributed tracing"],
    "Apache Kafka": ["consumer group rebalancing", "offset commits", "topic partitioning", "exactly-once semantics", "dead letter queue"],
    "Apache Flink": ["stateful streaming", "checkpointing", "watermarks", "event-time processing", "backpressure handling"],
    "Avro": ["schema registry", "schema evolution", "backward compatible schema", "binary serialization", "record union types"],
    "PostgreSQL": ["EXPLAIN ANALYZE", "btree indexes", "VACUUM routine", "transaction isolation", "connection pooling"],
    "MySQL": ["InnoDB engine", "row-level locking", "replication lag", "binlog retention", "query optimizer"],
    "Redis": ["pub/sub channels", "lua scripts", "connection pooling", "TTL eviction", "hot key mitigation"],
    "Prometheus": ["PromQL queries", "recording rules", "alertmanager routing", "scrape interval", "histogram buckets"],
    "SIEM": ["saved searches", "correlation rules", "log enrichment", "alert triage", "noise reduction"],
    "Incident Response": ["post-incident review", "runbook escalation", "MTTR reduction", "on-call rotation", "blast radius assessment"],
}

IMPLICIT_ANCHORS_STRONG: Dict[str, List[str]] = {
    **IMPLICIT_ANCHORS_SOFT,
    **IMPLICIT_ANCHORS_STRONG_OVERRIDES,
}
