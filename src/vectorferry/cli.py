"""`vf`: the vectorferry command line."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

from .connectors import open_sink, open_source
from .connectors.base import Source
from .core.model import CollectionSpec
from .engine.copy import copy_collection
from .engine.events import Event
from .engine.state import State
from .mapping import index_plan, table_name
from .snapshot import export_snapshot
from .verify import VerifyReport, verify_collection

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Move and sync data between vector databases, and prove it arrived intact.",
)
console = Console()

From = Annotated[
    str, typer.Option("--from", help="Source: chroma://host:port or a snapshot folder")
]
To = Annotated[str, typer.Option("--to", help="Target: postgresql://user:pass@host:port/db")]
Collections = Annotated[
    list[str] | None,
    typer.Option("--collection", "-c", help="Only these collections (repeatable). Default: all."),
]
Tenant = Annotated[str, typer.Option(help="Chroma tenant (0.6+/1.x)")]
Database = Annotated[str, typer.Option(help="Chroma database (0.6+/1.x)")]
PgSchema = Annotated[str, typer.Option(help="Postgres schema for the tables")]
Prefix = Annotated[str, typer.Option(help="Table name prefix")]
TableMap = Annotated[
    list[str] | None,
    typer.Option("--table-map", help="collection=table to name a table yourself (repeatable)"),
]


def _table_map(items: list[str] | None) -> dict[str, str]:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise typer.BadParameter(f"--table-map expects collection=table, got {item!r}")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def _select(source: Source, only: list[str] | None) -> list[CollectionSpec]:
    specs = source.list_collections()
    if only:
        names = {s.name for s in specs}
        unknown = [c for c in only if c not in names]
        if unknown:
            raise typer.BadParameter(f"no such collection(s): {', '.join(unknown)}")
        specs = [s for s in specs if s.name in only]
    return specs


@contextmanager
def _progress():
    with Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as p:
        tasks: dict[str, int] = {}

        def emit(e: Event) -> None:
            if e.collection not in tasks:
                tasks[e.collection] = p.add_task(e.collection, total=e.total)
            t = tasks[e.collection]
            p.update(t, completed=e.done, total=e.total)
            if e.kind in ("retry", "warning", "error") and e.message:
                p.console.print(f"[yellow]{e.collection}: {e.message}")

        yield emit


@app.command()
def inspect(
    src: From,
    tenant: Tenant = "default_tenant",
    database: Database = "default_database",
    table_prefix: Prefix = "vf_",
) -> None:
    """List the source's collections and how each would land in pgvector."""
    source = open_source(src, tenant=tenant, database=database)
    t = Table(
        title=f"{source.uri}" + (f"  (Chroma API {source.api})" if hasattr(source, "api") else "")
    )
    for col in ("collection", "records", "dims", "metric", "→ table", "→ index", "notes"):
        t.add_column(col)
    for s in source.list_collections():
        plan = index_plan(s)
        t.add_row(
            s.name,
            str(s.count),
            str(s.dim),
            s.metric.value,
            table_name(s.name, table_prefix),
            plan.opclass or "none",
            plan.warning or "",
        )
    console.print(t)


@app.command()
def copy(
    src: From,
    dst: To,
    collection: Collections = None,
    batch_size: Annotated[int, typer.Option(min=1)] = 500,
    state: Annotated[Path, typer.Option(help="Resume state file")] = Path(".vectorferry/state.db"),
    restart: Annotated[bool, typer.Option(help="Ignore saved progress and start over")] = False,
    no_index: Annotated[bool, typer.Option(help="Don't build the HNSW index")] = False,
    verify: Annotated[bool, typer.Option(help="Run `vf verify` after copying")] = True,
    tenant: Tenant = "default_tenant",
    database: Database = "default_database",
    pg_schema: PgSchema = "public",
    table_prefix: Prefix = "vf_",
    table_map: TableMap = None,
) -> None:
    """Copy collections from the source into pgvector. Safe to re-run: it resumes."""
    source = open_source(src, tenant=tenant, database=database)
    sink = open_sink(
        dst,
        schema=pg_schema,
        table_prefix=table_prefix,
        table_map=_table_map(table_map),
        no_index=no_index,
    )
    st = State(state)
    specs = _select(source, collection)
    for s in specs:
        if w := index_plan(s, no_index=no_index).warning:
            console.print(f"[yellow]{s.name}: {w}")
    with _progress() as emit:
        for s in specs:
            copy_collection(source, sink, s, st, batch_size=batch_size, restart=restart, emit=emit)
    console.print(f"[green]copied {len(specs)} collection(s) into {sink.uri}")
    if verify:
        _verify(
            source,
            sink,
            specs,
            batch_size=batch_size,
            sample=100,
            k=10,
            min_recall=0.95,
            checksums=True,
            report=None,
        )


def _verify(source, sink, specs, *, batch_size, sample, k, min_recall, checksums, report) -> None:
    reports: list[VerifyReport] = []
    with _progress() as emit:
        for s in specs:
            reports.append(
                verify_collection(
                    source,
                    sink,
                    s,
                    batch_size=batch_size,
                    sample=sample,
                    k=k,
                    min_recall=min_recall,
                    checksums=checksums,
                    emit=emit,
                )
            )
    t = Table(
        title="verify",
        caption="source agreement = how often the source's own ANN "
        "found the true neighbours (info only, not a migration check)",
    )
    for col in (
        "collection",
        "counts",
        "checksums",
        "similarity measure",
        f"index recall@{k}",
        "source agreement",
        "result",
    ):
        t.add_column(col)
    for r in reports:
        counts = (
            f"{r.source_count} = {r.sink_count}"
            if r.counts_ok
            else f"[red]{r.source_count} ≠ {r.sink_count}"
        )
        sums = (
            "skipped"
            if not checksums
            else (
                f"{r.checked} ok"
                if r.checksums_ok
                else f"[red]{r.missing_total} missing, {r.mismatched_total} different"
            )
        )
        if r.index_recall is None:
            measure = rec = agree = "—"
        else:
            measure = (
                f"ok ({r.queries} queries)"
                if r.metric_ok
                else f"[red]{r.metric_violations}/{r.queries} queries disagree"
            )
            rec = f"{r.index_recall:.3f}" if r.recall_ok else f"[red]{r.index_recall:.3f}"
            agree = f"[dim]{r.source_agreement:.3f}"
        t.add_row(
            r.collection, counts, sums, measure, rec, agree, "[green]OK" if r.ok else "[red]FAILED"
        )
    console.print(t)
    if report:
        Path(report).write_text(
            json.dumps(
                [
                    {
                        "collection": r.collection,
                        "ok": r.ok,
                        "source_count": r.source_count,
                        "sink_count": r.sink_count,
                        "checked": r.checked,
                        "missing_total": r.missing_total,
                        "mismatched_total": r.mismatched_total,
                        "missing": r.missing,
                        "mismatched": r.mismatched,
                        "queries": r.queries,
                        "metric_violations": r.metric_violations,
                        "index_recall": r.index_recall,
                        "min_recall": r.min_recall,
                        "source_agreement": r.source_agreement,
                    }
                    for r in reports
                ],
                indent=2,
            )
        )
        console.print(f"report written to {report}")
    if not all(r.ok for r in reports):
        raise typer.Exit(1)


@app.command()
def verify(
    src: From,
    dst: To,
    collection: Collections = None,
    sample: Annotated[int, typer.Option(help="Number of sample searches")] = 100,
    k: Annotated[int, typer.Option(help="Neighbours compared per search")] = 10,
    min_recall: Annotated[
        float, typer.Option(help="Fail if the target's index recall@k (vs exact) is below this")
    ] = 0.95,
    checksums: Annotated[bool, typer.Option(help="Compare every record's checksum")] = True,
    batch_size: Annotated[int, typer.Option(min=1)] = 500,
    report: Annotated[Path | None, typer.Option(help="Also write a JSON report here")] = None,
    tenant: Tenant = "default_tenant",
    database: Database = "default_database",
    pg_schema: PgSchema = "public",
    table_prefix: Prefix = "vf_",
    table_map: TableMap = None,
) -> None:
    """Prove the target matches the source: counts, checksums, search results. Exit 1 if not."""
    source = open_source(src, tenant=tenant, database=database)
    sink = open_sink(
        dst, schema=pg_schema, table_prefix=table_prefix, table_map=_table_map(table_map)
    )
    _verify(
        source,
        sink,
        _select(source, collection),
        batch_size=batch_size,
        sample=sample,
        k=k,
        min_recall=min_recall,
        checksums=checksums,
        report=report,
    )


@app.command()
def export(
    src: From,
    out: Annotated[Path, typer.Option("--out", help="Empty folder to write the snapshot to")],
    collection: Collections = None,
    batch_size: Annotated[int, typer.Option(min=1)] = 500,
    tenant: Tenant = "default_tenant",
    database: Database = "default_database",
) -> None:
    """Write collections to a portable Parquet snapshot (a backup you can keep anywhere)."""
    source = open_source(src, tenant=tenant, database=database)
    with _progress() as emit:
        path = export_snapshot(
            source, out, _select(source, collection), batch_size=batch_size, emit=emit
        )
    console.print(f"[green]snapshot written to {path}")


@app.command("import")
def import_(
    snapshot: Annotated[Path, typer.Option("--from", help="Snapshot folder from `vf export`")],
    dst: To,
    collection: Collections = None,
    batch_size: Annotated[int, typer.Option(min=1)] = 500,
    state: Annotated[Path, typer.Option(help="Resume state file")] = Path(".vectorferry/state.db"),
    restart: bool = False,
    no_index: bool = False,
    pg_schema: PgSchema = "public",
    table_prefix: Prefix = "vf_",
    table_map: TableMap = None,
) -> None:
    """Load a snapshot into pgvector (same as `copy --from <snapshot>`)."""
    copy(
        src=f"snapshot://{snapshot.resolve()}",
        dst=dst,
        collection=collection,
        batch_size=batch_size,
        state=state,
        restart=restart,
        no_index=no_index,
        verify=True,
        tenant="default_tenant",
        database="default_database",
        pg_schema=pg_schema,
        table_prefix=table_prefix,
        table_map=table_map,
    )


if __name__ == "__main__":
    app()
