"""Main entry point for gguf_sync."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .backends import (
    GPT4AllBackend,
    JanBackend,
    KoboldCppBackend,
    LlamaCppBackend,
    LlamaCppPythonBackend,
    LMStudioBackend,
    LocalAIBackend,
    OllamaBackend,
    TextGenBackend,
    vLLMBackend,
)
from .core.config import ConfigLoader
from .core.constants import (
    DEFAULT_LMSTUDIO_DIR,
    DEFAULT_LOCALAI_DIR,
    DEFAULT_MODELS_DST,
    DEFAULT_MODELS_SRC,
    DEFAULT_SERVICE_NAME,
)
from .core.discovery import BackendDiscovery, create_config_from_discovered
from .core.exceptions import GGUFSyncError
from .core.logging import get_logger, is_verbose, setup_logging
from .core.models import (
    AppConfig,
    GPT4AllConfig,
    JanConfig,
    KoboldCppConfig,
    LlamaCppConfig,
    LlamaCppPythonConfig,
    LMStudioConfig,
    LocalAIConfig,
    OllamaConfig,
    TextGenConfig,
    vLLMConfig,
)
from .core.service import ServiceInstaller
from .core.sync import SyncEngine
from .core.watcher import FileSystemWatcher

if TYPE_CHECKING:
    from .backends.base import Backend

# Rich console for pretty output
console = Console()
app = typer.Typer(
    name="gguf-sync",
    help="Cross-platform model linker for LLM inference engines",
    rich_markup_mode="rich",
)

logger = get_logger(__name__)


def version_callback(value: bool) -> None:
    """Display version information."""
    if value:
        from . import __version__

        console.print(f"gguf-sync version {__version__}")
        raise typer.Exit()


def get_backends(config: AppConfig) -> list[Backend]:
    """Create backend instances from configuration.

    Args:
        config: Application configuration

    Returns:
        List of initialized backends
    """
    backends: list[Backend] = []

    for name, backend_config in config.backends.items():
        if not backend_config.enabled:
            continue

        if isinstance(backend_config, LlamaCppConfig):
            backends.append(LlamaCppBackend(backend_config))
        elif isinstance(backend_config, LocalAIConfig):
            backends.append(LocalAIBackend(backend_config))
        elif isinstance(backend_config, LMStudioConfig):
            backends.append(LMStudioBackend(backend_config))
        elif isinstance(backend_config, OllamaConfig):
            backends.append(OllamaBackend(backend_config))
        elif isinstance(backend_config, TextGenConfig):
            backends.append(TextGenBackend(backend_config))
        elif isinstance(backend_config, GPT4AllConfig):
            backends.append(GPT4AllBackend(backend_config))
        elif isinstance(backend_config, KoboldCppConfig):
            backends.append(KoboldCppBackend(backend_config))
        elif isinstance(backend_config, vLLMConfig):
            backends.append(vLLMBackend(backend_config))
        elif isinstance(backend_config, JanConfig):
            backends.append(JanBackend(backend_config))
        elif isinstance(backend_config, LlamaCppPythonConfig):
            backends.append(LlamaCppPythonBackend(backend_config))
        else:
            logger.warning(f"Unknown backend type for {name}: {type(backend_config).__name__}")

    return backends


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version information",
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
        exists=True,
        dir_okay=False,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-V",
        help="Enable verbose output",
    ),
    json_logs: bool = typer.Option(
        False,
        "--json",
        help="Output logs as JSON",
    ),
) -> None:
    """Link Models - Cross-platform model linker for LLM inference engines."""
    # Setup logging early
    setup_logging(
        verbose=verbose,
        json_format=json_logs,
    )


@app.command()
def sync(
    source: Path | None = typer.Option(
        None,
        "--source",
        "--src",
        "-s",
        help=f"Source directory (default: {DEFAULT_MODELS_SRC})",
    ),
    llama_cpp_dir: Path | None = typer.Option(
        None,
        "--llama-cpp",
        "--llama",
        help=f"llama.cpp output directory (default: {DEFAULT_MODELS_DST})",
    ),
    localai_dir: Path | None = typer.Option(
        None,
        "--localai",
        "-l",
        help=f"LocalAI output directory (default: {DEFAULT_LOCALAI_DIR})",
    ),
    lmstudio_dir: Path | None = typer.Option(
        None,
        "--lmstudio",
        help=f"LM Studio output directory (default: {DEFAULT_LMSTUDIO_DIR})",
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
        exists=True,
    ),
    no_llama_cpp: bool = typer.Option(
        False,
        "--no-llama-cpp",
        help="Disable llama.cpp backend",
    ),
    no_localai: bool = typer.Option(
        False,
        "--no-localai",
        help="Disable LocalAI backend",
    ),
    no_lmstudio: bool = typer.Option(
        False,
        "--no-lmstudio",
        help="Disable LM Studio backend",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Show what would be done without making changes",
    ),
) -> None:
    """Run a one-time synchronization."""
    try:
        # Load configuration
        loader = ConfigLoader()
        cli_args: dict[str, Any] = {
            "sync": {"dry_run": dry_run},
        }

        if source:
            cli_args["source_dir"] = source

        # Build backend config from CLI args
        backends_config = {}

        if not no_llama_cpp:
            llama_config = {"enabled": True}
            if llama_cpp_dir:
                llama_config["output_dir"] = llama_cpp_dir
            else:
                # Use default if not specified
                llama_config["output_dir"] = Path(DEFAULT_MODELS_DST)
            backends_config["llama_cpp"] = llama_config

        if not no_localai:
            localai_config = {"enabled": True}
            if localai_dir:
                localai_config["output_dir"] = localai_dir
            else:
                # Use default if not specified
                localai_config["output_dir"] = Path(DEFAULT_LOCALAI_DIR)
            backends_config["localai"] = localai_config

        if lmstudio_dir and not no_lmstudio:
            backends_config["lmstudio"] = {
                "enabled": True,
                "output_dir": lmstudio_dir,
            }

        if backends_config:
            cli_args["backends"] = backends_config

        config = loader.load(config_path=config_file, cli_args=cli_args)

        # Create backends
        backends = get_backends(config)

        if not backends:
            console.print("[red]Error: No backends enabled[/red]")
            raise typer.Exit(1)

        # Create and run sync engine
        engine = SyncEngine(config, backends)
        engine.setup()

        console.print(
            Panel.fit(
                f"[bold green]Starting synchronization[/bold green]\n"
                f"Source: [cyan]{config.source_dir}[/cyan]\n"
                f"Backends: [yellow]{', '.join(b.name for b in backends)}[/yellow]"
            )
        )

        results = engine.full_sync()

        # Display results
        table = Table(title="Synchronization Results")
        table.add_column("Backend", style="cyan")
        table.add_column("Linked", justify="right", style="green")
        table.add_column("Updated", justify="right", style="yellow")
        table.add_column("Skipped", justify="right", style="blue")
        table.add_column("Removed", justify="right", style="red")
        table.add_column("Errors", justify="right", style="red")

        for name, result in results.items():
            table.add_row(
                name,
                str(result.linked),
                str(result.updated),
                str(result.skipped),
                str(result.removed),
                str(len(result.errors)) if result.errors else "0",
            )

        console.print(table)

        # Display skip reasons if any
        for name, result in results.items():
            if result.skip_reasons:
                console.print(f"\n[yellow]{name}:[/yellow] Skipped items:")
                # Group by reason for summary
                reason_counts: dict[str, int] = {}
                for reason in result.skip_reasons:
                    reason_type = reason.get("reason", "unknown")
                    reason_counts[reason_type] = reason_counts.get(reason_type, 0) + 1

                for reason_type, count in sorted(reason_counts.items()):
                    console.print(f"  [blue]{count}[/blue] {reason_type}")

                # Show details in verbose mode
                if is_verbose() and result.skip_reasons:
                    console.print("  [dim]Details:[/dim]")
                    for reason in result.skip_reasons[:20]:  # Limit to first 20
                        item = reason.get("item", "unknown")
                        reason_type = reason.get("reason", "unknown")
                        console.print(f"    [dim]- {item}: {reason_type}[/dim]")
                    if len(result.skip_reasons) > 20:
                        console.print(
                            f"    [dim]... and {len(result.skip_reasons) - 20} more[/dim]"
                        )

        # Check for errors
        has_errors = any(r.errors for r in results.values())
        if has_errors:
            console.print("[yellow]Some errors occurred during synchronization[/yellow]")
            for name, result in results.items():
                for error in result.errors:
                    console.print(f"  [red]{name}:[/red] {error}")

        console.print("[bold green]Synchronization complete![/bold green]")

    except GGUFSyncError as e:
        console.print(f"[red]Error: {e.message}[/red]")
        if e.details:
            console.print(f"[dim]{e.details}[/dim]")
        raise typer.Exit(1)
    except Exception as e:
        logger.exception("Unexpected error")
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def watch(
    source: Path | None = typer.Option(
        None,
        "--source",
        "--src",
        "-s",
        help=f"Source directory (default: {DEFAULT_MODELS_SRC})",
    ),
    llama_cpp_dir: Path | None = typer.Option(
        None,
        "--llama-cpp",
        "--llama",
        help="llama.cpp output directory",
    ),
    localai_dir: Path | None = typer.Option(
        None,
        "--localai",
        "-l",
        help="LocalAI output directory",
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
        exists=True,
    ),
    interval: float = typer.Option(
        2.0,
        "--interval",
        "-i",
        help="Download check interval in seconds",
    ),
    no_initial_sync: bool = typer.Option(
        False,
        "--no-initial-sync",
        help="Skip initial full sync on startup",
    ),
) -> None:
    """Run as a filesystem watcher (continuous monitoring)."""

    async def run_watcher() -> None:
        try:
            # Load configuration
            loader = ConfigLoader()
            cli_args: dict[str, Any] = {
                "watch": {"enabled": True, "check_interval": interval},
            }

            if source:
                cli_args["source_dir"] = source

            config = loader.load(config_path=config_file, cli_args=cli_args)

            # Create backends
            backends = get_backends(config)

            if not backends:
                console.print("[red]Error: No backends enabled[/red]")
                raise typer.Exit(1)

            # Create sync engine
            engine = SyncEngine(config, backends)
            engine.setup()

            # Initial sync (optional)
            console.print(
                Panel.fit(
                    f"[bold green]Starting filesystem watcher[/bold green]\n"
                    f"Source: [cyan]{config.source_dir}[/cyan]\n"
                    f"Backends: [yellow]{', '.join(b.name for b in backends)}[/yellow]\n"
                    f"Press [bold]Ctrl+C[/bold] to stop"
                )
            )

            if not no_initial_sync:
                console.print("[dim]Performing initial synchronization...[/dim]")
                engine.full_sync()
                console.print("[green]Initial sync complete. Watching for changes...[/green]")
            else:
                console.print(
                    "[dim]Skipping initial sync (--no-initial-sync). Watching for changes...[/dim]"
                )

            # Create and run watcher - only watch source directory
            # Watching backend directories causes duplicate events when hardlinks are created
            def on_event(event: Any) -> None:
                try:
                    engine.handle_event(event)
                except Exception as e:
                    logger.error("Error handling event", error=str(e))

            watcher = FileSystemWatcher(
                source_dirs=[config.source_dir],
                callback=on_event,
                check_interval=config.watch.check_interval,
                stable_count=config.watch.stable_count,
            )

            await watcher.run()

        except asyncio.CancelledError:
            console.print("\n[yellow]Watcher stopped[/yellow]")
        except GGUFSyncError as e:
            console.print(f"[red]Error: {e.message}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            logger.exception("Unexpected error")
            console.print(f"[red]Unexpected error: {e}[/red]")
            raise typer.Exit(1)

    try:
        anyio.run(run_watcher)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")


@app.command()
def service(
    action: str = typer.Argument(
        ...,
        help="Action to perform: install, uninstall, start, stop, status",
    ),
    name: str = typer.Option(
        DEFAULT_SERVICE_NAME,
        "--name",
        "-n",
        help="Service name",
    ),
) -> None:
    """Manage the gguf-sync service."""
    installer = ServiceInstaller(service_name=name)

    if action == "install":
        try:
            installer.install()
            console.print(f"[green]Service '{name}' installed successfully[/green]")
            console.print("[dim]Start with: [bold]gguf-sync service start[/bold][/dim]")
        except GGUFSyncError as e:
            console.print(f"[red]Failed to install service: {e.message}[/red]")
            raise typer.Exit(1)

    elif action == "uninstall":
        try:
            installer.uninstall()
            console.print(f"[green]Service '{name}' uninstalled successfully[/green]")
        except GGUFSyncError as e:
            console.print(f"[red]Failed to uninstall service: {e.message}[/red]")
            raise typer.Exit(1)

    elif action == "start":
        try:
            installer.start()
            console.print(f"[green]Service '{name}' started[/green]")
        except Exception as e:
            console.print(f"[red]Failed to start service: {e}[/red]")
            raise typer.Exit(1)

    elif action == "stop":
        try:
            installer.stop()
            console.print(f"[green]Service '{name}' stopped[/green]")
        except Exception as e:
            console.print(f"[red]Failed to stop service: {e}[/red]")
            raise typer.Exit(1)

    elif action == "status":
        status = installer.status()
        if status.get("installed"):
            state = "[green]active[/green]" if status.get("active") else "[yellow]inactive[/yellow]"
            console.print(f"Service '{name}': {state}")
        else:
            console.print(f"Service '{name}': [red]not installed[/red]")

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Valid actions: install, uninstall, start, stop, status")
        raise typer.Exit(1)


@app.command()
def config(
    generate: bool = typer.Option(
        False,
        "--generate",
        "-g",
        help="Generate default configuration file",
    ),
    output: Path = typer.Option(
        Path("gguf_sync.yaml"),
        "--output",
        "-o",
        help="Output path for generated config",
    ),
) -> None:
    """Configuration management."""
    if generate:
        loader = ConfigLoader()
        default_config = loader.generate_default_config()

        with open(output, "w") as f:
            f.write(default_config)

        console.print(f"[green]Default configuration written to: {output}[/green]")
        console.print("[dim]Edit this file and use with: gguf-sync -c {output} <command>[/dim]")
    else:
        # Show current effective configuration
        loader = ConfigLoader()
        cfg = loader.load()

        console.print(Panel.fit("[bold]Current Configuration[/bold]"))
        console.print(f"Source: [cyan]{cfg.source_dir}[/cyan]")
        console.print("\nBackends:")
        for name, backend in cfg.backends.items():
            status = "[green]enabled[/green]" if backend.enabled else "[red]disabled[/red]"
            console.print(f"  {name}: {status} -> [cyan]{backend.output_dir}[/cyan]")


@app.command()
def discover(
    generate_config: bool = typer.Option(
        False,
        "--generate-config",
        "-g",
        help="Generate config file with discovered backends",
    ),
    output: Path = typer.Option(
        Path("gguf_sync.yaml"),
        "--output",
        "-o",
        help="Output path for generated config",
    ),
) -> None:
    """Auto-discover installed LLM inference backends."""
    try:
        discovery = BackendDiscovery()
        backends = discovery.discover_all()

        if not backends:
            console.print("[yellow]No backends discovered[/yellow]")
            return

        # Display discovered backends
        table = Table(title="Discovered Backends")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="blue")
        table.add_column("Install Dir", style="dim")
        table.add_column("Models Dir", style="green")
        table.add_column("Running", style="yellow")
        table.add_column("Port", style="magenta")

        for backend in backends:
            running = "[green]Yes[/green]" if backend.is_running else "[red]No[/red]"
            port = str(backend.port) if backend.port else "-"
            table.add_row(
                backend.name,
                backend.backend_type,
                str(backend.install_dir)[:40],
                str(backend.models_dir)[:40] if backend.models_dir else "-",
                running,
                port,
            )

        console.print(table)

        # Generate config if requested
        if generate_config:
            config_dict = create_config_from_discovered(backends)
            import yaml

            config_yaml = yaml.dump({"backends": config_dict}, default_flow_style=False)

            with open(output, "w") as f:
                f.write(config_yaml)

            console.print(f"\n[green]Configuration written to: {output}[/green]")
            console.print("[dim]Edit this file and use with: gguf-sync -c {output} sync[/dim]")

    except Exception as e:
        logger.exception("Discovery failed")
        console.print(f"[red]Discovery failed: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
