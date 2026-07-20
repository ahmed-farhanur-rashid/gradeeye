"""
Rich-based training progress display.

Replaces raw tqdm output with structured, color-coded panels showing
phase transitions, batch progress, epoch metrics, and best-model tracking.
"""
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

console = Console()


def make_batch_progress() -> Progress:
    """Create a Progress bar for iterating over batches within an epoch."""
    return Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=None, complete_style="cyan", finished_style="green"),
        TextColumn("{task.percentage:>5.1f}%", style="bold"),
        MofNCompleteColumn(),
        TextColumn("·"),
        TimeElapsedColumn(),
        TextColumn("·", style="dim"),
        TextColumn("eta", style="dim"),
        TimeRemainingColumn(),
        TextColumn("·", style="dim"),
        TextColumn("{task.fields[status]}", style="yellow"),
        console=console,
        transient=True,
    )


def print_header(run_name: str, device: str, loss_type: str):
    """Print once at the very start of training."""
    title = Text()
    title.append("⚕  GRADEEYE", style="bold magenta")
    title.append("  ·  ", style="dim")
    title.append(run_name, style="bold cyan")
    title.append("  ·  ", style="dim")
    title.append(loss_type.upper(), style="bold yellow")
    title.append("  ·  ", style="dim")
    title.append(device.upper(), style="bold green")
    console.print()
    console.print(Panel(title, box=box.DOUBLE_EDGE, border_style="magenta",
                        padding=(0, 2)))


def print_phase_start(phase_name: str, phase_idx: int, total_phases: int,
                      num_epochs: int, batch_size: int, freeze: bool):
    """Print a phase transition banner."""
    console.print()
    info = Text()
    info.append(f"Phase {phase_idx}/{total_phases}", style="bold white")
    info.append("  ·  ", style="dim")
    info.append(phase_name, style="bold cyan")
    info.append("  ·  ", style="dim")
    info.append(f"{num_epochs} epochs", style="white")
    info.append("  ·  ", style="dim")
    info.append(f"bs={batch_size}", style="white")
    if freeze:
        info.append("  ·  ", style="dim")
        info.append("backbone frozen", style="bold yellow")
    console.print(Panel(info, box=box.HEAVY, border_style="blue", padding=(0, 2)))


def print_epoch_summary(epoch: int, num_epochs: int,
                        train_loss: float, train_acc: float,
                        val_loss: float, val_acc: float, val_qwk: float,
                        lr: float, best_qwk: float, is_new_best: bool,
                        best_path: str | None = None):
    """Print a compact metrics table after each epoch."""
    table = Table(
        box=box.SIMPLE_HEAVY, show_header=True, padding=(0, 1),
        title=f"Epoch {epoch + 1}/{num_epochs}",
        title_style="bold", min_width=44,
    )
    table.add_column("", style="bold", width=10)
    table.add_column("Train", justify="right", width=10)
    table.add_column("Val", justify="right", width=10)

    table.add_row("Loss", f"{train_loss:.4f}", f"{val_loss:.4f}")
    table.add_row("Accuracy", f"{train_acc:.2%}", f"{val_acc:.2%}")

    qwk_str = f"{val_qwk:.4f}"
    if is_new_best:
        qwk_str += " ★"
        qwk_style = "bold green"
    elif val_qwk < best_qwk * 0.95:
        qwk_style = "red"
    else:
        qwk_style = ""
    table.add_row("QWK", "—", Text(qwk_str, style=qwk_style))
    table.add_row("LR", f"{lr:.2e}", "—", style="dim")

    console.print(table)

    if is_new_best and best_path:
        console.print(f"  [bold green]★ New best → {best_path}[/bold green]")


def print_training_complete(best_qwk: float, run_name: str):
    """Print final summary."""
    console.print()
    result = Text()
    result.append("✓ Training complete", style="bold green")
    result.append("  ·  ", style="dim")
    result.append(run_name, style="bold cyan")
    result.append("  ·  ", style="dim")
    result.append(f"Best QWK: {best_qwk:.4f}", style="bold magenta")
    console.print(Panel(result, box=box.DOUBLE_EDGE, border_style="green",
                        padding=(0, 2)))
    console.print()


def log(msg: str, style: str = ""):
    """Print a log message to console (replaces tqdm.write)."""
    console.print(f"  {msg}", style=style)
