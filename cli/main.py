"""
Interactive command-line interface for time-series anomaly detection.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax

from core.config import get_settings
from core.exceptions import AnomalyDetectionError, FileProcessingError, ConfigurationError
from core.models import AnomalyDetectionRequest, TimeSeriesData
from agents.supervisor import AnomalyDetectionSupervisor
from agents.tools.file_reader import FileReaderTool
from agents.tools.anomaly_detector import AnomalyDetectionTool


# Create CLI app
app = typer.Typer(
    name="anomaly-detector",
    help="Time-Series Anomaly Detection CLI - Multi-Agent AI System",
    rich_markup_mode="rich"
)

# Create console for rich output
console = Console()

# Configure logging for CLI
def setup_logging(debug=False, log_file="logs/app.log"):
    """Setup logging configuration"""
    import os
    from pathlib import Path
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set logging level
    level = logging.DEBUG if debug else logging.INFO
    
    # Configure logging with both file and console handlers
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True  # Override any existing configuration
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()


def print_banner():
    """Print application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                   Time-Series Anomaly Detection System                       ║
║                         Multi-Agent AI Analysis                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    rprint(f"[bold cyan]{banner}[/bold cyan]")


def print_methods_info():
    """Print information about available detection methods."""
    table = Table(title="Available Anomaly Detection Methods")
    table.add_column("Method", style="cyan", no_wrap=True)
    table.add_column("Best For", style="magenta")
    table.add_column("Default Threshold", style="green")
    table.add_column("Description", style="white")
    
    detector = AnomalyDetectionTool()
    methods = ['z-score', 'iqr', 'rolling-iqr', 'dbscan']
    
    for method in methods:
        info = detector.get_method_info(method)
        table.add_row(
            info.get('name', method),
            info.get('best_for', 'General'),
            str(info.get('default_threshold', 'Auto')),
            info.get('description', 'Statistical anomaly detection')
        )
    
    console.print(table)


def format_results(result):
    """Format analysis results for display."""
    if not result.anomaly_result:
        return "No results available"
    
    # Anomaly results table
    results_table = Table(title="Anomaly Detection Results")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="white")
    
    anomaly = result.anomaly_result
    results_table.add_row("Method Used", anomaly.method_used.upper())
    results_table.add_row("Threshold", f"{anomaly.threshold_used}")
    results_table.add_row("Total Points", f"{anomaly.total_points:,}")
    results_table.add_row("Anomalies Found", f"[red]{anomaly.anomaly_count:,}[/red]")
    results_table.add_row("Percentage", f"[red]{anomaly.anomaly_percentage:.2f}%[/red]")
    results_table.add_row("Processing Time", f"{result.processing_time:.2f}s")
    
    console.print(results_table)
    
    # Insights summary
    if result.insights:
        insights_panel = Panel(
            result.insights.summary,
            title="[bold green]Analysis Summary[/bold green]",
            border_style="green"
        )
        console.print(insights_panel)
        
        if result.insights.recommendations:
            rprint("\n[bold yellow]Recommendations:[/bold yellow]")
            for i, rec in enumerate(result.insights.recommendations, 1):
                rprint(f"  {i}. {rec}")
    
    # Visualization info
    if result.visualization:
        viz_info = f"Visualization saved to: [green]{result.visualization.plot_path}[/green]"
        console.print(Panel(viz_info, title="Visualization", border_style="blue"))


@app.command()
def interactive():
    """
    Start interactive mode for anomaly detection analysis.
    """
    print_banner()
    
    try:
        # Validate configuration
        settings = get_settings()
        console.print(f"[green]✓[/green] Configuration loaded successfully")
        console.print(f"[dim]LLM Model: {settings.llm_model}[/dim]\n")
        
    except ConfigurationError as e:
        console.print(f"[red]✗ Configuration Error:[/red] {str(e)}")
        console.print("\n[yellow]Please check your .env file and ensure all required settings are configured.[/yellow]")
        raise typer.Exit(1)
    
    console.print("[bold]Welcome to the Interactive Anomaly Detection System![/bold]\n")
    
    # Main interactive loop
    while True:
        try:
            console.print("[cyan]What would you like to do?[/cyan]")
            console.print("1. Analyze a data file")
            console.print("2. Get method recommendation")
            console.print("3. View available methods")
            console.print("4. Exit")
            
            choice = Prompt.ask("Choose an option", choices=["1", "2", "3", "4"], default="1")
            
            if choice == "1":
                asyncio.run(run_analysis())
            elif choice == "2":
                asyncio.run(recommend_method())
            elif choice == "3":
                print_methods_info()
            elif choice == "4":
                console.print("[green]Goodbye![/green]")
                break
                
            console.print("\n" + "="*80 + "\n")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled by user[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
            continue


async def run_analysis():
    """Run the complete analysis workflow."""
    console.print("\n[bold cyan]Anomaly Detection Analysis[/bold cyan]")
    
    # Setup logging for analysis
    app_logger = setup_logging(debug=False, log_file="logs/app.log")
    app_logger.info("Starting anomaly detection analysis from CLI")
    
    # Get settings
    settings = get_settings()
    app_logger.info(f"Configuration loaded: LLM={settings.llm_model}, Enable_LLM={settings.enable_llm}")
    
    # Get file path
    file_path = Prompt.ask("Enter path to your data file (CSV/Excel)")
    
    if not Path(file_path).exists():
        console.print(f"[red]Error:[/red] File not found: {file_path}")
        return
    
    # Get method
    console.print("\nAvailable methods: [cyan]z-score[/cyan], [cyan]iqr[/cyan], [cyan]rolling-iqr[/cyan], [cyan]dbscan[/cyan]")
    method = Prompt.ask("Choose detection method", default="z-score")
    
    if method not in ['z-score', 'iqr', 'rolling-iqr', 'dbscan']:
        console.print(f"[red]Error:[/red] Invalid method: {method}")
        return
    
    # Get threshold (optional)
    threshold = None
    if Confirm.ask("Do you want to specify a custom threshold?", default=False):
        try:
            threshold = float(Prompt.ask("Enter threshold value"))
        except ValueError:
            console.print("[yellow]Warning:[/yellow] Invalid threshold, using default")
    
    # Get user query
    query = Prompt.ask("Describe what you're looking for", default="Find anomalies in this data")
    
    # Run analysis with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        
        # Create single progress task
        task_analysis = progress.add_task("Analyzing data...", total=100)
        
        try:
            # Initialize supervisor
            app_logger.info(f"Initializing supervisor with LLM enabled: {settings.enable_llm}")
            supervisor = AnomalyDetectionSupervisor(enable_llm=settings.enable_llm)
            
            # Create request
            request = AnomalyDetectionRequest(
                file_path=file_path,
                method=method,
                threshold=threshold,
                query=query
            )
            app_logger.info(f"Created analysis request: {method} method, file: {file_path}")
            
            # Run analysis
            app_logger.info("Starting analysis execution")
            result = await supervisor.analyze(request)
            app_logger.info(f"Analysis completed: {result.anomaly_result.anomaly_count} anomalies found")
            
            # Update progress when complete
            progress.update(task_analysis, completed=100)
            
            # Display results
            console.print("\n[bold green]Analysis Complete![/bold green]\n")
            format_results(result)
            
        except FileProcessingError as e:
            console.print(f"\n[red]File Processing Error:[/red] {str(e)}")
        except Exception as e:
            console.print(f"\n[red]Analysis Error:[/red] {str(e)}")


async def recommend_method():
    """Recommend the best detection method for given data."""
    console.print("\n[bold cyan]Method Recommendation[/bold cyan]")
    
    file_path = Prompt.ask("Enter path to your data file")
    
    if not Path(file_path).exists():
        console.print(f"[red]Error:[/red] File not found: {file_path}")
        return
    
    try:
        with console.status("[bold green]Analyzing data characteristics..."):
            # Read file
            file_reader = FileReaderTool()
            data = file_reader.read_file(file_path)
            
            # Get recommendation
            detector = AnomalyDetectionTool()
            recommended = detector.recommend_method(data)
            method_info = detector.get_method_info(recommended)
        
        # Display recommendation
        rec_panel = Panel(
            f"[bold green]Recommended Method: {method_info.get('name', recommended)}[/bold green]\n\n"
            f"Reason: {method_info.get('best_for', 'General analysis')}\n"
            f"Default Threshold: {method_info.get('default_threshold', 'Auto')}\n"
            f"Description: {method_info.get('description', 'Statistical detection')}",
            title="Method Recommendation",
            border_style="green"
        )
        console.print(rec_panel)
        
        # Data info
        import pandas as pd
        series = pd.Series(data.values)
        info_table = Table(title="Data Characteristics")
        info_table.add_column("Characteristic", style="cyan")
        info_table.add_column("Value", style="white")
        
        info_table.add_row("Total Points", f"{len(data.values):,}")
        info_table.add_row("Time Range", f"{data.timestamp[0]} to {data.timestamp[-1]}")
        info_table.add_row("Value Range", f"{series.min():.2f} to {series.max():.2f}")
        info_table.add_row("Mean", f"{series.mean():.2f}")
        info_table.add_row("Std Dev", f"{series.std():.2f}")
        info_table.add_row("Skewness", f"{series.skew():.2f}")
        
        console.print(info_table)
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")


@app.command()
def analyze(
    file_path: str = typer.Argument(..., help="Path to the data file"),
    method: str = typer.Option("z-score", "--method", "-m", help="Detection method"),
    threshold: Optional[float] = typer.Option(None, "--threshold", "-t", help="Detection threshold"),
    query: str = typer.Option("Find anomalies in this data", "--query", "-q", help="Analysis query"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    enable_llm: Optional[bool] = typer.Option(None, "--llm", help="Enable LLM insights (overrides .env setting)"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging")
):
    """
    Analyze a data file for anomalies.
    
    Example:
        anomaly-detector analyze data.csv --method z-score --threshold 3.0
    """
    if debug:
        # Setup debug logging with file output
        setup_logging(debug=True, log_file="logs/debug.log")
        
        # Suppress matplotlib warnings and verbose logging
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import warnings
        warnings.filterwarnings("ignore", category=matplotlib.cbook.MatplotlibDeprecationWarning)
        warnings.filterwarnings("ignore", module="matplotlib")
        
        # Disable matplotlib debug logging
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)
        
        # Enable LangChain debug logging
        import os
        os.environ["LANGCHAIN_VERBOSE"] = "true"
        os.environ["LANGCHAIN_DEBUG"] = "true"
        
        # Enable additional LangChain logging
        langchain_logger = logging.getLogger("langchain")
        langchain_logger.setLevel(logging.DEBUG)
        
        openai_logger = logging.getLogger("openai")
        openai_logger.setLevel(logging.DEBUG)
        
        google_logger = logging.getLogger("google")
        google_logger.setLevel(logging.DEBUG)
        
        httpx_logger = logging.getLogger("httpx")
        httpx_logger.setLevel(logging.DEBUG)
        
        # Enable LangChain tracing for detailed LLM calls
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "anomaly-detection-debug"
        
        console.print("[yellow]Debug mode enabled - all logs will be saved to logs/debug.log[/yellow]")
        console.print("[yellow]LLM calls, prompts, and responses will be logged in detail[/yellow]")
    elif verbose:
        setup_logging(debug=False, log_file="logs/app.log")
        logging.getLogger().setLevel(logging.INFO)
        console.print("[yellow]Verbose mode enabled - logs will be saved to logs/app.log[/yellow]")
    
    console.print(f"[cyan]Analyzing file:[/cyan] {file_path}")
    
    # Validate inputs
    if not Path(file_path).exists():
        console.print(f"[red]Error:[/red] File not found: {file_path}")
        raise typer.Exit(1)
    
    if method not in ['z-score', 'iqr', 'rolling-iqr', 'dbscan']:
        console.print(f"[red]Error:[/red] Invalid method: {method}")
        raise typer.Exit(1)
    
    async def run():
        try:
            # Get settings and determine LLM enabling
            settings = get_settings()
            use_llm = enable_llm if enable_llm is not None else settings.enable_llm
            
            if verbose or debug:
                source = "CLI flag" if enable_llm is not None else ".env file"
                console.print(f"[dim]LLM enabled: {use_llm} (from {source})[/dim]")
            
            with console.status("[bold green]Running analysis..."):
                supervisor = AnomalyDetectionSupervisor(enable_llm=use_llm)
                
                request = AnomalyDetectionRequest(
                    file_path=file_path,
                    method=method,
                    threshold=threshold,
                    query=query
                )
                
                result = await supervisor.analyze(request)
            
            console.print("[bold green]Analysis Complete![/bold green]\n")
            format_results(result)
            
            # Save output if requested
            if output:
                import json
                output_data = {
                    "anomaly_result": result.anomaly_result.dict(),
                    "insights": result.insights.dict() if result.insights else None,
                    "processing_time": result.processing_time
                }
                
                with open(output, 'w') as f:
                    json.dump(output_data, f, indent=2, default=str)
                
                console.print(f"[green]Results saved to:[/green] {output}")
            
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
            raise typer.Exit(1)
    
    asyncio.run(run())


@app.command()
def validate(
    file_path: str = typer.Argument(..., help="Path to the data file")
):
    """
    Validate a data file for anomaly detection compatibility.
    """
    console.print(f"[cyan]Validating file:[/cyan] {file_path}")
    
    if not Path(file_path).exists():
        console.print(f"[red]Error:[/red] File not found: {file_path}")
        raise typer.Exit(1)
    
    try:
        with console.status("[bold green]Validating data..."):
            file_reader = FileReaderTool()
            
            # Get file info
            file_info = file_reader.get_file_info(file_path)
            
            # Read data
            data = file_reader.read_file(file_path)
        
        # Display file info
        info_table = Table(title="File Information")
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="white")
        
        info_table.add_row("Filename", file_info.filename)
        info_table.add_row("File Type", file_info.file_type.upper())
        info_table.add_row("File Size", f"{file_info.file_size:,} bytes")
        info_table.add_row("Columns", ", ".join(file_info.columns))
        info_table.add_row("Rows", f"{file_info.row_count:,}")
        info_table.add_row("Value Column", data.column_name)
        info_table.add_row("Valid Points", f"{len(data.values):,}")
        
        console.print(info_table)
        
        # Validation status
        console.print(f"\n[bold green]✓ File is valid for anomaly detection![/bold green]")
        console.print(f"[dim]Ready to analyze {len(data.values):,} data points[/dim]")
        
    except Exception as e:
        console.print(f"[red]✗ Validation failed:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def methods():
    """
    List available anomaly detection methods and their descriptions.
    """
    print_methods_info()


@app.command()
def config():
    """
    Show current configuration.
    """
    try:
        settings = get_settings()
        
        config_table = Table(title="Current Configuration")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="white")
        
        config_table.add_row("LLM Model", settings.llm_model)
        config_table.add_row("LLM Temperature", str(settings.llm_temperature))
        config_table.add_row("Data Directory", str(settings.data_dir))
        config_table.add_row("Output Directory", str(settings.output_dir))
        config_table.add_row("Plots Directory", str(settings.plots_dir))
        config_table.add_row("Log Level", settings.log_level)
        config_table.add_row("Debug Mode", str(settings.debug))
        
        # Mask API key for security
        api_key = settings.google_ai_api_key
        masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
        config_table.add_row("API Key", masked_key)
        
        console.print(config_table)
        
    except Exception as e:
        console.print(f"[red]Error loading configuration:[/red] {str(e)}")


if __name__ == "__main__":
    app()