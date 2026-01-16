"""Unified CLI for IPOL methods."""
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from .base import MethodCategory, InputType
from .registry import get_method, list_methods, get_all_methods
from .runner import run_method
from .comparison import create_comparison
from .web_viewer import save_gallery, generate_comparison_html


def parse_param(param_str: str) -> tuple:
    """Parse KEY=VALUE parameter string."""
    if "=" not in param_str:
        raise ValueError(f"Invalid parameter format: {param_str} (expected KEY=VALUE)")
    key, value = param_str.split("=", 1)
    # Try to parse as number or bool
    if value.lower() == "true":
        return key, True
    elif value.lower() == "false":
        return key, False
    try:
        return key, int(value)
    except ValueError:
        pass
    try:
        return key, float(value)
    except ValueError:
        pass
    return key, value


def cmd_list(args):
    """List available methods."""
    methods = get_all_methods()
    if not methods:
        print("No methods registered.")
        return 1

    # Filter by category if specified
    if args.category:
        try:
            cat = MethodCategory(args.category)
            methods = {k: v for k, v in methods.items() if v.category == cat}
        except ValueError:
            print(f"Unknown category: {args.category}")
            print(f"Valid categories: {', '.join(c.value for c in MethodCategory)}")
            return 1

    # Filter by input type if specified
    if args.input_type:
        try:
            inp = InputType(args.input_type)
            methods = {k: v for k, v in methods.items() if v.input_type == inp}
        except ValueError:
            print(f"Unknown input type: {args.input_type}")
            print(f"Valid types: {', '.join(t.value for t in InputType)}")
            return 1

    print(f"\nAvailable methods ({len(methods)}):\n")
    for name, method in methods.items():
        if args.verbose:
            print(f"  {name}")
            print(f"    Name: {method.display_name}")
            print(f"    Category: {method.category.value}")
            print(f"    Input: {method.input_type.value}")
            print(f"    Description: {method.description or 'N/A'}")
            print()
        else:
            cat = method.category.value[:12]
            desc = method.description or method.display_name
            if len(desc) > 50:
                desc = desc[:47] + "..."
            print(f"  {name:18} [{cat:12}] {desc}")
    return 0


def cmd_info(args):
    """Show method information."""
    method = get_method(args.method)
    if not method:
        print(f"Error: Unknown method '{args.method}'")
        print(f"Available methods: {', '.join(list_methods())}")
        return 1

    print(f"\n{method.display_name}")
    print("=" * len(method.display_name))
    print(f"Name: {method.name}")
    print(f"Category: {method.category.value}")
    print(f"Input type: {method.input_type.value}")
    if method.description:
        print(f"Description: {method.description}")
    print(f"Required inputs: {method.input_count}")
    if method.requirements_file and method.requirements_file.exists():
        print(f"Dependencies: {method.requirements_file}")

    params = method.get_parameters()
    if params:
        print("\nParameters:")
        for name, spec in params.items():
            required = spec.get("required", False)
            default = spec.get("default", "N/A")
            desc = spec.get("description", "")
            req_str = " (required)" if required else f" [default: {default}]"
            print(f"  --param {name}=<value>{req_str}")
            if desc:
                print(f"      {desc}")
            if spec.get("type") == "choice" and "choices" in spec:
                print(f"      Choices: {', '.join(spec['choices'])}")
    return 0


def cmd_run(args):
    """Run a method."""
    method = get_method(args.method)
    if not method:
        print(f"Error: Unknown method '{args.method}'")
        print(f"Available methods: {', '.join(list_methods())}")
        return 1

    # Parse inputs
    inputs = [Path(p) for p in args.input]

    # Parse parameters
    params = {}
    if args.param:
        for p in args.param:
            try:
                key, value = parse_param(p)
                params[key] = value
            except ValueError as e:
                print(f"Error: {e}")
                return 1

    # Output directory
    output_dir = Path(args.output)

    # Run
    result = run_method(method, inputs, output_dir, params, verbose=args.verbose)

    if result.success:
        print(f"\nSuccess! Output saved to: {result.output_dir}")
        if result.primary_output:
            print(f"Primary output: {result.primary_output}")
        if result.metrics:
            print("\nMetrics:")
            for k, v in result.metrics.items():
                print(f"  {k}: {v:.4f}")
        return 0
    else:
        print(f"\nError: {result.error_message}")
        return 1


def cmd_compare(args):
    """Compare method outputs."""
    # Parse inputs
    inputs = [Path(p) for p in args.input]
    output_dir = Path(args.output)

    result = create_comparison(
        method_names=args.methods,
        inputs=inputs,
        output_dir=output_dir,
        verbose=args.verbose
    )

    if result:
        print(f"\nComparison saved to: {output_dir}")
        return 0
    return 1


def cmd_gallery(args):
    """Generate web gallery for visual comparison."""
    import webbrowser

    output_dir = Path(args.output)
    output_base = Path(args.results) if args.results else output_dir

    print(f"Generating gallery from: {output_base}")
    gallery_path = save_gallery(output_dir, output_base)
    print(f"Gallery saved to: {gallery_path}")

    if args.open:
        webbrowser.open(f"file://{gallery_path.absolute()}")
        print("Opened in browser.")

    return 0


def cmd_test(args):
    """Test method(s) with sample inputs."""
    from .testing import test_method, test_all_methods

    if args.method:
        method = get_method(args.method)
        if not method:
            print(f"Error: Unknown method '{args.method}'")
            return 1
        success, message = test_method(method, verbose=args.verbose)
        print(message)
        return 0 if success else 1
    else:
        results = test_all_methods(verbose=args.verbose)
        passed = sum(1 for s, _ in results.values() if s)
        total = len(results)
        print(f"\nResults: {passed}/{total} methods passed")
        return 0 if passed == total else 1


def cmd_status(args):
    """Show validation status of all methods."""
    from .validation import print_validation_report
    print_validation_report()
    return 0


def cmd_devices(args):
    """Show available compute devices."""
    from .device_utils import print_device_info
    print_device_info()

    # Also show methods that support each device
    print()
    print("Methods with MPS Support:")
    print("-" * 40)
    methods = get_all_methods()
    mps_methods = [name for name, m in methods.items() if m.supports_mps]
    if mps_methods:
        for name in sorted(mps_methods):
            print(f"  - {name}")
    else:
        print("  (none)")

    return 0


def cmd_web(args):
    """Start web interface."""
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed. Install with: pip install uvicorn[standard]")
        return 1

    print(f"Starting IPOL Runner web interface on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")

    uvicorn.run(
        "ipol_runner.web.app:app",
        host=args.host,
        port=args.port,
        reload=False
    )
    return 0


def cmd_assets(args):
    """Manage large model assets."""
    from .assets import AssetManager, print_asset_status

    if args.assets_command == "status" or args.assets_command is None:
        print_asset_status()
        return 0

    manager = AssetManager()

    if args.assets_command == "download":
        assets = manager.list_assets()

        if args.method:
            # Download assets for specific method
            method_assets = [a for a in assets if a.get('method') == args.method]
            if not method_assets:
                print(f"No assets found for method: {args.method}")
                return 1
            for asset in method_assets:
                print(f"Downloading {asset['key']}...")
                manager.download_asset(asset['key'])
        elif args.all:
            # Download all assets
            for asset in assets:
                print(f"Downloading {asset['key']}...")
                manager.download_asset(asset['key'])
        else:
            print("Specify --method or --all")
            return 1
        return 0

    if args.assets_command == "clear":
        older_than = getattr(args, 'older_than', None)
        manager.clear_cache(older_than_days=older_than)
        return 0

    print("Unknown assets command")
    return 1


def cmd_deps(args):
    """Install dependencies for a method."""
    import subprocess

    method = get_method(args.method)
    if not method:
        print(f"Error: Unknown method '{args.method}'")
        return 1

    req_file = method.requirements_file
    if not req_file:
        print(f"No requirements file defined for {args.method}")
        return 1

    if not req_file.exists():
        print(f"Requirements file not found: {req_file}")
        return 1

    print(f"Installing dependencies for {method.display_name}...")
    print(f"Requirements: {req_file}")

    if req_file.suffix == ".toml":
        # For pyproject.toml, use pip install with the directory
        cmd = [sys.executable, "-m", "pip", "install", "-e", str(req_file.parent)]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_file)]

    if args.quiet:
        cmd.append("-q")

    result = subprocess.run(cmd)
    return result.returncode


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="ipol",
        description="Unified CLI for IPOL image processing methods"
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version="%(prog)s 0.1.0"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # list command
    list_p = subparsers.add_parser("list", help="List available methods")
    list_p.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    list_p.add_argument(
        "-c", "--category",
        help=f"Filter by category: {', '.join(c.value for c in MethodCategory)}"
    )
    list_p.add_argument(
        "-t", "--input-type",
        dest="input_type",
        help=f"Filter by input type: {', '.join(t.value for t in InputType)}"
    )

    # info command
    info_p = subparsers.add_parser("info", help="Show method information")
    info_p.add_argument("method", help="Method name")

    # run command
    run_p = subparsers.add_parser("run", help="Run a method")
    run_p.add_argument("method", help="Method name")
    run_p.add_argument(
        "-i", "--input",
        action="append",
        required=True,
        help="Input file (can specify multiple)"
    )
    run_p.add_argument(
        "-o", "--output",
        default="./output",
        help="Output directory (default: ./output)"
    )
    run_p.add_argument(
        "--param",
        action="append",
        help="Set parameter: --param KEY=VALUE"
    )
    run_p.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # deps command
    deps_p = subparsers.add_parser("deps", help="Install dependencies for a method")
    deps_p.add_argument("method", help="Method name")
    deps_p.add_argument("-q", "--quiet", action="store_true", help="Quiet output")

    # compare command
    compare_p = subparsers.add_parser("compare", help="Compare method outputs")
    compare_p.add_argument(
        "-m", "--methods",
        nargs="+",
        required=True,
        help="Methods to compare"
    )
    compare_p.add_argument(
        "-i", "--input",
        action="append",
        required=True,
        help="Input file(s)"
    )
    compare_p.add_argument(
        "-o", "--output",
        default="./comparison",
        help="Output directory"
    )
    compare_p.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # gallery command
    gallery_p = subparsers.add_parser("gallery", help="Generate web gallery for visual comparison")
    gallery_p.add_argument(
        "-o", "--output",
        default="./output",
        help="Output directory for gallery.html"
    )
    gallery_p.add_argument(
        "-r", "--results",
        help="Directory containing method results (defaults to output dir)"
    )
    gallery_p.add_argument(
        "--open", action="store_true",
        help="Open gallery in browser"
    )

    # test command
    test_p = subparsers.add_parser("test", help="Test a method with sample input")
    test_p.add_argument("method", nargs="?", help="Method name (tests all if omitted)")
    test_p.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # status command
    subparsers.add_parser("status", help="Show validation status of all methods")

    # devices command
    subparsers.add_parser("devices", help="Show available compute devices (CPU, CUDA, MPS)")

    # web command
    web_p = subparsers.add_parser("web", help="Start web interface")
    web_p.add_argument("--port", type=int, default=8000, help="Port to run on (default: 8000)")
    web_p.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")

    # assets command
    assets_p = subparsers.add_parser("assets", help="Manage large model assets")
    assets_sub = assets_p.add_subparsers(dest="assets_command")
    assets_sub.add_parser("status", help="Show asset cache status")
    assets_download_p = assets_sub.add_parser("download", help="Download assets")
    assets_download_p.add_argument("-m", "--method", help="Download assets for a specific method")
    assets_download_p.add_argument("-a", "--all", action="store_true", help="Download all assets")
    assets_clear_p = assets_sub.add_parser("clear", help="Clear asset cache")
    assets_clear_p.add_argument("--older-than", type=int, help="Only clear files older than N days")

    return parser


def main(argv: Optional[List[str]] = None):
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        return cmd_list(args)
    elif args.command == "info":
        return cmd_info(args)
    elif args.command == "run":
        return cmd_run(args)
    elif args.command == "deps":
        return cmd_deps(args)
    elif args.command == "compare":
        return cmd_compare(args)
    elif args.command == "gallery":
        return cmd_gallery(args)
    elif args.command == "test":
        return cmd_test(args)
    elif args.command == "status":
        return cmd_status(args)
    elif args.command == "devices":
        return cmd_devices(args)
    elif args.command == "web":
        return cmd_web(args)
    elif args.command == "assets":
        return cmd_assets(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
