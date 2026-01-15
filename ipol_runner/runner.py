"""Method execution logic."""
import time
from pathlib import Path
from typing import Dict, Any, List

from .base import IPOLMethod, MethodResult


def run_method(
    method: IPOLMethod,
    inputs: List[Path],
    output_dir: Path,
    params: Dict[str, Any],
    verbose: bool = False
) -> MethodResult:
    """Execute a method with the given inputs and parameters.

    Args:
        method: The IPOL method to run
        inputs: List of input file paths
        output_dir: Directory to write outputs
        params: Method parameters
        verbose: Print verbose output

    Returns:
        MethodResult with outputs and status
    """
    # Validate inputs
    error = method.validate_inputs(inputs)
    if error:
        return MethodResult(
            success=False,
            output_dir=output_dir,
            error_message=error
        )

    # Validate parameters
    error = method.validate_params(params)
    if error:
        return MethodResult(
            success=False,
            output_dir=output_dir,
            error_message=error
        )

    # Apply defaults for missing parameters
    param_schema = method.get_parameters()
    full_params = {}
    for name, spec in param_schema.items():
        if name in params:
            full_params[name] = params[name]
        elif "default" in spec:
            full_params[name] = spec["default"]

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nRunning {method.display_name}...")
        print(f"Inputs: {[str(p) for p in inputs]}")
        print(f"Output: {output_dir}")
        print(f"Parameters: {full_params}")

    # Run the method
    start_time = time.time()
    try:
        result = method.run(inputs, output_dir, full_params)
        result.execution_time = time.time() - start_time
        if verbose:
            print(f"Execution time: {result.execution_time:.2f}s")
        return result
    except Exception as e:
        return MethodResult(
            success=False,
            output_dir=output_dir,
            execution_time=time.time() - start_time,
            error_message=str(e)
        )
