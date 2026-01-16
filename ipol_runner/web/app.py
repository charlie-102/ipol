"""FastAPI web application for IPOL Runner."""
import asyncio
import shutil
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from ..registry import get_method, get_all_methods
from ..runner import run_method
from ..device_utils import check_device_available, get_recommended_device, VALID_DEVICES
from .db import Database
from .gallery import get_gallery_html
from .models import (
    MethodInfo,
    MethodListItem,
    UploadResponse,
    ExperimentCreate,
    ExperimentStatus,
    ExperimentNotes,
    DevicesResponse,
    DeviceInfo,
    generate_id,
)

# Initialize app
app = FastAPI(
    title="IPOL Runner",
    description="Web interface for IPOL image processing methods",
    version="0.1.0"
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database
db = Database()

# Upload and output directories
def _get_data_dir():
    """Get data directory, falling back to temp if home is not writable."""
    home_dir = Path.home() / ".ipol_runner"
    try:
        home_dir.mkdir(parents=True, exist_ok=True)
        return home_dir
    except PermissionError:
        import tempfile
        fallback = Path(tempfile.gettempdir()) / "ipol_runner"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

_DATA_DIR = _get_data_dir()
UPLOAD_DIR = _DATA_DIR / "uploads"
OUTPUT_DIR = _DATA_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- Methods API ---

@app.get("/api/methods", response_model=List[MethodListItem])
async def list_methods(
    category: Optional[str] = None,
    input_type: Optional[str] = None
):
    """List all available methods."""
    methods = get_all_methods()

    result = []
    for name, method in methods.items():
        # Apply filters
        if category and method.category.value != category:
            continue
        if input_type and method.input_type.value != input_type:
            continue

        result.append(MethodListItem(
            name=name,
            display_name=method.display_name,
            category=method.category.value,
            input_type=method.input_type.value,
            description=method.description or ""
        ))

    return sorted(result, key=lambda m: m.name)


@app.get("/api/methods/{name}", response_model=MethodInfo)
async def get_method_info(name: str):
    """Get detailed information about a method."""
    method = get_method(name)
    if not method:
        raise HTTPException(status_code=404, detail=f"Method not found: {name}")

    return MethodInfo(
        name=method.name,
        display_name=method.display_name,
        description=method.description or "",
        category=method.category.value,
        input_type=method.input_type.value,
        input_count=method.input_count,
        requires_cuda=method.requires_cuda,
        supports_mps=method.supports_mps,
        parameters=method.get_parameters()
    )


# --- Upload API ---

@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a file for processing."""
    upload_id = generate_id()
    file_ext = Path(file.filename).suffix
    file_path = UPLOAD_DIR / f"{upload_id}{file_ext}"

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    db.save_upload(upload_id, file.filename, str(file_path))

    return UploadResponse(
        id=upload_id,
        filename=file.filename,
        path=str(file_path)
    )


@app.get("/api/uploads/{upload_id}")
async def get_upload(upload_id: str):
    """Get upload info or serve the file."""
    upload = db.get_upload(upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")

    file_path = Path(upload["path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, filename=upload["filename"])


# --- Experiments API ---

def run_experiment_task(exp_id: str, method_name: str, input_paths: List[Path], params: dict):
    """Background task to run an experiment."""
    method = get_method(method_name)
    if not method:
        db.update_experiment_status(exp_id, "failed", error_message=f"Method not found: {method_name}")
        return

    output_dir = OUTPUT_DIR / exp_id
    output_dir.mkdir(parents=True, exist_ok=True)

    db.update_experiment_status(exp_id, "running")

    start_time = time.time()
    try:
        result = run_method(method, input_paths, output_dir, params, verbose=False)
        execution_time = time.time() - start_time

        if result.success:
            outputs = {k: str(v) for k, v in result.outputs.items()}
            primary = str(result.primary_output) if result.primary_output else None
            db.update_experiment_status(
                exp_id, "completed",
                outputs=outputs,
                primary_output=primary,
                execution_time=execution_time
            )
        else:
            db.update_experiment_status(
                exp_id, "failed",
                error_message=result.error_message,
                execution_time=execution_time
            )
    except Exception as e:
        execution_time = time.time() - start_time
        db.update_experiment_status(
            exp_id, "failed",
            error_message=str(e),
            execution_time=execution_time
        )


@app.post("/api/experiments", response_model=ExperimentStatus)
async def create_experiment(
    experiment: ExperimentCreate,
    background_tasks: BackgroundTasks
):
    """Create and run a new experiment."""
    method = get_method(experiment.method_name)
    if not method:
        raise HTTPException(status_code=404, detail=f"Method not found: {experiment.method_name}")

    # Resolve input paths
    input_paths = []
    for upload_id in experiment.input_ids:
        upload = db.get_upload(upload_id)
        if not upload:
            raise HTTPException(status_code=404, detail=f"Upload not found: {upload_id}")
        input_paths.append(Path(upload["path"]))

    # Create experiment
    exp_id = db.create_experiment(
        experiment.method_name,
        experiment.input_ids,
        experiment.parameters
    )

    # Run in background
    background_tasks.add_task(
        run_experiment_task,
        exp_id,
        experiment.method_name,
        input_paths,
        experiment.parameters
    )

    return db.get_experiment(exp_id)


@app.get("/api/experiments", response_model=List[ExperimentStatus])
async def list_experiments(
    method_name: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100
):
    """List experiments."""
    return db.list_experiments(method_name=method_name, status=status, limit=limit)


@app.get("/api/experiments/{exp_id}", response_model=ExperimentStatus)
async def get_experiment(exp_id: str):
    """Get experiment status."""
    experiment = db.get_experiment(exp_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


@app.put("/api/experiments/{exp_id}/notes")
async def update_notes(exp_id: str, notes: ExperimentNotes):
    """Update experiment notes."""
    experiment = db.get_experiment(exp_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    db.update_experiment_notes(exp_id, notes.notes)
    return {"status": "ok"}


@app.delete("/api/experiments/{exp_id}")
async def delete_experiment(exp_id: str):
    """Delete an experiment."""
    experiment = db.get_experiment(exp_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Delete output directory
    output_dir = OUTPUT_DIR / exp_id
    if output_dir.exists():
        shutil.rmtree(output_dir)

    db.delete_experiment(exp_id)
    return {"status": "ok"}


# --- Outputs API ---

@app.get("/api/outputs/{exp_id}/{filename}")
async def get_output_file(exp_id: str, filename: str):
    """Serve an experiment output file."""
    file_path = OUTPUT_DIR / exp_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(file_path)


# --- Devices API ---

@app.get("/api/devices", response_model=DevicesResponse)
async def get_devices():
    """Get available compute devices."""
    devices = []
    for device in VALID_DEVICES:
        avail, msg = check_device_available(device)
        devices.append(DeviceInfo(name=device, available=avail, message=msg))

    methods = get_all_methods()
    mps_methods = [name for name, m in methods.items() if m.supports_mps]

    return DevicesResponse(
        devices=devices,
        recommended=get_recommended_device(),
        mps_methods=sorted(mps_methods)
    )


# --- Validation Status API ---

@app.get("/api/validation")
async def get_validation_status():
    """Get validation status of all methods."""
    import json
    status_file = Path(__file__).parent.parent.parent / "validation_status.json"
    if status_file.exists():
        with open(status_file) as f:
            return json.load(f)
    return {"passed": {}, "failed": {}, "skipped_cuda": {}, "skipped_docker": {}, "mps_enabled": {}}


@app.get("/api/categories")
async def get_categories():
    """Get method counts by category."""
    methods = get_all_methods()
    categories = {}
    for name, method in methods.items():
        cat = method.category.value
        if cat not in categories:
            categories[cat] = {"count": 0, "methods": []}
        categories[cat]["count"] += 1
        categories[cat]["methods"].append({
            "name": name,
            "display_name": method.display_name,
            "input_type": method.input_type.value,
            "supports_mps": method.supports_mps,
            "requires_cuda": method.requires_cuda
        })
    return categories


# --- Gallery Frontend ---

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve gallery frontend."""
    return get_gallery_html()


@app.get("/gallery", response_class=HTMLResponse)
async def gallery():
    """Serve gallery frontend (alias)."""
    return get_gallery_html()


def create_app(db_path: Optional[Path] = None, upload_dir: Optional[Path] = None):
    """Create app with custom configuration."""
    global db, UPLOAD_DIR, OUTPUT_DIR

    if db_path:
        db = Database(db_path)

    if upload_dir:
        UPLOAD_DIR = upload_dir / "uploads"
        OUTPUT_DIR = upload_dir / "outputs"
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    return app
