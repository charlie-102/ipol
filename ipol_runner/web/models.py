"""Pydantic models for web API."""
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid


class MethodInfo(BaseModel):
    """Method information response."""
    name: str
    display_name: str
    description: str
    category: str
    input_type: str
    input_count: int
    requires_cuda: bool
    supports_mps: bool
    parameters: Dict[str, Dict[str, Any]]


class MethodListItem(BaseModel):
    """Abbreviated method info for list view."""
    name: str
    display_name: str
    category: str
    input_type: str
    description: str = ""


class UploadResponse(BaseModel):
    """Response after file upload."""
    id: str
    filename: str
    path: str


class ExperimentCreate(BaseModel):
    """Request to create a new experiment."""
    method_name: str
    input_ids: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ExperimentStatus(BaseModel):
    """Experiment status response."""
    id: str
    method_name: str
    status: str  # pending, running, completed, failed
    created_at: datetime
    completed_at: Optional[datetime] = None
    execution_time: Optional[float] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    inputs: List[str] = Field(default_factory=list)
    outputs: Dict[str, str] = Field(default_factory=dict)
    primary_output: Optional[str] = None
    error_message: Optional[str] = None
    notes: str = ""


class ExperimentNotes(BaseModel):
    """Update notes for an experiment."""
    notes: str


class ComparisonCreate(BaseModel):
    """Request to create a comparison."""
    experiment_ids: List[str]
    notes: str = ""


class ComparisonStatus(BaseModel):
    """Comparison status response."""
    id: str
    experiment_ids: List[str]
    notes: str
    created_at: datetime


class DeviceInfo(BaseModel):
    """Device availability information."""
    name: str
    available: bool
    message: str


class DevicesResponse(BaseModel):
    """Response with all device information."""
    devices: List[DeviceInfo]
    recommended: str
    mps_methods: List[str]


def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())[:8]
