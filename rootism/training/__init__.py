"""Rootism Distributed & Memory-Optimised Training Engine Package (Complete Steps 47–60)."""

from rootism.training.batching import BatchScheduleCoordinator, GlobalBatchPlan
from rootism.training.checkpointing import (
    AtomicCheckpointManager,
    CheckpointIntegrityError,
    LoadedCheckpointState,
    PreemptionSignalCatcher,
)
from rootism.training.distributed import (
    DistributedRuntimeContext,
    all_reduce_token_counts,
    build_tensor_parallel_plan,
    gradient_accumulation_sync_context,
    initialize_distributed_context,
    partition_layers_for_pipeline,
    resolve_distributed_strategy,
    wrap_model_for_distributed,
)
from rootism.training.hardware import (
    GPU_SPEC_PRESETS,
    HardwareCapabilities,
    InfrastructureEstimate,
    detect_hardware_capabilities,
    estimate_training_infrastructure,
)
from rootism.training.memory import PrecisionAndMemoryController
from rootism.training.optimizer import (
    OptimizerGroupSummary,
    RootismFP32MasterAdamW,
    build_optimizer,
    partition_parameter_groups,
)
from rootism.training.packing import SequencePacker
from rootism.training.profiler import (
    StepProfileRecord,
    ThroughputSummaryReport,
    TrainingThroughputProfiler,
)
from rootism.training.scheduler import RootismLRScheduler
from rootism.training.stability import StabilityDiagnostics, TrainingStabilityMonitor
from rootism.training.step_engine import StepExecutionResult, TrainingStepExecutor
from rootism.training.trainer import PretrainingRunSummary, RootismPretrainer

__all__ = [
    "GPU_SPEC_PRESETS",
    "HardwareCapabilities",
    "InfrastructureEstimate",
    "detect_hardware_capabilities",
    "estimate_training_infrastructure",
    "DistributedRuntimeContext",
    "resolve_distributed_strategy",
    "initialize_distributed_context",
    "wrap_model_for_distributed",
    "gradient_accumulation_sync_context",
    "all_reduce_token_counts",
    "build_tensor_parallel_plan",
    "partition_layers_for_pipeline",
    "PrecisionAndMemoryController",
    "OptimizerGroupSummary",
    "RootismFP32MasterAdamW",
    "partition_parameter_groups",
    "build_optimizer",
    "StepExecutionResult",
    "TrainingStepExecutor",
    "RootismLRScheduler",
    "BatchScheduleCoordinator",
    "GlobalBatchPlan",
    "SequencePacker",
    "StabilityDiagnostics",
    "TrainingStabilityMonitor",
    "AtomicCheckpointManager",
    "CheckpointIntegrityError",
    "LoadedCheckpointState",
    "PreemptionSignalCatcher",
    "StepProfileRecord",
    "ThroughputSummaryReport",
    "TrainingThroughputProfiler",
    "PretrainingRunSummary",
    "RootismPretrainer",
]
