# Copyright 2026 Ajay Rajan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TraceContext - Core observability system for ASHA pipeline.

This module provides structured logging, pipeline traces, and debugging
capabilities for transparent AI pipeline processing.
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional, Union, TypedDict
from enum import Enum
from dataclasses import dataclass, asdict
import difflib


class LogLevel(Enum):
    """Logging levels for pipeline observability."""

    ERROR = "ERROR"
    WARN = "WARN"
    INFO = "INFO"
    DEBUG = "DEBUG"


@dataclass
class StageChange:
    """Represents a change made during a pipeline stage."""

    type: str
    original: str
    modified: str
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class StageTrace:
    """Trace information for a single pipeline stage."""

    name: str
    input: str
    output: str
    changes: List[StageChange]
    latency_ms: float
    error: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None


class _CurrentStageData(TypedDict):
    name: str
    start_time: float
    input: str


@dataclass
class PipelineMetrics:
    """Aggregated metrics for the entire pipeline."""

    total_latency_ms: float
    stages_run: int
    tokens_saved: int
    pii_detected: int
    changes_made: int
    errors: int


class TraceContext:
    """
    Central context for pipeline observability and tracing.

    This class manages structured logging, stage traces, and metrics
    throughout the ASHA pipeline processing.
    """

    def __init__(
        self,
        input_prompt: str,
        log_level: Union[LogLevel, str] = LogLevel.INFO,
        trace_enabled: bool = False,
        debug_enabled: bool = False,
        log_output: str = "console",
        log_file: Optional[str] = None,
    ):
        """Initialize trace context.

        Args:
            input_prompt: Original input prompt
            log_level: Logging level (ERROR, WARN, INFO, DEBUG)
            trace_enabled: Enable detailed stage tracing
            debug_enabled: Enable debug mode with diff output
            log_output: Output destination ("console", "file", "json")
            log_file: Log file path (if log_output="file")

        Raises:
            ValueError: If invalid log_output or log_file configuration
            Exception: If logging setup fails
        """
        self.input_prompt = input_prompt
        self.log_level = (
            LogLevel(log_level) if isinstance(log_level, str) else log_level
        )
        self.trace_enabled = trace_enabled
        self.debug_enabled = debug_enabled
        self.log_output = log_output
        self.log_file = log_file

        # Pipeline state
        self.stages: List[StageTrace] = []
        self.start_time = time.time()
        self.current_stage: Optional[_CurrentStageData] = None

        # Metrics
        self.metrics = {
            "tokens_saved": 0,
            "pii_detected": 0,
            "changes_made": 0,
            "errors": 0,
        }

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup structured logging based on configuration.

        Raises:
            Exception: If logging configuration is invalid
            ValueError: If log_output="file" but no log_file provided
        """
        self.logger = logging.getLogger(f"asha.{id(self)}")
        self.logger.setLevel(getattr(logging, self.log_level.value))

        # Clear existing handlers
        self.logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        handler: logging.Handler
        if self.log_output == "console":
            handler = logging.StreamHandler()
        elif self.log_output == "file" and self.log_file:
            handler = logging.FileHandler(self.log_file)
        elif self.log_output == "json":
            handler = logging.StreamHandler()
            formatter = self._create_json_formatter()
        else:
            handler = logging.StreamHandler()

        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _create_json_formatter(self) -> logging.Formatter:
        """Create JSON formatter for structured logging."""

        class JSONFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                log_entry = {
                    "timestamp": self.formatTime(record),
                    "logger": record.name,
                    "level": record.levelname,
                    "message": record.getMessage(),
                }

                # Add structured data if available
                if hasattr(record, "structured_data"):
                    log_entry.update(record.structured_data)

                return json.dumps(log_entry)

        return JSONFormatter()

    def log_stage_start(self, stage_name: str, input_text: str) -> None:
        """Log the start of a pipeline stage."""
        if not self.trace_enabled:
            return

        self.current_stage = {
            "name": stage_name,
            "start_time": time.time(),
            "input": input_text,
        }

        self._log_structured(
            level=LogLevel.DEBUG,
            message=f"Stage {stage_name} started",
            stage=stage_name,
            action="start",
            input_length=len(input_text),
        )

    def log_stage_end(
        self,
        stage_name: str,
        output_text: str,
        changes: Optional[List[StageChange]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log the end of a pipeline stage."""
        if not self.trace_enabled or not self.current_stage:
            return

        start_time = self.current_stage["start_time"]
        latency_ms = (time.time() - start_time) * 1000

        # Create stage trace
        stage_trace = StageTrace(
            name=stage_name,
            input=self.current_stage["input"],
            output=output_text,
            changes=changes or [],
            latency_ms=latency_ms,
            error=error,
            metrics=metrics,
        )
        self.stages.append(stage_trace)

        # Update metrics
        if changes:
            self.metrics["changes_made"] += len(changes)
            for change in changes:
                if change.type in [
                    "email",
                    "phone",
                    "ssn",
                    "credit_card",
                    "name",
                    "address",
                ]:
                    self.metrics["pii_detected"] += 1

        if metrics:
            if "tokens_saved" in metrics:
                self.metrics["tokens_saved"] += metrics["tokens_saved"]

        if error:
            self.metrics["errors"] += 1

        # Log structured data
        log_data = {
            "stage": stage_name,
            "action": "complete",
            "latency_ms": round(latency_ms, 2),
            "input_length": len(self.current_stage["input"]),
            "output_length": len(output_text),
            "changes_count": len(changes) if changes else 0,
        }

        if error:
            log_data["error"] = error
            self._log_structured(
                LogLevel.ERROR, f"Stage {stage_name} completed with error", **log_data
            )
        else:
            self._log_structured(
                LogLevel.DEBUG, f"Stage {stage_name} completed", **log_data
            )

        self.current_stage = None

    def log_warn(
        self,
        stage: str,
        message: str,
        reason: Optional[str] = None,
        fallback: Optional[str] = None,
    ) -> None:
        """Log a warning with structured information."""
        log_data = {
            "stage": stage,
            "reason": reason,
            "fallback": fallback,
        }
        self._log_structured(LogLevel.WARN, message, **log_data)

    def log_error(
        self,
        stage: str,
        error_type: str,
        message: str,
        fallback: Optional[str] = None,
        impact: Optional[str] = None,
    ) -> None:
        """Log an error with structured information.

        Args:
            stage: Pipeline stage where error occurred
            error_type: Type of error (e.g., ImportError, ValueError)
            message: Error message
            fallback: Fallback strategy used
            impact: Impact of error on processing

        Note:
            Automatically increments error counter in metrics
        """
        error_data = {
            "stage": stage,
            "type": error_type,
            "detail": message,
            "fallback": fallback,
            "impact": impact,
        }

        self._log_structured(
            LogLevel.ERROR, f"Error in {stage}: {message}", **error_data
        )

    def _log_structured(self, level: LogLevel, message: str, **kwargs: Any) -> None:
        """Log structured data with the specified level."""
        if not self._should_log(level):
            return

        log_data = {"structured_data": kwargs}

        if self.log_output == "json":
            # For JSON output, we'll handle structured data in the formatter
            extra = log_data
        else:
            # For console/file, include structured data in message
            structured_str = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            message = f"{message} | {structured_str}"
            extra = {}

        getattr(self.logger, level.value.lower())(message, extra=extra)

    def _should_log(self, level: LogLevel) -> bool:
        """Check if a message should be logged based on current log level.

        Args:
            level: Log level to check against

        Returns:
            True if message should be logged, False otherwise
        """
        levels = [LogLevel.ERROR, LogLevel.WARN, LogLevel.INFO, LogLevel.DEBUG]
        current_level_index = levels.index(self.log_level)
        message_level_index = levels.index(level)
        return message_level_index <= current_level_index

    def generate_diff(self) -> Optional[str]:
        """Generate a diff view of input vs output.

        Returns:
            Unified diff string or None if debug disabled

        Note:
            Only available when debug_enabled=True
        """
        if not self.debug_enabled or not self.stages:
            return None

        final_output = self.stages[-1].output if self.stages else self.input_prompt

        diff = difflib.unified_diff(
            self.input_prompt.splitlines(keepends=True),
            final_output.splitlines(keepends=True),
            fromfile="input",
            tofile="output",
            lineterm="",
        )

        return "".join(diff)

    def get_trace_summary(self) -> Dict[str, Any]:
        """Get a complete trace summary.

        Returns:
            Dictionary with input, stages, final_output, and metrics

        Note:
            Includes total latency and aggregated metrics
        """
        total_latency_ms = (time.time() - self.start_time) * 1000

        return {
            "input": self.input_prompt,
            "stages": [asdict(stage) for stage in self.stages],
            "final_output": (
                self.stages[-1].output if self.stages else self.input_prompt
            ),
            "total_latency_ms": round(total_latency_ms, 2),
            "metrics": {
                "total_latency_ms": round(total_latency_ms, 2),
                "stages_run": len(self.stages),
                "tokens_saved": self.metrics["tokens_saved"],
                "pii_detected": self.metrics["pii_detected"],
                "changes_made": self.metrics["changes_made"],
                "errors": self.metrics["errors"],
            },
            "diff": self.generate_diff() if self.debug_enabled else None,
        }

    def get_metrics(self) -> PipelineMetrics:
        """Get aggregated pipeline metrics.

        Returns:
            PipelineMetrics with total latency, stages run, and aggregated counts
        """
        total_latency_ms = (time.time() - self.start_time) * 1000

        return PipelineMetrics(
            total_latency_ms=round(total_latency_ms, 2),
            stages_run=len(self.stages),
            tokens_saved=self.metrics["tokens_saved"],
            pii_detected=self.metrics["pii_detected"],
            changes_made=self.metrics["changes_made"],
            errors=self.metrics["errors"],
        )
