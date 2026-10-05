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
Metrics Dashboard - Phase 5 Developer Addiction

CLI stats and observability that make developers LOVE using ASHA:
- Token usage analytics and forecasting
- Cost reduction tracking
- Performance metrics
- Security threat statistics
- Usage patterns analysis
- Export capabilities

Shows developers instant value and ongoing benefits.
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics


@dataclass
class MetricEvent:
    """Single metric event."""

    timestamp: datetime
    event_type: str
    value: float
    metadata: Dict[str, Any]


@dataclass
class UsageStats:
    """Usage statistics."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    total_tokens_processed: int
    total_tokens_saved: int
    cost_reduction_percentage: float
    threats_blocked: int
    pii_detected: int


@dataclass
class PerformanceMetrics:
    """Performance metrics."""

    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput_rps: float
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percentage: float


class MetricsCollector:
    """Collects and aggregates metrics."""

    def __init__(self, max_events: int = 10000) -> None:
        """Initialize metrics collector."""
        self.max_events = max_events
        self.events: deque = deque(maxlen=max_events)
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(lambda: [])
        self.start_time = datetime.now()

    def record_event(
        self, event_type: str, value: float, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a metric event."""
        event = MetricEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            value=value,
            metadata=metadata or {},
        )

        self.events.append(event)
        self.counters[event_type] += value
        self.histograms[event_type].append(value)

        # Keep histogram size manageable
        if len(self.histograms[event_type]) > 1000:
            self.histograms[event_type] = self.histograms[event_type][-1000:]

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge metric."""
        self.gauges[name] = value

    def get_usage_stats(self, time_window: Optional[timedelta] = None) -> UsageStats:
        """Get usage statistics for time window."""
        cutoff_time = datetime.now() - time_window if time_window else self.start_time

        # Filter events by time window
        relevant_events = [e for e in self.events if e.timestamp >= cutoff_time]

        if not relevant_events:
            return UsageStats(0, 0, 0, 0.0, 0, 0, 0.0, 0, 0)

        # Calculate statistics
        total_requests = len([e for e in relevant_events if e.event_type == "request"])
        successful_requests = len(
            [e for e in relevant_events if e.event_type == "success"]
        )
        failed_requests = total_requests - successful_requests

        # Latency statistics
        latencies = [e.value for e in relevant_events if e.event_type == "latency"]
        avg_latency = statistics.mean(latencies) if latencies else 0.0

        # Token statistics
        total_tokens = sum(
            e.value for e in relevant_events if e.event_type == "tokens_processed"
        )
        tokens_saved = sum(
            e.value for e in relevant_events if e.event_type == "tokens_saved"
        )

        # Calculate cost reduction
        cost_reduction = (
            (tokens_saved / total_tokens * 100) if total_tokens > 0 else 0.0
        )

        # Security statistics
        threats_blocked = len(
            [e for e in relevant_events if e.event_type == "threat_blocked"]
        )
        pii_detected = len(
            [e for e in relevant_events if e.event_type == "pii_detected"]
        )

        return UsageStats(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_latency_ms=avg_latency,
            total_tokens_processed=total_tokens,
            total_tokens_saved=tokens_saved,
            cost_reduction_percentage=cost_reduction,
            threats_blocked=threats_blocked,
            pii_detected=pii_detected,
        )

    def get_performance_metrics(
        self, time_window: Optional[timedelta] = None
    ) -> PerformanceMetrics:
        """Get performance metrics."""
        cutoff_time = datetime.now() - time_window if time_window else self.start_time

        # Filter events by time window
        relevant_events = [e for e in self.events if e.timestamp >= cutoff_time]

        if not relevant_events:
            return PerformanceMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Latency percentiles
        latencies = [e.value for e in relevant_events if e.event_type == "latency"]
        if latencies:
            latencies.sort()
            p50 = latencies[len(latencies) // 2]
            p95 = latencies[int(len(latencies) * 0.95)]
            p99 = latencies[int(len(latencies) * 0.99)]
        else:
            p50 = p95 = p99 = 0.0

        # Throughput (requests per second)
        time_diff = (datetime.now() - cutoff_time).total_seconds()
        requests = len([e for e in relevant_events if e.event_type == "request"])
        throughput = requests / time_diff if time_diff > 0 else 0.0

        # Error rate
        total_requests = len([e for e in relevant_events if e.event_type == "request"])
        errors = len([e for e in relevant_events if e.event_type == "error"])
        error_rate = errors / total_requests if total_requests > 0 else 0.0

        # System metrics (from gauges)
        memory_usage = self.gauges.get("memory_usage_mb", 0.0)
        cpu_usage = self.gauges.get("cpu_usage_percentage", 0.0)

        return PerformanceMetrics(
            latency_p50=p50,
            latency_p95=p95,
            latency_p99=p99,
            throughput_rps=throughput,
            error_rate=error_rate,
            memory_usage_mb=memory_usage,
            cpu_usage_percentage=cpu_usage,
        )


class MetricsDashboard:
    """
    CLI Metrics Dashboard for ASHA.

    Provides instant value visualization and ongoing benefit tracking.
    """

    def __init__(self) -> None:
        """Initialize metrics dashboard."""
        self.collector = MetricsCollector()
        self.session_start = datetime.now()

    def record_request(
        self,
        latency_ms: float,
        tokens_processed: int,
        tokens_saved: int,
        success: bool = True,
        threats_blocked: int = 0,
        pii_detected: int = 0,
    ) -> None:
        """Record a request."""
        self.collector.record_event("request", 1)

        if success:
            self.collector.record_event("success", 1)
        else:
            self.collector.record_event("error", 1)

        self.collector.record_event("latency", latency_ms)
        self.collector.record_event("tokens_processed", tokens_processed)
        self.collector.record_event("tokens_saved", tokens_saved)

        for _ in range(threats_blocked):
            self.collector.record_event("threat_blocked", 1)

        for _ in range(pii_detected):
            self.collector.record_event("pii_detected", 1)

    def update_system_metrics(self, memory_mb: float, cpu_percentage: float) -> None:
        """Update system metrics."""
        self.collector.set_gauge("memory_usage_mb", memory_mb)
        self.collector.set_gauge("cpu_usage_percentage", cpu_percentage)

    def print_dashboard(self, time_window: Optional[str] = None) -> None:
        """Print metrics dashboard."""
        # Parse time window
        if time_window:
            time_delta = self._parse_time_window(time_window)
        else:
            time_delta = timedelta(hours=24)  # Default to last 24 hours

        # Get metrics
        usage_stats = self.collector.get_usage_stats(time_delta)
        performance_metrics = self.collector.get_performance_metrics(time_delta)

        # Print dashboard
        print("\n" + "=" * 80)
        print("🔥 ASHA METRICS DASHBOARD")
        print("=" * 80)

        # Time window
        if time_delta.total_seconds() >= 86400:  # 24+ hours
            window_str = f"Last {time_delta.days} days"
        elif time_delta.total_seconds() >= 3600:  # 1+ hours
            window_str = f"Last {int(time_delta.total_seconds() / 3600)} hours"
        else:
            window_str = f"Last {int(time_delta.total_seconds() / 60)} minutes"

        print(f"📊 Time Window: {window_str}")
        print(f"🕐 Session Started: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print("")

        # Usage Statistics
        print("📈 USAGE STATISTICS:")
        print("-" * 40)
        print(f"  Total Requests:     {usage_stats.total_requests:,}")
        print(
            f"  Success Rate:       {(usage_stats.successful_requests / max(usage_stats.total_requests, 1)) * 100:.1f}%"
        )
        print(f"  Avg Latency:        {usage_stats.avg_latency_ms:.1f}ms")
        print(f"  Tokens Processed:   {usage_stats.total_tokens_processed:,}")
        print(f"  Tokens Saved:       {usage_stats.total_tokens_saved:,}")
        print(f"  Cost Reduction:     {usage_stats.cost_reduction_percentage:.1f}%")
        print(f"  Threats Blocked:    {usage_stats.threats_blocked:,}")
        print(f"  PII Detected:       {usage_stats.pii_detected:,}")
        print("")

        # Performance Metrics
        print("⚡ PERFORMANCE METRICS:")
        print("-" * 40)
        print(f"  Latency P50:        {performance_metrics.latency_p50:.1f}ms")
        print(f"  Latency P95:        {performance_metrics.latency_p95:.1f}ms")
        print(f"  Latency P99:        {performance_metrics.latency_p99:.1f}ms")
        print(f"  Throughput:        {performance_metrics.throughput_rps:.1f} RPS")
        print(f"  Error Rate:         {performance_metrics.error_rate * 100:.2f}%")
        print(f"  Memory Usage:       {performance_metrics.memory_usage_mb:.1f}MB")
        print(f"  CPU Usage:          {performance_metrics.cpu_usage_percentage:.1f}%")
        print("")

        # Value Summary
        total_value = self._calculate_value_summary(usage_stats)
        print("💰 VALUE SUMMARY:")
        print("-" * 40)
        print(f"  Money Saved:        ${total_value['money_saved']:.2f}")
        print(f"  Time Saved:         {total_value['time_saved']:.1f}ms")
        print(f"  Security Incidents: {total_value['security_incidents']:,} prevented")
        print(f"  ROI Score:          {total_value['roi_score']:.1f}/10")
        print("")

        # Recommendations
        recommendations = self._generate_recommendations(
            usage_stats, performance_metrics
        )
        if recommendations:
            print("💡 RECOMMENDATIONS:")
            print("-" * 40)
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
            print("")

    def print_usage_trends(self, days: int = 7) -> None:
        """Print usage trends over time."""
        print(f"\n📈 USAGE TRENDS - Last {days} Days")
        print("=" * 50)

        # Generate daily statistics
        daily_stats = []
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)

            # Filter events for this day
            day_events = [
                e for e in self.collector.events if day_start <= e.timestamp < day_end
            ]

            if day_events:
                requests = len([e for e in day_events if e.event_type == "request"])
                tokens_saved = sum(
                    e.value for e in day_events if e.event_type == "tokens_saved"
                )
                threats_blocked = len(
                    [e for e in day_events if e.event_type == "threat_blocked"]
                )

                daily_stats.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "requests": requests,
                        "tokens_saved": tokens_saved,
                        "threats_blocked": threats_blocked,
                    }
                )

        # Print trends
        if daily_stats:
            print("Date       | Requests | Tokens Saved | Threats Blocked")
            print("-" * 55)
            for stat in reversed(daily_stats):  # Show oldest first
                print(
                    f"{stat['date']} | {stat['requests']:8d} | {stat['tokens_saved']:11d} | {stat['threats_blocked']:14d}"
                )
        else:
            print("No data available for the selected period")

    def print_security_report(self) -> None:
        """Print detailed security report."""
        print("\n🔒 SECURITY REPORT")
        print("=" * 50)

        # Get all security events
        threat_events = [
            e for e in self.collector.events if e.event_type == "threat_blocked"
        ]
        pii_events = [
            e for e in self.collector.events if e.event_type == "pii_detected"
        ]

        print(f"Total Threats Blocked: {len(threat_events):,}")
        print(f"Total PII Detected: {len(pii_events):,}")
        print("")

        # Threat breakdown
        if threat_events:
            threat_types: Dict[str, int] = defaultdict(int)
            for event in threat_events:
                threat_type = event.metadata.get("threat_type", "unknown")
                threat_types[threat_type] += 1

            print("Threat Types Blocked:")
            for threat_type, count in sorted(
                threat_types.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  {threat_type}: {count:,}")
            print("")

        # PII breakdown
        if pii_events:
            pii_types: Dict[str, int] = defaultdict(int)
            for event in pii_events:
                pii_type = event.metadata.get("pii_type", "unknown")
                pii_types[pii_type] += 1

            print("PII Types Detected:")
            for pii_type, count in sorted(
                pii_types.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  {pii_type}: {count:,}")
            print("")

        # Recent security events
        print("Recent Security Events (Last 10):")
        recent_events = sorted(
            [e for e in threat_events + pii_events],
            key=lambda x: x.timestamp,
            reverse=True,
        )[:10]

        for event in recent_events:
            time_str = event.timestamp.strftime("%H:%M:%S")
            event_type = "Threat" if event.event_type == "threat_blocked" else "PII"
            details = event.metadata.get("details", "N/A")
            print(f"  {time_str} - {event_type}: {details}")

    def export_metrics(self, filename: str, format_type: str = "json") -> None:
        """Export metrics to file."""
        # Collect all metrics
        usage_stats = self.collector.get_usage_stats()
        performance_metrics = self.collector.get_performance_metrics()

        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "session_start": self.session_start.isoformat(),
            "usage_stats": asdict(usage_stats),
            "performance_metrics": asdict(performance_metrics),
            "counters": dict(self.collector.counters),
            "gauges": self.collector.gauges,
            "total_events": len(self.collector.events),
        }

        if format_type == "json":
            with open(filename, "w") as f:
                json.dump(export_data, f, indent=2)
        elif format_type == "csv":
            import csv

            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "event_type", "value", "metadata"])
                for event in self.collector.events:
                    writer.writerow(
                        [
                            event.timestamp.isoformat(),
                            event.event_type,
                            event.value,
                            json.dumps(event.metadata),
                        ]
                    )

        print(f"📁 Metrics exported to: {filename}")

    def _parse_time_window(self, window_str: str) -> timedelta:
        """Parse time window string."""
        window_str = window_str.lower()

        if window_str.endswith("h"):
            hours = int(window_str[:-1])
            return timedelta(hours=hours)
        elif window_str.endswith("d"):
            days = int(window_str[:-1])
            return timedelta(days=days)
        elif window_str.endswith("m"):
            minutes = int(window_str[:-1])
            return timedelta(minutes=minutes)
        else:
            # Default to hours
            hours = int(window_str)
            return timedelta(hours=hours)

    def _calculate_value_summary(self, usage_stats: UsageStats) -> Dict[str, Any]:
        """Calculate value summary metrics."""
        # Money saved (assuming $0.002 per token)
        money_per_token = 0.002
        money_saved = usage_stats.total_tokens_saved * money_per_token

        # Time saved (assuming 100ms per request saved through optimization)
        time_per_request = 100  # ms
        time_saved = usage_stats.total_requests * time_per_request

        # Security incidents prevented
        security_incidents = usage_stats.threats_blocked + usage_stats.pii_detected

        # ROI score (0-10 scale)
        roi_score = min(
            10,
            (
                (usage_stats.cost_reduction_percentage / 10) * 3  # Cost reduction (30%)
                + (security_incidents / max(usage_stats.total_requests, 1))
                * 100
                * 3  # Security (30%)
                + (usage_stats.successful_requests / max(usage_stats.total_requests, 1))
                * 2  # Reliability (20%)
                + (min(100, 1000 - usage_stats.avg_latency_ms) / 1000)
                * 2  # Performance (20%)
            ),
        )

        return {
            "money_saved": money_saved,
            "time_saved": time_saved,
            "security_incidents": security_incidents,
            "roi_score": roi_score,
        }

    def _generate_recommendations(
        self, usage_stats: UsageStats, performance_metrics: PerformanceMetrics
    ) -> List[str]:
        """Generate recommendations based on metrics."""
        recommendations = []

        # Performance recommendations
        if performance_metrics.latency_p95 > 500:
            recommendations.append(
                "Consider optimizing for better P95 latency (>500ms)"
            )

        if performance_metrics.error_rate > 0.05:
            recommendations.append(
                "High error rate detected - investigate failure patterns"
            )

        # Usage recommendations
        if usage_stats.cost_reduction_percentage < 10:
            recommendations.append(
                "Low token reduction - consider enabling more aggressive optimization"
            )

        if usage_stats.threats_blocked > usage_stats.total_requests * 0.1:
            recommendations.append(
                "High threat rate - consider stricter input validation"
            )

        # System recommendations
        if performance_metrics.memory_usage_mb > 1000:
            recommendations.append("High memory usage - consider memory optimization")

        if performance_metrics.cpu_usage_percentage > 80:
            recommendations.append("High CPU usage - consider scaling or optimization")

        # Value recommendations
        if usage_stats.total_requests < 100:
            recommendations.append("Low usage volume - consider wider adoption")

        return recommendations


# Global dashboard instance
dashboard = MetricsDashboard()


# Convenience functions for easy access
def record_request(
    latency_ms: float,
    tokens_processed: int,
    tokens_saved: int,
    success: bool = True,
    threats_blocked: int = 0,
    pii_detected: int = 0,
) -> None:
    """Record a request in the metrics dashboard."""
    dashboard.record_request(
        latency_ms,
        tokens_processed,
        tokens_saved,
        success,
        threats_blocked,
        pii_detected,
    )


def update_system_metrics(memory_mb: float, cpu_percentage: float) -> None:
    """Update system metrics."""
    dashboard.update_system_metrics(memory_mb, cpu_percentage)


def show_dashboard(time_window: Optional[str] = None) -> None:
    """Show metrics dashboard."""
    dashboard.print_dashboard(time_window)


def show_trends(days: int = 7) -> None:
    """Show usage trends."""
    dashboard.print_usage_trends(days)


def show_security_report() -> None:
    """Show security report."""
    dashboard.print_security_report()


def export_metrics(filename: str, format_type: str = "json") -> None:
    """Export metrics to file."""
    dashboard.export_metrics(filename, format_type)


# Quick test function
def test_metrics_dashboard() -> None:
    """Test metrics dashboard."""
    print("Testing Metrics Dashboard:")
    print("=" * 50)

    # Simulate some requests
    for i in range(10):
        record_request(
            latency_ms=45 + i * 5,
            tokens_processed=100 + i * 10,
            tokens_saved=15 + i * 2,
            success=i % 8 != 0,  # 1 failure every 8 requests
            threats_blocked=1 if i % 3 == 0 else 0,
            pii_detected=2 if i % 2 == 0 else 0,
        )

    # Update system metrics
    update_system_metrics(256.5, 45.2)

    # Show dashboard
    show_dashboard("1h")

    # Show trends
    show_trends(1)

    # Show security report
    show_security_report()


if __name__ == "__main__":
    test_metrics_dashboard()
