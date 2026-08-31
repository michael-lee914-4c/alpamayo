"""Memory meters must use mx.get_active_memory, not deprecated mx.metal.*."""

import warnings

from mlx_port.profiling import (
    MemoryMonitor,
    record_memory_sample,
    reset_global_memory_peak,
)


def test_record_memory_sample_no_metal_deprecation():
    reset_global_memory_peak()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sample = record_memory_sample("unit")
    metal_depr = [
        w
        for w in caught
        if issubclass(w.category, DeprecationWarning)
        and "get_active_memory" in str(w.message)
    ]
    if metal_depr:
        raise AssertionError(f"deprecated memory API: {metal_depr[0].message}")
    assert "resident" in sample


def test_memory_monitor_no_metal_deprecation():
    reset_global_memory_peak()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with MemoryMonitor(poll_interval=0.01, label="unit"):
            pass
    metal_depr = [
        w
        for w in caught
        if issubclass(w.category, DeprecationWarning)
        and "get_active_memory" in str(w.message)
    ]
    if metal_depr:
        raise AssertionError(f"deprecated memory API: {metal_depr[0].message}")
