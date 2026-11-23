"""Pruebas del monitor de recursos de las pruebas de carga (Docker y Podman)."""

import pytest

from tests.load.monitor_resources import (
    docker_row,
    format_bytes,
    podman_cpu_percent,
    podman_row,
)


def podman_sample(cpu_nano, system_nano, memory=185 * 1024**2):
    return {
        "CPUNano": cpu_nano,
        "SystemNano": system_nano,
        "MemUsage": memory,
        "MemLimit": 512 * 1024**2,
        "MemPerc": memory / (512 * 1024**2) * 100,
        "NetInput": 2048,
        "NetOutput": 512,
        "BlockInput": 0,
        "BlockOutput": 0,
        "PIDs": 16,
    }


def test_docker_row_parses_percentages():
    row = docker_row(
        {
            "CPUPerc": "103.25%",
            "MemUsage": "214.6MiB / 7.607GiB",
            "MemPerc": "2.75%",
            "NetIO": "515kB / 285kB",
            "BlockIO": "26MB / 4.1kB",
            "PIDs": "19",
        }
    )
    assert row["cpu_percent"] == pytest.approx(103.25)
    assert row["memory_limit"] == "7.607GiB"
    assert row["memory_percent"] == pytest.approx(2.75)


def test_podman_cpu_uses_interval_not_lifetime_average():
    previous = podman_sample(cpu_nano=10_000_000_000, system_nano=1_000_000_000_000)
    # 1.5 s de CPU en 1 s de reloj: 150 % (un núcleo y medio).
    current = podman_sample(cpu_nano=11_500_000_000, system_nano=1_001_000_000_000)

    assert podman_cpu_percent(previous, current) == pytest.approx(150)
    assert podman_cpu_percent(current, current) == 0


def test_podman_row_matches_docker_memory_format():
    previous = podman_sample(0, 1_000_000_000)
    row = podman_row(previous, podman_sample(500_000_000, 2_000_000_000))

    assert row["cpu_percent"] == pytest.approx(50)
    assert row["memory_usage"] == "185.00MiB / 512.00MiB"
    assert row["memory_limit"] == "512.00MiB"
    assert row["network_io"] == "2.00KiB / 512B"


def test_format_bytes_uses_binary_units():
    assert format_bytes(3 * 1024**3) == "3.00GiB"
    assert format_bytes(1536) == "1.50KiB"
    assert format_bytes(10) == "10B"
