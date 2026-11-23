"""Capacity test for the MiBici prediction API.

Unlike ``locustfile.py`` (users that wait 1-2 s between requests, so the client
limits the throughput), these users send the next request as soon as the
previous one finishes. Concurrency grows in steps until the API saturates,
which gives the maximum sustainable throughput used in the FinOps analysis.

    locust -f tests/load/locustfile_capacity.py --headless \
        --host http://127.0.0.1:8000 --csv tests/load/results/capacity/capacity \
        --csv-full-history

The load shape sets the users and the duration; ``-u``, ``-r`` and ``-t`` are
ignored. Override the steps with CAPACITY_STEPS (comma-separated users) and
CAPACITY_STEP_SECONDS.
"""

import os

from locust import LoadTestShape, constant

# The directory of the locustfile is on sys.path, so the validated task of the
# base scenario is reused. Import the module (not the class) so that Locust
# does not also run MiBiciUser from this file.
import locustfile as base

STEPS = [int(users) for users in os.environ.get("CAPACITY_STEPS", "1,2,4,8,16,32").split(",")]
STEP_SECONDS = int(os.environ.get("CAPACITY_STEP_SECONDS", "30"))


class CapacityUser(base.MiBiciUser):
    """Same request and response validation, without think time."""

    wait_time = constant(0)


class StepLoadShape(LoadTestShape):
    """Hold each concurrency level for STEP_SECONDS, then stop."""

    def tick(self):
        step = int(self.get_run_time() // STEP_SECONDS)
        if step >= len(STEPS):
            return None
        users = STEPS[step]
        return users, users
