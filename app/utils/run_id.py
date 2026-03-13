from datetime import datetime


def make_run_id() -> str:
    """Generate a timestamp-based run id for grouping outputs."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
