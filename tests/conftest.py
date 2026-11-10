"""Shared pytest setup: a headless matplotlib backend for the print-figure tests."""

import matplotlib

matplotlib.use("Agg")
