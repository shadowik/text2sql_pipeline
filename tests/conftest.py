"""Pytest 공통 fixture 설정."""

import sys
from pathlib import Path

import pytest

# src 디렉토리를 Python path에 추가
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

