"""스모크 테스트 - 프로젝트 설정 검증."""


def test_project_imports():
    """text2sql 패키지가 정상적으로 임포트되는지 확인한다."""
    import text2sql

    assert text2sql.__version__ == "0.1.0"


def test_core_module_imports():
    """core 모듈이 정상적으로 임포트되는지 확인한다."""
    from text2sql import core

    assert core is not None

