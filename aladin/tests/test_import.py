"""Tests that the aladin package (Python API + compiled C++ backend) can be imported."""


def test_import_aladin_package():
    import aladin

    assert hasattr(aladin, "ALADIN")


def test_import_cpp_backend():
    import aladin._main as cpp_backend

    # A couple of classes that the Python layer relies on directly.
    assert hasattr(cpp_backend, "Record")
    assert hasattr(cpp_backend, "Delineation")
    assert hasattr(cpp_backend, "Delineations")


def test_import_core_classes():
    from aladin import ALADIN
    from aladin.core import Record

    assert ALADIN is not None
    assert Record is not None


def test_import_backend_segmenter():
    from aladin.backend.segmenter import UNetSegmenter

    assert UNetSegmenter is not None
