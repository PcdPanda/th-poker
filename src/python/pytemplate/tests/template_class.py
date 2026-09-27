from __future__ import annotations

from   pytemplate.cy            import CythonClass
from   pytemplate.template_class \
                                import TemplateClass
from   pytemplate.vector        import vec
import pytest


@pytest.fixture
def template_class() -> TemplateClass:
    """Fixture for TemplateClass instance with value 3."""
    return TemplateClass(3)


def test_get_data(template_class):
    """Test that get_data returns a list of repeated values."""
    result = template_class.get_data(4)
    assert result == [3, 3, 3, 3]


def test_cython(capsys):
    """Test CythonClass print and python_print methods."""
    x = 1
    cy = CythonClass(x)
    cy.print()
    captured = capsys.readouterr()
    assert "print" in captured.out
    assert cy.python_print() == x


def test_libcpp(capsys):
    """Test vec function from C++ extension."""
    vec()
    captured = capsys.readouterr()
    assert "vec" in captured.out
