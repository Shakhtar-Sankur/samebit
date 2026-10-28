import pytest

from samebit.backends import available, load

BACKENDS = [n for n in available() if n != "reference"]


@pytest.fixture(params=BACKENDS)
def be(request):
    return load(request.param)


@pytest.fixture
def ref():
    return load("reference")
