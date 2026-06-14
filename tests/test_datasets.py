from datasets.iris_cancer import make_iris_cancer
from datasets.toy_color import make_toy_color


def test_toy_color_shapes():
    train, test, input_dim, output_dim = make_toy_color(n_train=10, n_test=5)
    assert len(train) == 10
    assert len(test) == 5
    assert input_dim == 75
    assert output_dim == 2
    x, y, mask = train[0]
    assert x.shape == mask.shape
    assert y.ndim == 0


def test_iris_cancer_shapes():
    train, test, without_iris, input_dim, output_dim, names = make_iris_cancer()
    assert len(train) > 0
    assert len(test) == len(without_iris)
    assert input_dim == len(names)
    assert output_dim == 2
