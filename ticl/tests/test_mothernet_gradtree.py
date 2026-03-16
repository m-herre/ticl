import numpy as np

import ticl.prediction.mothernet as mothernet_prediction


class DummyGradTreeModel:
    child_model = "gradtree"

    def to(self, device):
        return self


def test_gradtree_label_offset_is_undone(monkeypatch):
    X = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    y = np.array([0, 1, 2])

    monkeypatch.setattr(
        mothernet_prediction,
        "extract_gradtree_model",
        lambda *args, **kwargs: {"dummy": True},
    )

    base_probs = np.array(
        [[0.1, 0.7, 0.2], [0.3, 0.2, 0.5]],
        dtype=float,
    )
    monkeypatch.setattr(
        mothernet_prediction,
        "predict_with_gradtree_model",
        lambda *args, **kwargs: base_probs.copy(),
    )

    classifier = mothernet_prediction.MotherNetClassifier(
        path="dummy",
        device="cpu",
        label_offset=1,
        model=DummyGradTreeModel(),
        config={"model_type": "mothernet"},
    )
    classifier.fit(X, y)

    probs = classifier.predict_proba(np.array([[2.0, 2.0], [3.0, 3.0]]))

    expected = base_probs[:, [1, 2, 0]]
    assert np.allclose(probs, expected)
