import numpy as np
from scipy.special import expit, softmax, xlogy


def _to_prob(predictions: np.ndarray) -> np.ndarray:
    """
    Normalize predictions into a probability distribution if they aren't
    already one (mirrors baal.utils.array_utils.to_prob).
    """
    not_bounded = np.min(predictions) < 0 or np.max(predictions) > 1.0
    multiclass = predictions.shape[1] > 1
    sum_to_one = np.allclose(predictions.sum(1), 1)

    if not_bounded or (multiclass and not sum_to_one):
        if multiclass:
            predictions = softmax(predictions, axis=1)
        else:
            predictions = expit(predictions)

    return predictions


class BALD:
    """
    Bayesian Active Learning by Disagreement (BALD) acquisition score.

    Standalone reimplementation of baal.active.heuristics.BALD, covering
    only the `compute_score` path used by ALADIN for segmentation
    uncertainty (probabilities in, per-class disagreement score out).

    Reference: https://arxiv.org/abs/1703.02910
    """

    def compute_score(self, predictions: np.ndarray) -> np.ndarray:
        """
        Args:
            predictions: array of shape [n_sample, n_classes, ..., n_iterations]
                containing probabilities across ensemble members/iterations.
                Will be softmax/sigmoid-normalized if not already a valid
                distribution.

        Returns:
            Array of BALD scores with shape [n_sample, ...].
        """
        assert predictions.ndim >= 3
        predictions = _to_prob(predictions)

        # [n_sample, ...]
        expected_entropy = -np.mean(
            np.sum(xlogy(predictions, predictions), axis=1), axis=-1
        )
        expected_p = np.mean(predictions, axis=-1)  # [n_sample, n_classes, ...]
        entropy_expected_p = -np.sum(xlogy(expected_p, expected_p), axis=1)

        return entropy_expected_p - expected_entropy
