from pathlib import Path

import pytest

from research.calibration.collect import classification_rows, recompute_accuracy
from research.calibration.specs import CalibrationError


def test_independent_score_uses_trusted_six_decimal_precision(tmp_path):
    truth = tmp_path / 'answers.csv'
    prediction = tmp_path / 'prediction.csv'
    truth.write_text('id,label\n0,0\n1,1\n2,1\n')
    prediction.write_text('id,label\n2,0.0\n1,1.0\n0,0.0\n')
    result = recompute_accuracy(prediction, truth)
    assert result == {'correct': 2, 'rows': 3, 'exact_accuracy': 2 / 3,
                      'trusted_precision_accuracy': 0.666667}


@pytest.mark.parametrize('rows', [
    'id,label\n0,0\n0,1\n', 'id,label\n0,NaN\n', 'id,label\n0,2\n',
    'id,label\n,1\n', 'id,label\n', 'id,label,extra\n0,1,0\n',
])
def test_invalid_prediction_cannot_become_a_score(tmp_path, rows):
    path = tmp_path / 'prediction.csv'
    path.write_text(rows)
    with pytest.raises(CalibrationError):
        classification_rows(path)


def test_wrong_id_set_refused_even_if_row_count_matches(tmp_path):
    truth, prediction = tmp_path / 'truth.csv', tmp_path / 'prediction.csv'
    truth.write_text('id,label\n0,0\n1,1\n')
    prediction.write_text('id,label\n0,0\n2,1\n')
    with pytest.raises(CalibrationError, match='id_set'):
        recompute_accuracy(prediction, truth)
