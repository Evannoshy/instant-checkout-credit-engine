import pandas as pd
import pytest
import sys

from src.tabular import desc_distress_ablation as experiment
from src.tabular import preprocess


def sample():
    features = pd.DataFrame({"loan_amnt": [1, 2]}, index=pd.Index(["a", "b"], name="loan_id"))
    export = pd.DataFrame({"loan_id": ["b", "a"], "split": ["train", "train"],
                           experiment.FEATURE: [0.0, 0.3]})
    return features, export


def test_only_description_feature_is_selected_without_mutating_baseline():
    original = preprocess.load_feature_config()
    candidate = experiment.variant_config(original)
    assert len(original["model"]["numeric"] + original["model"]["categorical"]) == 9
    assert candidate["model"]["numeric"] == original["model"]["numeric"] + [experiment.FEATURE]
    assert experiment.OLD_FEATURE not in candidate["model"]["numeric"] + candidate["model"]["categorical"]


def test_join_aligns_by_id_and_preserves_zero():
    features, export = sample()
    result = experiment.join_density(features, "train", export)
    assert result.index.equals(features.index)
    assert result[experiment.FEATURE].tolist() == [0.3, 0.0]


@pytest.mark.parametrize("failure", ["missing", "duplicate", "wrong_split", "null", "infinite"])
def test_join_rejects_invalid_coverage(failure):
    features, export = sample()
    if failure == "missing":
        export = export.iloc[:1]
    elif failure == "duplicate":
        export = pd.concat([export, export.iloc[:1]])
    elif failure == "wrong_split":
        export.loc[0, "split"] = "validation"
    elif failure == "null":
        export.loc[0, experiment.FEATURE] = float("nan")
    else:
        export.loc[0, experiment.FEATURE] = float("inf")
    with pytest.raises(ValueError):
        experiment.join_density(features, "train", export)


def test_old_distress_input_is_rejected():
    features, export = sample()
    features[experiment.OLD_FEATURE] = 0.0
    with pytest.raises(ValueError, match="Unexpected distress"):
        experiment.join_density(features, "train", export)


def test_cli_refuses_training_without_lead_confirmation(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["experiment"])
    monkeypatch.setattr(experiment, "run", lambda *args: pytest.fail("training must not start"))
    with pytest.raises(SystemExit) as error:
        experiment.main()
    assert error.value.code == 2
