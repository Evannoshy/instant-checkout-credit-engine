import pandas as pd
import pytest

from src.tabular import expanded_lexical_ablation as study
from src.tabular import desc_distress_ablation as desc, preprocess


def test_feature_selection_includes_only_requested_lexical_inputs():
    base = preprocess.load_feature_config()
    config = study.full_config(base)
    columns = config["model"]["numeric"] + config["model"]["categorical"]
    assert len(columns) == 15
    assert all(column in columns for column in [desc.FEATURE, *study.EXTRA])
    assert desc.OLD_FEATURE not in columns
    assert len(base["model"]["numeric"] + base["model"]["categorical"]) == 9


def test_mapping_rejects_duplicate_or_missing_ids():
    mapping = pd.DataFrame({"loan_id": ["a", "a"], "split": ["train", "train"],
                            "role": ["base_fit", "base_fit"], "issue_month": ["2012-01", "2012-01"]})
    with pytest.raises(ValueError, match="duplicated"):
        study.check_mapping(mapping, pd.DataFrame())
    mapping.loc[1, "loan_id"] = None
    with pytest.raises(ValueError, match="missing"):
        study.check_mapping(mapping, pd.DataFrame())


def test_lightgbm_and_xgboost_share_same_preprocessing():
    config = study.full_config(preprocess.load_feature_config())
    xgb_columns = study.xgb.build_xgboost_pipeline(config, {}).named_steps["preprocess"].transformers
    challenger_columns = study.light_pipeline(config).named_steps["preprocess"].transformers
    assert xgb_columns[0][2] == challenger_columns[0][2]
    assert xgb_columns[1][2] == challenger_columns[1][2]
    assert study.LIGHTGBM_PARAMS["random_state"] == 42
