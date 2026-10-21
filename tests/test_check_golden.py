from evals import check_golden as cg


def test_norm_folds_whitespace_dashes_and_case():
    assert cg.norm("AUC\u2009ROC  0.86 \u2013 F1") == "auc roc 0.86 - f1"


def test_a_line_break_hyphen_can_be_either_a_soft_hyphen_or_a_real_one():
    text = cg.variants("cur" + "\ufffe" + "rently a single" + "\ufffe" + "center study")
    assert any("currently" in t for t in text) and any("single-center" in t for t in text)


def test_every_facts_all_part_is_checked(monkeypatch):
    monkeypatch.setattr(cg, "pages", lambda f: {1: "mortality was 13.8% in the cohort"})
    monkeypatch.setattr(cg.Path, "exists", lambda self: True)
    case = {"id": "x", "source": ["A.pdf"], "facts_all": ["13.8", "10.4"]}
    assert cg.problems(case) == ["fact '10.4' not in the source PDFs"]
