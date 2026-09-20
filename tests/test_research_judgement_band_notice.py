"""A missing noise band must be visible at `loop judge` time, not only in the saved JSON."""

from inference_server.research.gates import Judgement
from inference_server.research.schemas import Experiment, GateResult, Hypothesis
from inference_server.research.session import format_judgement

NO_BAND = ("no noise band recorded for this situation; significance judged on the t-test alone")


def _judgement() -> Judgement:
    gates = {n: GateResult(n, True, "ok")
             for n in ("validity", "sanity", "significance", "correctness", "cost")}
    return Judgement(passed=True, gates=gates)


def _hypothesis() -> Hypothesis:
    return Hypothesis(statement="a policy change lowers p95 TTFT", gap_id="gap-1",
                      predicted_metric="ttft_p95", predicted_direction="decrease",
                      predicted_magnitude=">10%", falsification_tier=1,
                      falsification_test="simulate both arms")


def _experiment(notes: str) -> Experiment:
    return Experiment(hypothesis_id="hyp-1", engine_sha_base="abc1234", notes=notes)


def test_a_missing_band_is_announced_in_the_rendered_verdict():
    out = format_judgement(_judgement(), _hypothesis(), _experiment(NO_BAND))
    assert NO_BAND in out, "a verdict judged without a band must say so on the terminal"


def test_an_applied_band_does_not_raise_the_warning():
    out = format_judgement(_judgement(), _hypothesis(),
                           _experiment("noise band kb-1 (10 null runs) applied"))
    assert "!!" not in out


def test_no_experiment_renders_without_the_warning():
    assert "!!" not in format_judgement(_judgement(), _hypothesis(), None)
