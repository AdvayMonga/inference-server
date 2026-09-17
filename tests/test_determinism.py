"""The machine is a variable, and until now it was an unrecorded one.

Two identical A100-80GB draws measured 2.31x apart on byte-identical config. These tests pin the
two things that make that attributable next time: the device is read on every run, and the clock
lock is either taken or *recorded as refused*. No GPU is involved — a fake shell stands in, the
same way tests/test_venues.py fakes ssh.
"""

from __future__ import annotations

import json
import subprocess

from inference_server.research.compare import comparable
from inference_server.research.determinism import (
    LOCK_TOLERANCE_MHZ,
    NO_ROOT,
    DeviceState,
    lock_clocks,
    query_device,
    unlock_clocks,
)
from inference_server.research.harness import build_validity
from inference_server.research.schemas import Validity, Vitals

# One real A100 row, in the order determinism._QUERY_FIELDS asks for it.
A100_ROW = ("NVIDIA A100 80GB PCIe, GPU-8f2e1c3a-1111-2222-3333-444455556666, 550.90.07, "
            "1410, 1512, 1410, 300.00, Enabled, Enabled, 0x0000000000000000\n")


class FakeShell:
    """Answers nvidia-smi by the verb it was given. Records every argv it saw."""

    def __init__(self, *, query_out: str = A100_ROW, query_rc: int = 0,
                 pm: tuple[int, str] = (0, ""), lgc: tuple[int, str] = (0, ""),
                 missing: bool = False, sm_after_lock: int | str | None = None):
        self.calls: list[list[str]] = []
        self.query_out, self.query_rc = query_out, query_rc
        self.pm, self.lgc = pm, lgc
        self.missing = missing
        # What clocks.sm reads once -lgc has been issued. None = the card did what it was
        # told, i.e. it reads back whatever -lgc asked for.
        self.sm_after_lock = sm_after_lock
        self.locked_to: int | None = None

    def __call__(self, cmd: list[str]) -> subprocess.CompletedProcess:
        self.calls.append(cmd)
        if self.missing:
            raise FileNotFoundError("nvidia-smi")
        joined = " ".join(cmd)
        if "--query-gpu" in joined:
            out = self.query_out
            if self.locked_to is not None and self.query_rc == 0:
                sm = self.sm_after_lock if self.sm_after_lock is not None else self.locked_to
                cells = out.strip().split(",")
                cells[3] = f" {sm}"
                out = ", ".join(c.strip() for c in cells) + "\n"
            return subprocess.CompletedProcess(cmd, self.query_rc, out, "")
        if "-pm" in cmd:
            return subprocess.CompletedProcess(cmd, self.pm[0], "", self.pm[1])
        if "-lgc" in cmd:
            if self.lgc[0] == 0:
                self.locked_to = int(cmd[cmd.index("-lgc") + 1])
            return subprocess.CompletedProcess(cmd, self.lgc[0], "", self.lgc[1])
        return subprocess.CompletedProcess(cmd, 0, "", "")

    @property
    def verbs(self) -> list[str]:
        return [" ".join(c[1:]) for c in self.calls]


# ---------------------------------------------------------------- reading the device

def test_a_real_nvidia_smi_row_parses_into_every_field():
    st = query_device(FakeShell())
    assert st.gpu_name == "NVIDIA A100 80GB PCIe"
    assert st.gpu_uuid.startswith("GPU-8f2e1c3a")
    assert (st.driver_version, st.sm_clock_mhz, st.mem_clock_mhz) == ("550.90.07", 1410, 1512)
    assert (st.max_sm_clock_mhz, st.power_limit_w) == (1410, 300.0)
    assert (st.persistence_mode, st.ecc_mode) == ("Enabled", "Enabled")
    assert st.throttle_reasons == "0x0000000000000000"
    assert st.clocks_locked is False and st.host_id is None


def test_the_query_is_one_call_and_is_read_only():
    """A read that needed root would be useless: every container venue would record nothing."""
    sh = FakeShell()
    query_device(sh)
    assert len(sh.calls) == 1
    assert not ({"-pm", "-lgc", "-ac", "-pl", "-rgc"} & set(sh.calls[0]))


def test_not_available_fields_become_none_rather_than_poisoning_the_row():
    """A consumer GPU reports [N/A] for ECC and a blank power limit. Losing those must not cost
    us the GPU name, which is the field compare.py actually bars on."""
    row = "NVIDIA GeForce RTX 4090, GPU-abc, 550.54.14, 2520, 10501, [N/A], , Disabled, [N/A], "
    st = query_device(FakeShell(query_out=row))
    assert st.gpu_name == "NVIDIA GeForce RTX 4090" and st.sm_clock_mhz == 2520
    assert st.max_sm_clock_mhz is None and st.power_limit_w is None
    assert st.ecc_mode is None and st.throttle_reasons is None


def test_a_short_row_loses_only_the_fields_it_is_missing():
    st = query_device(FakeShell(query_out="NVIDIA A100 80GB PCIe, GPU-abc, 550.90.07\n"))
    assert st.gpu_name == "NVIDIA A100 80GB PCIe" and st.driver_version == "550.90.07"
    assert st.sm_clock_mhz is None and st.power_limit_w is None


def test_a_non_numeric_clock_is_none_not_an_exception():
    row = "A100, GPU-abc, 550.90.07, banana, 1512, 1410, lots, Enabled, Enabled, 0x0"
    st = query_device(FakeShell(query_out=row))
    assert st.sm_clock_mhz is None and st.power_limit_w is None and st.mem_clock_mhz == 1512


def test_no_gpu_at_all_is_an_all_none_state_and_never_raises():
    """The laptop case, and the case where a pod's driver is wedged. A raise here would turn a
    missing detail into a failed rental."""
    st = query_device(FakeShell(missing=True))
    assert st == DeviceState()
    assert query_device(FakeShell(query_rc=9, query_out="")) == DeviceState()


def test_multi_gpu_output_records_the_first_device():
    sh = FakeShell(query_out=A100_ROW + "NVIDIA A100 80GB PCIe, GPU-second, 550.90.07, "
                                        "1410, 1512, 1410, 300.00, Enabled, Enabled, 0x0\n")
    assert query_device(sh).gpu_uuid.endswith("444455556666")


# ---------------------------------------------------------------- locking, and being refused

def test_permission_denied_is_an_outcome_not_an_error():
    """The case that matters: every container venue lands here. The run must continue."""
    sh = FakeShell(pm=(3, "Insufficient Permissions to set persistence mode for GPU 00000000:01:00.0"))
    locked, why = lock_clocks(sh)
    assert locked is False and why == NO_ROOT
    assert "-lgc" not in " ".join(sh.verbs), "a refused -pm must not be followed by -lgc"


def test_a_refused_lgc_is_also_reported_rather_than_raised():
    sh = FakeShell(lgc=(3, "Permission denied"))
    assert lock_clocks(sh, sm_clock=1410) == (False, NO_ROOT)


def test_an_unrecognised_failure_names_itself():
    """Not every failure is permissions; a reason of 'requires root' would be a lie."""
    locked, why = lock_clocks(FakeShell(pm=(255, "Unable to determine the device handle")))
    assert locked is False and "255" in why and "device handle" in why


def test_an_unsupported_verb_is_not_reported_as_a_root_problem():
    """MIG parts and most GeForce cards answer -lgc with 'not supported'. Filing that under
    'requires root' would send the next reader to find a venue with root, which would not help."""
    locked, why = lock_clocks(FakeShell(lgc=(3, "Setting GPU clocks is not supported for GPU 0")),
                              sm_clock=1410)
    assert locked is False
    assert why != NO_ROOT and "not supported" in why


def test_locking_succeeds_and_says_what_it_pinned():
    sh = FakeShell()
    locked, why = lock_clocks(sh, sm_clock=1200)
    assert locked is True and "1200" in why
    assert sh.verbs[0] == "-pm 1" and sh.verbs[1] == "-lgc 1200"
    assert "--query-gpu" in sh.verbs[-1], "the pin must be read back, not assumed"


def test_with_no_clock_given_it_locks_to_the_cards_maximum():
    sh = FakeShell()
    locked, why = lock_clocks(sh)
    assert locked is True and "1410" in why           # clocks.max.sm from the A100 row
    assert sh.verbs[0] == "-pm 1" and "--query-gpu" in sh.verbs[1]
    assert sh.verbs[2] == "-lgc 1410"


def test_a_lock_that_exits_zero_without_moving_the_clock_is_not_a_lock():
    """The worst defect this module could have: -lgc can return 0 and leave the card where it
    was. Believing the exit code writes a wrong-but-plausible clocks_locked=true into the
    validity block, and every comparison downstream then trusts a control that never existed."""
    sh = FakeShell(sm_after_lock=1005)
    locked, why = lock_clocks(sh, sm_clock=1410)
    assert locked is False
    assert "read back at 1005" in why and "1410" in why


def test_a_readback_within_one_boost_bin_still_counts_as_locked():
    sh = FakeShell(sm_after_lock=1410 - LOCK_TOLERANCE_MHZ)
    assert lock_clocks(sh, sm_clock=1410)[0] is True


def test_a_clock_that_cannot_be_read_back_is_not_reported_as_locked():
    """No read-back, no claim. Silence is not confirmation."""
    locked, why = lock_clocks(FakeShell(sm_after_lock="[N/A]"), sm_clock=1410)
    assert locked is False and "could not be read back" in why


def test_an_unreadable_max_clock_refuses_rather_than_guessing():
    """Locking to a number we invented would be worse than not locking: it would look controlled."""
    sh = FakeShell(query_out="A100, GPU-abc, 550.90.07, 1410, 1512, [N/A], 300.00, x, y, z")
    locked, why = lock_clocks(sh)
    assert locked is False and "maximum SM clock" in why
    assert not any("-lgc" in v for v in sh.verbs)


def test_unlock_is_best_effort_and_swallows_a_missing_nvidia_smi():
    assert unlock_clocks(FakeShell(missing=True)) is None
    sh = FakeShell()
    unlock_clocks(sh)
    assert sh.verbs == ["-rgc"]


# ---------------------------------------------------------------- into the panel

def _validity(**kw) -> Validity:
    base = dict(harness="bench", harness_config={"pool_size": 64}, n_samples=10,
                workload_regime="mixed")
    base.update(kw)
    return Validity(engine_sha="abc123", dirty=False, run_group="grp-1", **base)


def _panel(**kw) -> Vitals:
    return Vitals(validity=_validity(**kw), tpot_p50=20.0)


def test_lock_attempted_separates_never_tried_from_tried_and_refused():
    """Both record clocks_locked=False. Only lock_attempted says which, without parsing prose."""
    assert DeviceState().lock_attempted is False
    d = DeviceState(lock_attempted=True, clocks_locked=False, lock_error=NO_ROOT)
    assert d.to_dict()["lock_attempted"] is True


def test_build_validity_picks_the_device_state_up_from_the_env(monkeypatch):
    state = DeviceState(gpu_name="NVIDIA A100 80GB PCIe", host_id="mach-7", clocks_locked=True)
    monkeypatch.setenv("RESEARCH_DEVICE_STATE", json.dumps(state.to_dict()))
    v = build_validity("bench", {"pool_size": 64}, n_samples=4, workload_regime="mixed")
    assert v.clocks_locked is True
    assert v.device_state["gpu_name"] == "NVIDIA A100 80GB PCIe"
    assert v.device_state["host_id"] == "mach-7"


def test_without_the_env_a_laptop_run_is_untouched(monkeypatch):
    monkeypatch.delenv("RESEARCH_DEVICE_STATE", raising=False)
    v = build_validity("bench", {"pool_size": 64}, n_samples=4, workload_regime="mixed")
    assert v.device_state is None and v.clocks_locked is None


def test_a_corrupt_device_state_env_is_ignored_not_fatal(monkeypatch):
    """A truncated export must not cost us a measured run."""
    monkeypatch.setenv("RESEARCH_DEVICE_STATE", '{"gpu_name": "A100"')
    v = build_validity("bench", {"pool_size": 64}, n_samples=4, workload_regime="mixed")
    assert v.device_state is None and v.clocks_locked is None


# ---------------------------------------------------------------- what compare refuses

def test_a_locked_arm_cannot_be_compared_against_an_unlocked_one():
    c = comparable(_panel(clocks_locked=True), _panel(clocks_locked=False))
    assert not c
    assert any("boost and thermal drift" in r for r in c.reasons)


def test_two_unlocked_arms_still_compare():
    """Unlocked is the normal state on a container venue; barring it would bar every GPU run."""
    assert comparable(_panel(clocks_locked=False), _panel(clocks_locked=False))


def test_different_gpu_models_are_a_bar():
    a = _panel(device_state={"gpu_name": "NVIDIA A100 80GB PCIe"})
    b = _panel(device_state={"gpu_name": "NVIDIA GeForce RTX 4090"})
    c = comparable(a, b)
    assert not c and any("different hardware" in r for r in c.reasons)


def test_a_different_host_is_a_note_not_a_bar():
    """Deliberate. Same-model-different-host IS a legal comparison — and it is the other live
    explanation for the 2.31x spread, so it has to be visible without being refused."""
    a = _panel(device_state={"gpu_name": "A100", "host_id": "mach-1"})
    b = _panel(device_state={"gpu_name": "A100", "host_id": "mach-2"})
    c = comparable(a, b)
    assert c, c.reasons
    assert any("different physical machine" in n for n in c.notes)


def test_a_panel_written_before_any_of_this_still_loads_and_still_compares():
    """Back-compat is the whole reason these fields default to None: PANEL_VERSION did not move,
    so every panel already in runs/ must behave exactly as it did."""
    old = _panel().to_dict()
    for key in ("device_state", "clocks_locked"):
        old["validity"].pop(key)
    revived = Vitals.from_dict(old)
    assert revived.validity.device_state is None and revived.validity.clocks_locked is None
    assert comparable(revived, _panel(clocks_locked=True)), "unknown is not a reason to refuse"
    assert comparable(revived, _panel(device_state={"gpu_name": "A100"}))
