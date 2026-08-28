from microserve.scorecard import run_curriculum_scorecard


def test_complete_curriculum_uses_one_correct_model_workload() -> None:
    scorecard = run_curriculum_scorecard()

    assert [stage.stage for stage in scorecard.execution] == list(range(7))
    assert all(stage.correct for stage in scorecard.execution)
    assert scorecard.execution[3].peak_kv_blocks is not None
    assert scorecard.execution[4].prefix_hits == 1
    assert scorecard.execution[6].target_calls is not None
    assert [stage.stage for stage in scorecard.system] == [7, 8]
    assert all(stage.summary.requests == 3 for stage in scorecard.system)
