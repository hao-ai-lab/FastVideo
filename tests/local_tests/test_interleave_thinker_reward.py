import pytest

from fastvideo.train.methods.rl.rewards import (
    InterleavePlannerRewardScorer,
    InterleaveThinkerEditScore,
    InterleaveThinkerRewardScorer,
    extract_interleave_answer,
    extract_interleave_plan_payload,
    interleave_format_reward,
    interleave_planner_format_reward,
    score_interleave_planner_rewards,
    score_interleave_thinker_rewards,
)


def _response(success=True, refine_prompt="make it sharper"):
    return f"""
    <think>
    The generated image needs a stricter prompt.
    </think>
    <answer>
    {{'previous_step_success': {success}, 'refine_prompt': {refine_prompt!r}}}
    </answer>
    """


def test_extract_answer_accepts_upstream_single_quote_jsonish_payload():
    answer = extract_interleave_answer(_response(False, "add red highlights"))

    assert answer is not None
    assert answer.previous_step_success is False
    assert answer.refine_prompt == "add red highlights"


def test_format_reward_requires_think_before_valid_answer():
    assert interleave_format_reward(_response(True)) == 1.0

    answer_first = """
    <answer>{"previous_step_success": true, "refine_prompt": "ok"}</answer>
    <think>late reasoning</think>
    """
    assert interleave_format_reward(answer_first) == 0.0

    invalid_answer = """
    <think>reasoning</think>
    <answer>{"previous_step_success": "true", "refine_prompt": "ok"}</answer>
    """
    assert interleave_format_reward(invalid_answer) == 0.0


def test_reward_scorer_matches_upstream_default_weighting_with_absolute_edit_scores():
    scorer = InterleaveThinkerRewardScorer()

    result = scorer([{
        "response": _response(False),
        "ground_truth": {
            "success": False,
            "semantics": 6.0,
            "quality": 8.0,
        },
        "edit_score": {
            "semantics": 8.0,
            "quality": 7.0,
        },
    }])[0]

    assert result.format_reward == 1.0
    assert result.judge_accuracy_reward == 1.0
    assert result.edited_image_reward_semantic == pytest.approx(0.6)
    assert result.edited_image_reward_quality == pytest.approx(0.45)
    assert result.overall == pytest.approx(0.825)


def test_reward_scorer_uses_injected_edit_scorer_and_json_string_ground_truth():
    requests = []

    def edit_scorer(request):
        requests.append(request)
        return InterleaveThinkerEditScore(semantic_reward=0.75, quality_reward=0.25)

    scorer = InterleaveThinkerRewardScorer(format_weight=0.0, edit_scorer=edit_scorer)
    results = score_interleave_thinker_rewards(
        [{
            "response": _response(True, ""),
            "ground_truth": '{"success": true, "semantics": 4, "quality": 4}',
            "origin_prompt": "draw a glass vase",
            "previous_prompt": "a vase on a table",
            "origin_image_path": "origin.png",
        }],
        format_weight=0.0,
        edit_scorer=edit_scorer,
    )

    assert requests[0].refine_prompt == "a vase on a table"
    assert requests[0].previous_step_success is True
    assert results[0]["overall"] == pytest.approx(0.2 * 1.0 + 0.6 * 0.75 + 0.2 * 0.25)
    assert scorer([{
        "response": _response(True, ""),
        "ground_truth": {
            "success": True
        },
    }])[0].overall >= 0.0


def _planner_response(prompt="a clean cat sketch"):
    prompt = prompt.replace('"', '\\"')
    return f"""
    <think>Plan the sequence.</think>
    <answer>
    {{"execution_plan": [
      {{"step_number": 1, "step_name": "Sketch", "instruction": "Draw a cat", "prompt": "{prompt}", "auxiliary_text": null}}
    ]}}
    </answer>
    """


def test_planner_reward_accepts_execution_plan_answer_block():
    payload = extract_interleave_plan_payload(_planner_response())

    assert payload is not None
    assert interleave_planner_format_reward(_planner_response()) == 1.0
    assert interleave_planner_format_reward("<answer>{}</answer>") == 0.0


def test_planner_reward_scorer_blends_format_and_scalar_plan_score():
    scorer = InterleavePlannerRewardScorer(format_weight=0.25, fallback_plan_reward=0.1)

    result = scorer([{
        "response": _planner_response(),
        "plan_score": 0.9,
    }])[0]
    wrapped = score_interleave_planner_rewards([{
        "response": _planner_response(),
        "ground_truth": {
            "score": 0.5
        },
    }], format_weight=0.5)[0]

    assert result.format_reward == 1.0
    assert result.planner_score == pytest.approx(0.9)
    assert result.overall == pytest.approx(0.25 * 1.0 + 0.75 * 0.9)
    assert wrapped["overall"] == pytest.approx(0.5 * 1.0 + 0.5 * 0.5)
