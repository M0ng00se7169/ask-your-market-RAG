"""
Tests for eval/timing.py — Phase 5 sub-phase 5d business-effect timing.

Tests cover:
- Loading the 5 representative timing questions
- Manual time constants are present for all 5 questions
- Cost computation (free model = $0, missing usage = placeholder)
- Timing report structure and aggregation
- Summary markdown writer
- Edge cases: missing questions, errors during orchestration
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eval.timing import (
    load_timing_questions,
    MANUAL_TIMES,
    compute_cost,
    TIMING_QUESTION_IDS,
    _manual_time_basis,
    _manual_time_basis_general,
)


EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_orchestrator_result() -> MagicMock:
    """A sample orchestrator result with latency and usage."""
    result = MagicMock()
    result.latency_seconds = 3.5
    result.model = "nvidia/nemotron-3.5-lightning:free"
    result.usage = {
        "prompt_tokens": 150,
        "completion_tokens": 80,
        "total_tokens": 230,
    }
    result.refusal = False
    return result


# ---------------------------------------------------------------------------
# Question loading tests
# ---------------------------------------------------------------------------


class TestLoadTimingQuestions:
    def test_loads_5_questions(self):
        questions = load_timing_questions()
        assert len(questions) == 5

    def test_question_ids_match_expected(self):
        questions = load_timing_questions()
        ids = [q["id"] for q in questions]
        assert ids == TIMING_QUESTION_IDS

    def test_all_questions_have_required_fields(self):
        questions = load_timing_questions()
        for q in questions:
            assert "id" in q
            assert "type" in q
            assert "question" in q
            assert "expected" in q

    def test_question_types_are_mixed(self):
        questions = load_timing_questions()
        types = {q["type"] for q in questions}
        assert "exact" in types
        assert "semantic" in types
        assert "out_of_scope" in types

    def test_raises_on_missing_question(self, tmp_path: Path):
        """If a question ID in TIMING_QUESTION_IDS is missing from questions.json,
        the load function silently skips it (returns fewer questions)."""
        bad_questions_path = tmp_path / "bad_questions.json"
        bad_questions_path.write_text(json.dumps({
            "questions": [
                {"id": "exact-01", "type": "exact", "question": "Q1", "expected": {}},
                # missing exact-02, exact-06, semantic-02, outofscope-01
            ]
        }))
        questions = load_timing_questions(bad_questions_path)
        # Should only find exact-01
        assert len(questions) == 1
        assert questions[0]["id"] == "exact-01"


# ---------------------------------------------------------------------------
# Manual time constants tests
# ---------------------------------------------------------------------------


class TestManualTimes:
    def test_has_all_five_questions(self):
        assert set(MANUAL_TIMES.keys()) == set(TIMING_QUESTION_IDS)

    def test_all_times_are_positive(self):
        for qid, t in MANUAL_TIMES.items():
            assert t > 0, f"{qid} has non-positive time: {t}"

    def test_times_are_reasonable(self):
        """Manual times should be in a plausible range (10s to 5min)."""
        for qid, t in MANUAL_TIMES.items():
            assert 5 <= t <= 300, f"{qid} time {t}s is outside plausible range"

    def test_out_of_scope_is_fastest(self):
        """Refusal questions should be quick to answer manually."""
        oos_time = MANUAL_TIMES["outofscope-01"]
        for qid, t in MANUAL_TIMES.items():
            if qid != "outofscope-01":
                assert t >= oos_time, f"{qid} ({t}s) should be >= outofscope-01 ({oos_time}s)"


# ---------------------------------------------------------------------------
# Cost computation tests
# ---------------------------------------------------------------------------


class TestComputeCost:
    def test_free_model_cost_is_zero(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        result = compute_cost(usage, "nvidia/nemotron-3.5-lightning:free")
        assert result["cost_usd"] == 0.0
        assert result["cost_basis"] == "free tier model (no charge)"
        assert result["prompt_tokens"] == 100
        assert result["total_tokens"] == 150

    def test_missing_usage_returns_placeholder(self):
        result = compute_cost(None, "some-model")
        assert result["prompt_tokens"] is None
        assert result["total_tokens"] is None
        assert result["cost_usd"] is None
        assert result["cost_basis"] == "no usage data available"

    def test_paid_model_is_unconfigured(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        result = compute_cost(usage, "gpt-4")
        assert result["cost_usd"] is None
        assert "gpt-4" in result["cost_basis"]

    def test_model_field_preserved(self):
        result = compute_cost({"total_tokens": 100}, "test-model")
        assert result["model"] == "test-model"


# ---------------------------------------------------------------------------
# Manual time basis tests
# ---------------------------------------------------------------------------


class TestManualTimeBasis:
    def test_returns_basis_for_known_question(self):
        basis = _manual_time_basis("exact-01")
        assert "Stopwatch estimate" in basis
        assert "1.5 min" in basis

    def test_returns_basis_for_oos_question(self):
        basis = _manual_time_basis("outofscope-01")
        assert "check columns" in basis or "recognize" in basis

    def test_returns_placeholder_for_unknown_question(self):
        basis = _manual_time_basis("nonexistent-question")
        assert "placeholder" in basis.lower() or "Not measured" in basis


class TestManualTimeBasisGeneral:
    def test_is_nonempty(self):
        basis = _manual_time_basis_general()
        assert basis.strip()

    def test_mentions_stopwatch(self):
        basis = _manual_time_basis_general()
        assert "stopwatch" in basis.lower()

    def test_mentions_placeholder(self):
        basis = _manual_time_basis_general()
        assert "placeholder" in basis.lower()

    def test_mentions_transparency(self):
        basis = _manual_time_basis_general()
        assert "transparent" in basis.lower() or "basis" in basis.lower()


# ---------------------------------------------------------------------------
# Integration-style tests with mocked orchestrator
# ---------------------------------------------------------------------------


class TestRunTiming:
    def test_returns_dict_with_expected_structure(self, tmp_path: Path):
        """Test that run_timing returns a properly structured report."""
        questions_path = tmp_path / "questions.json"
        questions_path.write_text(json.dumps({
            "questions": [
                {"id": "exact-01", "type": "exact", "question": "Q1", "expected": {}},
                {"id": "exact-02", "type": "exact", "question": "Q2", "expected": {}},
                {"id": "exact-06", "type": "exact", "question": "Q3", "expected": {}},
                {"id": "semantic-02", "type": "semantic", "question": "Q4", "expected": {}},
                {"id": "outofscope-01", "type": "out_of_scope", "question": "Q5", "expected": {}},
            ]
        }))

        with patch("eval.timing.orchestrator") as mock_orch:
            mock_orch.answer.return_value = MagicMock(
                latency_seconds=2.0,
                model="test-model",
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                refusal=False,
            )

            with patch("eval.timing.load_snapshots") as mock_load:
                mock_load.return_value = MagicMock()
            with patch("eval.timing.get_collection") as mock_get:
                mock_get.return_value = MagicMock()
                mock_get.return_value.count.return_value = 20

                from eval.timing import run_timing
                report = run_timing(
                    questions_path=questions_path,
                    output_path=tmp_path / "timing.json",
                    summary_path=tmp_path / "timing.md",
                )

        assert "metadata" in report
        assert "questions" in report
        assert "summary" in report
        assert "manual_time_basis" in report
        assert len(report["questions"]) == 5

        # Check summary fields
        s = report["summary"]
        assert "total_manual_time_seconds" in s
        assert "total_assistant_time_seconds" in s
        assert "total_time_saved_seconds" in s
        assert "total_time_saved_percent" in s

    def test_aggregates_correctly(self, tmp_path: Path):
        """Test that time savings aggregate correctly."""
        questions_path = tmp_path / "questions.json"
        questions_path.write_text(json.dumps({
            "questions": [
                {"id": "exact-01", "type": "exact", "question": "Q1", "expected": {}},
                {"id": "exact-02", "type": "exact", "question": "Q2", "expected": {}},
            ]
        }))

        with patch("eval.timing.orchestrator") as mock_orch:
            mock_orch.answer.side_effect = [
                MagicMock(latency_seconds=1.0, model="m", usage={"total_tokens": 100}, refusal=False),
                MagicMock(latency_seconds=2.0, model="m", usage={"total_tokens": 200}, refusal=False),
            ]
            with patch("eval.timing.load_snapshots") as mock_load:
                mock_load.return_value = MagicMock()
            with patch("eval.timing.get_collection") as mock_get:
                mock_get.return_value = MagicMock()
                mock_get.return_value.count.return_value = 10

                from eval.timing import run_timing
                report = run_timing(
                    questions_path=questions_path,
                    output_path=tmp_path / "timing.json",
                    summary_path=tmp_path / "timing.md",
                )

        # manual: 90 + 15 = 105s, assistant: 1 + 2 = 3s, saved: 102s
        assert report["summary"]["total_manual_time_seconds"] == 105.0
        assert report["summary"]["total_assistant_time_seconds"] == 3.0
        assert report["summary"]["total_time_saved_seconds"] == 102.0

    def test_token_aggregation(self, tmp_path: Path):
        """Test that tokens aggregate across questions."""
        questions_path = tmp_path / "questions.json"
        questions_path.write_text(json.dumps({
            "questions": [
                {"id": "exact-01", "type": "exact", "question": "Q1", "expected": {}},
                {"id": "exact-02", "type": "exact", "question": "Q2", "expected": {}},
            ]
        }))

        with patch("eval.timing.orchestrator") as mock_orch:
            mock_orch.answer.side_effect = [
                MagicMock(latency_seconds=1.0, model="m", usage={"total_tokens": 100}, refusal=False),
                MagicMock(latency_seconds=1.0, model="m", usage={"total_tokens": 200}, refusal=False),
            ]
            with patch("eval.timing.load_snapshots") as mock_load:
                mock_load.return_value = MagicMock()
            with patch("eval.timing.get_collection") as mock_get:
                mock_get.return_value = MagicMock()
                mock_get.return_value.count.return_value = 10

                from eval.timing import run_timing
                report = run_timing(
                    questions_path=questions_path,
                    output_path=tmp_path / "timing.json",
                    summary_path=tmp_path / "timing.md",
                )

        assert report["summary"]["total_tokens"] == 300

    def test_error_handling(self, tmp_path: Path):
        """Test that errors during orchestration are handled gracefully."""
        questions_path = tmp_path / "questions.json"
        questions_path.write_text(json.dumps({
            "questions": [
                {"id": "exact-01", "type": "exact", "question": "Q1", "expected": {}},
                {"id": "exact-02", "type": "exact", "question": "Q2", "expected": {}},
            ]
        }))

        with patch("eval.timing.orchestrator") as mock_orch:
            # First succeeds, second fails
            mock_orch.answer.side_effect = [
                MagicMock(latency_seconds=1.0, model="m", usage={"total_tokens": 100}, refusal=False),
                RuntimeError("LLM unavailable"),
            ]
            with patch("eval.timing.load_snapshots") as mock_load:
                mock_load.return_value = MagicMock()
            with patch("eval.timing.get_collection") as mock_get:
                mock_get.return_value = MagicMock()
                mock_get.return_value.count.return_value = 10

                from eval.timing import run_timing
                report = run_timing(
                    questions_path=questions_path,
                    output_path=tmp_path / "timing.json",
                    summary_path=tmp_path / "timing.md",
                )

        assert len(report["questions"]) == 2
        # First question should have times
        assert report["questions"][0]["assistant_time_seconds"] == 1.0
        # Second question should have error
        assert "error" in report["questions"][1]
        assert report["questions"][1]["assistant_time_seconds"] is None

    def test_writes_output_files(self, tmp_path: Path):
        """Test that output files are written."""
        questions_path = tmp_path / "questions.json"
        questions_path.write_text(json.dumps({
            "questions": [
                {"id": "exact-01", "type": "exact", "question": "Q1", "expected": {}},
                {"id": "exact-02", "type": "exact", "question": "Q2", "expected": {}},
                {"id": "exact-06", "type": "exact", "question": "Q3", "expected": {}},
                {"id": "semantic-02", "type": "semantic", "question": "Q4", "expected": {}},
                {"id": "outofscope-01", "type": "out_of_scope", "question": "Q5", "expected": {}},
            ]
        }))

        output_path = tmp_path / "timing.json"
        summary_path = tmp_path / "timing.md"

        with patch("eval.timing.orchestrator") as mock_orch:
            mock_orch.answer.return_value = MagicMock(
                latency_seconds=1.0,
                model="test-model",
                usage={"total_tokens": 100},
                refusal=False,
            )
            with patch("eval.timing.load_snapshots") as mock_load:
                mock_load.return_value = MagicMock()
            with patch("eval.timing.get_collection") as mock_get:
                mock_get.return_value = MagicMock()
                mock_get.return_value.count.return_value = 10

                from eval.timing import run_timing
                run_timing(
                    questions_path=questions_path,
                    output_path=output_path,
                    summary_path=summary_path,
                )

        assert output_path.exists()
        assert summary_path.exists()

        # Verify JSON output
        data = json.loads(output_path.read_text())
        assert "metadata" in data
        assert "questions" in data
        assert "summary" in data

        # Verify markdown output
        md = summary_path.read_text()
        assert "# Ask Your Market" in md
        assert "Business-Effect Timing" in md
        assert "Method" in md


# ---------------------------------------------------------------------------
# Summary format tests
# ---------------------------------------------------------------------------


class TestSummaryFormat:
    def test_contains_question_table(self, tmp_path: Path):
        """Test that the summary includes a table of all questions."""
        questions_path = tmp_path / "questions.json"
        questions_path.write_text(json.dumps({
            "questions": [
                {"id": "exact-01", "type": "exact", "question": "Q1", "expected": {}},
                {"id": "outofscope-01", "type": "out_of_scope", "question": "Q5", "expected": {}},
            ]
        }))

        output_path = tmp_path / "timing.json"
        summary_path = tmp_path / "timing.md"

        with patch("eval.timing.orchestrator") as mock_orch:
            mock_orch.answer.return_value = MagicMock(
                latency_seconds=1.0, model="m", usage={"total_tokens": 100}, refusal=False
            )
            with patch("eval.timing.load_snapshots") as mock_load:
                mock_load.return_value = MagicMock()
            with patch("eval.timing.get_collection") as mock_get:
                mock_get.return_value = MagicMock()
                mock_get.return_value.count.return_value = 10

                from eval.timing import run_timing
                run_timing(
                    questions_path=questions_path,
                    output_path=output_path,
                    summary_path=summary_path,
                )

        md = summary_path.read_text()
        assert "exact-01" in md
        assert "outofscope-01" in md
        assert "|---|" in md  # Markdown table separator

    def test_contains_basis_statement(self, tmp_path: Path):
        """Test that the summary includes the calculation basis."""
        questions_path = tmp_path / "questions.json"
        questions_path.write_text(json.dumps({
            "questions": [
                {"id": "exact-01", "type": "exact", "question": "Q1", "expected": {}},
            ]
        }))

        output_path = tmp_path / "timing.json"
        summary_path = tmp_path / "timing.md"

        with patch("eval.timing.orchestrator") as mock_orch:
            mock_orch.answer.return_value = MagicMock(
                latency_seconds=1.0, model="m", usage={"total_tokens": 100}, refusal=False
            )
            with patch("eval.timing.load_snapshots") as mock_load:
                mock_load.return_value = MagicMock()
            with patch("eval.timing.get_collection") as mock_get:
                mock_get.return_value = MagicMock()
                mock_get.return_value.count.return_value = 10

                from eval.timing import run_timing
                run_timing(
                    questions_path=questions_path,
                    output_path=output_path,
                    summary_path=summary_path,
                )

        md = summary_path.read_text()
        assert "Calculation Basis" in md
        assert "saves" in md.lower()  # mentions the savings calculation
