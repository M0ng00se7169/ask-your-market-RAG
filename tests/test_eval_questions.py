"""
Tests for eval/questions.json — the Phase 5 subphase 5a question set.

Validates structure, counts, required fields, per-type expected shapes, and
basic sanity (no duplicate ids, no empty questions, all required keys present).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
QUESTIONS_PATH = EVAL_DIR / "questions.json"


@pytest.fixture
def questions():
    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    return data


class TestFileLoadsAndHasTopLevelShape:
    def test_file_is_valid_json(self):
        data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_has_questions_array(self, questions):
        assert "questions" in questions
        assert isinstance(questions["questions"], list)

    def test_has_metadata_fields(self, questions):
        assert "title" in questions
        assert "description" in questions


class TestCounts:
    def test_total_count(self, questions):
        qs = questions["questions"]
        assert len(qs) == 25

    def test_exact_count(self, questions):
        qs = questions["questions"]
        exact = [q for q in qs if q["type"] == "exact"]
        assert len(exact) == 10

    def test_semantic_count(self, questions):
        qs = questions["questions"]
        semantic = [q for q in qs if q["type"] == "semantic"]
        assert len(semantic) == 8

    def test_mixed_count(self, questions):
        qs = questions["questions"]
        mixed = [q for q in qs if q["type"] == "mixed"]
        assert len(mixed) == 4

    def test_out_of_scope_count(self, questions):
        qs = questions["questions"]
        oos = [q for q in qs if q["type"] == "out_of_scope"]
        assert len(oos) == 3


class TestRequiredFields:
    REQUIRED = {"id", "type", "question", "expected"}

    def test_every_question_has_required_fields(self, questions):
        for q in questions["questions"]:
            missing = self.REQUIRED - set(q.keys())
            assert not missing, f"{q.get('id', '?')} missing {sorted(missing)}"

    def test_id_is_nonempty_string(self, questions):
        for q in questions["questions"]:
            assert isinstance(q["id"], str)
            assert q["id"].strip()

    def test_type_is_valid(self, questions):
        allowed = {"exact", "semantic", "mixed", "out_of_scope"}
        for q in questions["questions"]:
            assert q["type"] in allowed, f"{q['id']} has unknown type {q['type']!r}"

    def test_question_is_nonempty_string(self, questions):
        for q in questions["questions"]:
            assert isinstance(q["question"], str)
            assert q["question"].strip()

    def test_expected_is_dict(self, questions):
        for q in questions["questions"]:
            assert isinstance(q["expected"], dict)


class TestNoDuplicateIds:
    def test_ids_are_unique(self, questions):
        ids = [q["id"] for q in questions["questions"]]
        assert len(ids) == len(set(ids)), "duplicate question ids"


class TestExactQuestions:
    def test_exact_has_question_type(self, questions):
        exact = [q for q in questions["questions"] if q["type"] == "exact"]
        for q in exact:
            assert "question_type" in q["expected"]

    def test_exact_check_has_kind(self, questions):
        exact = [q for q in questions["questions"] if q["type"] == "exact"]
        for q in exact:
            assert "check" in q["expected"]
            assert "kind" in q["expected"]["check"]


class TestSemanticQuestions:
    def test_semantic_has_query(self, questions):
        semantic = [q for q in questions["questions"] if q["type"] == "semantic"]
        for q in semantic:
            assert "query" in q["expected"]
            assert isinstance(q["expected"]["query"], str)
            assert q["expected"]["query"].strip()

    def test_semantic_has_top_k(self, questions):
        semantic = [q for q in questions["questions"] if q["type"] == "semantic"]
        for q in semantic:
            assert "top_k" in q["expected"]
            assert isinstance(q["expected"]["top_k"], int)

    def test_semantic_check_has_kind(self, questions):
        semantic = [q for q in questions["questions"] if q["type"] == "semantic"]
        for q in semantic:
            assert "check" in q["expected"]
            assert "kind" in q["expected"]["check"]


class TestMixedQuestions:
    def test_mixed_has_tools_list(self, questions):
        mixed = [q for q in questions["questions"] if q["type"] == "mixed"]
        for q in mixed:
            assert "tools" in q["expected"]
            assert isinstance(q["expected"]["tools"], list)
            assert len(q["expected"]["tools"]) >= 2

    def test_mixed_tools_are_known(self, questions):
        known = {"query_snapshots", "search_descriptions", "get_data_summary"}
        mixed = [q for q in questions["questions"] if q["type"] == "mixed"]
        for q in mixed:
            for t in q["expected"]["tools"]:
                assert t in known, f"{q['id']} unknown tool {t!r}"

    def test_mixed_check_has_kind(self, questions):
        mixed = [q for q in questions["questions"] if q["type"] == "mixed"]
        for q in mixed:
            assert "check" in q["expected"]
            assert "kind" in q["expected"]["check"]


class TestOutOfScopeQuestions:
    def test_out_of_scope_expected_is_refusal_assert(self, questions):
        oos = [q for q in questions["questions"] if q["type"] == "out_of_scope"]
        for q in oos:
            assert "assert" in q["expected"]
            assert q["expected"]["assert"] == "refusal"

    def test_out_of_scope_has_note(self, questions):
        oos = [q for q in questions["questions"] if q["type"] == "out_of_scope"]
        for q in oos:
            assert "note" in q["expected"]
            assert isinstance(q["expected"]["note"], str)
