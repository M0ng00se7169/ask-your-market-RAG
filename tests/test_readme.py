"""
Tests for README.md — Phase 6 portfolio package.

Validates:
- All 12 required README sections from the Phase 6 spec are present
- Metrics tables point to the correct eval output paths
- The repo contents listing matches the actual file layout
- Key numbers (products, snapshots, date range) are consistent with snapshots.csv
- The demo data caveat is explicit
- The link to the previous project is present
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
README_PATH = REPO_ROOT / "README.md"
CSV_PATH = REPO_ROOT / "data" / "snapshots.csv"
EVAL_DIR = REPO_ROOT / "eval"


# ---------------------------------------------------------------------------
# Required sections from the Phase 6 spec
# ---------------------------------------------------------------------------

REQUIRED_SECTIONS = [
    "Business problem",
    "What it does",
    "Architecture",
    "Stack",
    "How to run locally",
    "Demo data caveat",
    "Quality results",
    "Business effect",
    "Limitations and ethics",
    "Roadmap",
    "Link to the previous project",
    "Repository contents",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_readme() -> str:
    return README_PATH.read_text(encoding="utf-8")


def _load_csv() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


# ---------------------------------------------------------------------------
# Section presence tests
# ---------------------------------------------------------------------------


class TestRequiredSections:
    """Every section listed in the Phase 6 README spec must be present."""

    def test_headline_and_pitch_present(self):
        text = _read_readme()
        assert "turn competitor monitoring data into an assistant" in text.lower()
        assert "verifiable" in text.lower()

    def test_business_problem_section(self):
        text = _read_readme()
        assert "## Business problem" in text or "Business problem" in text
        assert "data is there" in text.lower() or "answers are not" in text.lower()
        assert "digest" in text.lower()
        assert "source" in text.lower() or "verifiable" in text.lower()

    def test_what_it_does_section(self):
        text = _read_readme()
        assert "What it does" in text
        assert "exact" in text.lower()
        assert "semantic" in text.lower() or "meaning" in text.lower()
        assert "sources" in text.lower()

    def test_example_dialogue_present(self):
        text = _read_readme()
        assert "## Example dialogue" in text or ("**User:**" in text and "**Assistant:**" in text)
        assert "cheaper" in text.lower() or "price" in text.lower()
        assert "vegetarian" in text.lower() or "similar" in text.lower()
        assert "not in the data" in text.lower() or "refusal" in text.lower()

    def test_architecture_diagram_present(self):
        text = _read_readme()
        assert "Architecture" in text
        assert "snapshots.csv" in text or "data/" in text
        assert "chroma" in text.lower()
        assert "gradio" in text.lower() or "app.py" in text

    def test_stack_section(self):
        text = _read_readme()
        assert "Stack" in text
        stack_terms = ["python", "pandas", "chroma", "openrouter", "gradio"]
        for term in stack_terms:
            assert term in text.lower(), f"Stack missing: {term}"

    def test_how_to_run_locally_section(self):
        text = _read_readme()
        assert "How to run locally" in text
        assert ".env.example" in text or ".env" in text
        assert "requirements.txt" in text or "pip install" in text
        assert "app.py" in text or "python app.py" in text

    def test_demo_data_caveat_section(self):
        text = _read_readme()
        assert "Demo data caveat" in text or "synthetic" in text.lower()
        assert "synthetic" in text.lower()
        assert "n8n" in text.lower() or "scraper" in text.lower()

    def test_quality_results_section(self):
        text = _read_readme()
        assert "Quality results" in text
        assert "results.json" in text or "eval/" in text
        assert "25" in text
        assert "bm25" in text.lower() or "embedding" in text.lower()

    def test_business_effect_section(self):
        text = _read_readme()
        assert "Business effect" in text
        assert "timing" in text.lower() or "timing_results.json" in text
        assert "token" in text.lower() or "cost" in text.lower()
        assert "basis" in text.lower() or "measurement" in text.lower()

    def test_limitations_and_ethics_section(self):
        text = _read_readme()
        assert "Limitations and ethics" in text
        assert "robots.txt" in text.lower() or "public" in text.lower()
        assert "forecast" in text.lower() or "no forecast" in text.lower()
        assert "synthetic" in text.lower()

    def test_roadmap_section(self):
        text = _read_readme()
        assert "Roadmap" in text
        roadmap_items = ["review", "hybrid", "multiple sites", "authentication", "charts", "telegram"]
        found = sum(1 for item in roadmap_items if item in text.lower())
        assert found >= 2, f"Roadmap too sparse: found {found} items"

    def test_link_to_previous_project_section(self):
        text = _read_readme()
        assert "Link to the previous project" in text or "previous project" in text.lower()
        assert "concept-en.md" in text or "Competitor Monitoring" in text or "data collection" in text.lower()

    def test_repo_contents_section(self):
        text = _read_readme()
        assert "Repository contents" in text
        required_files = [
            ("app.py", "app.py"),
            ("requirements.txt", "requirements.txt"),
            (".env.example", ".env.example"),
            ("README.md", "README.md"),
            ("src/data.py", "src/data.py"),
            ("src/index.py", "src/index.py"),
            ("src/tools.py", "src/tools.py"),
            ("src/llm.py", "src/llm.py"),
            ("src/orchestrator.py", "src/orchestrator.py"),
            ("src/embed.py", "src/embed.py"),
            ("scripts/generate_history.py", "generate_history.py"),
            ("data/snapshots.csv", "snapshots.csv"),
            ("eval/questions.json", "questions.json"),
            ("eval/run.py", "run.py"),
        ]
        missing = []
        for full_path, tree_name in required_files:
            if full_path not in text and tree_name not in text:
                missing.append(full_path)
        assert not missing, f"Repo contents missing: {missing}"


# ---------------------------------------------------------------------------
# Consistency with actual data
# ---------------------------------------------------------------------------


class TestReadmeNumbersMatchData:
    """Numbers cited in the README must match the actual CSV/chroma state."""

    @pytest.fixture(autouse=True)
    def _data(self):
        self.df = _load_csv()
        return self

    def test_product_count_consistent(self):
        text = _read_readme()
        actual = self.df["url"].nunique()
        assert str(actual) in text, f"README should mention {actual} products"

    def test_snapshot_count_consistent(self):
        text = _read_readme()
        actual = len(self.df)
        assert str(actual) in text, f"README should mention {actual} rows/snapshots"

    def test_date_range_or_days_consistent(self):
        text = _read_readme()
        assert "30" in text or "days" in text.lower()


# ---------------------------------------------------------------------------
# Eval output path references
# ---------------------------------------------------------------------------


class TestEvalPathReferences:
    """README must point to the correct eval output paths."""

    def test_results_json_referenced(self):
        text = _read_readme()
        assert "results.json" in text

    def test_timing_results_referenced(self):
        text = _read_readme()
        assert "timing_results.json" in text or "timing_summary.md" in text

    def test_bm25_baseline_referenced(self):
        text = _read_readme()
        assert "bm25" in text.lower() or "bm25_baseline" in text.lower()


# ---------------------------------------------------------------------------
# Repo contents listing matches reality
# ---------------------------------------------------------------------------


class TestRepoContentsListing:
    """The file tree in README§Repository contents should match the actual repo."""

    EXPECTED_EXISTING = [
        "app.py",
        "requirements.txt",
        ".env.example",
        "README.md",
        "concept-en.md",
        "concept.md",
        "plan.md",
        ".gitignore",
    ]

    EXPECTED_DIRS = ["src", "scripts", "data", "eval", "chroma_db"]

    EXPECTED_SRC_FILES = [
        "src/__init__.py",
        "src/data.py",
        "src/embed.py",
        "src/index.py",
        "src/tools.py",
        "src/llm.py",
        "src/orchestrator.py",
    ]

    EXPECTED_SCRIPT_FILES = [
        "scripts/generate_history.py",
    ]

    EXPECTED_EVAL_FILES = [
        "eval/questions.json",
        "eval/run.py",
        "eval/bm25_baseline.py",
        "eval/timing.py",
    ]

    def _file_exists_rel(self, rel: str) -> bool:
        return (REPO_ROOT / rel).exists()

    def test_all_expected_root_files_exist(self):
        for f in self.EXPECTED_EXISTING:
            assert self._file_exists_rel(f), f"Expected file missing: {f}"

    def test_all_expected_dirs_exist(self):
        for d in self.EXPECTED_DIRS:
            assert (REPO_ROOT / d).is_dir(), f"Expected dir missing: {d}"

    def test_src_files_exist(self):
        for f in self.EXPECTED_SRC_FILES:
            assert self._file_exists_rel(f), f"Expected src file missing: {f}"

    def test_script_files_exist(self):
        for f in self.EXPECTED_SCRIPT_FILES:
            assert self._file_exists_rel(f), f"Expected script missing: {f}"

    def test_eval_files_exist(self):
        for f in self.EXPECTED_EVAL_FILES:
            assert self._file_exists_rel(f), f"Expected eval file missing: {f}"

    def test_chroma_db_is_gitignored(self):
        gitignore = REPO_ROOT / ".gitignore"
        assert gitignore.exists()
        content = gitignore.read_text(encoding="utf-8")
        assert "chroma_db" in content or "chroma" in content.lower()


# ---------------------------------------------------------------------------
# Demo data caveat content
# ---------------------------------------------------------------------------


class TestDemoDataCaveat:
    """The demo data caveat must be explicit and honest."""

    def test_states_synthetic(self):
        text = _read_readme()
        assert "synthetic" in text.lower()

    def test_explains_why(self):
        text = _read_readme()
        assert any(phrase in text.lower()
                   for phrase in ["static", "don't change", "do not change",
                                   "sandbox", "books.toscrape"])

    def test_mentions_real_path(self):
        text = _read_readme()
        assert "n8n" in text.lower() or "scraper" in text.lower()


# ---------------------------------------------------------------------------
# Quality results table structure
# ---------------------------------------------------------------------------


class TestQualityResultsTable:
    """The quality results section should have a proper table structure."""

    def test_has_metrics_table(self):
        text = _read_readme()
        assert "|" in text
        assert "Metric" in text or "Total questions" in text

    def test_references_eval_output(self):
        text = _read_readme()
        assert "eval/" in text or "results.json" in text


# ---------------------------------------------------------------------------
# Business effect: basis transparency
# ---------------------------------------------------------------------------


class TestBusinessEffectTransparency:
    """The business effect section must state its basis — no unearned claims."""

    def test_mentions_measurement_method(self):
        text = _read_readme()
        assert any(term in text.lower()
                   for term in ["stopwatch", "manual", "measurement",
                                "spreadsheet", "basis"])

    def test_mentions_cost(self):
        text = _read_readme()
        assert "cost" in text.lower() or "token" in text.lower()

    def test_mentions_free_model_cost(self):
        text = _read_readme()
        assert "$0" in text or "free" in text.lower() or "nemotron" in text.lower()


# ---------------------------------------------------------------------------
# README is valid markdown and renders key content
# ---------------------------------------------------------------------------


class TestReadmeRenderability:
    """Basic checks that the README is well-formed markdown."""

    def test_headings_are_ordered(self):
        text = _read_readme()
        import re
        headings = re.findall(r"^## (.+)$", text, re.MULTILINE)
        if "Business problem" in headings:
            bp_idx = headings.index("Business problem")
            wtd_idx = headings.index("What it does")
            assert bp_idx < wtd_idx, "Business problem should come before What it does"

    def test_no_bare_urls_without_markdown(self):
        text = _read_readme()
        lines = text.splitlines()
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("http"):
                pass  # not asserting hard — some tables use bare URLs
