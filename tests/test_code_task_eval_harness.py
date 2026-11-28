import pytest

from llm_control_plane.code_task_eval_harness import (
    CodeCandidateSubmission,
    CodeEvaluationReport,
    CodeLanguage,
    CodeProblem,
    CodeTaskEvalHarness,
    EvaluationGrade,
    TestCase,
)


def test_code_task_eval_normal_case():
    """1. Normal case: Valid Python submission passes syntax, all unit tests, and receives PERFECT grade."""
    harness = CodeTaskEvalHarness()
    problem = CodeProblem(
        problem_id="prob-fibonacci",
        title="Fibonacci Number",
        language=CodeLanguage.PYTHON,
        description="Compute the nth Fibonacci number efficiently.",
        entrypoint="fib",
        test_cases=[
            TestCase(case_id="tc-0", entrypoint="fib", args=[0], expected=0),
            TestCase(case_id="tc-1", entrypoint="fib", args=[1], expected=1),
            TestCase(case_id="tc-2", entrypoint="fib", args=[5], expected=5),
            TestCase(case_id="tc-3", entrypoint="fib", args=[10], expected=55),
        ],
        max_complexity=10,
    )
    harness.register_problem(problem)

    code = '''"""Fibonacci implementation module."""

def fib(n: int) -> int:
    """Return the nth Fibonacci number."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
'''
    submission = CodeCandidateSubmission(
        submission_id="sub-101",
        problem_id="prob-fibonacci",
        candidate_id="cand-engineer-42",
        language=CodeLanguage.PYTHON,
        source_code=code,
    )

    report = harness.evaluate_submission(submission)
    assert report.submission_id == "sub-101"
    assert report.syntax_valid is True
    assert report.total_cases == 4
    assert report.passed_cases == 4
    assert report.pass_rate_percent == 100.0
    assert report.grade == EvaluationGrade.PERFECT
    assert report.code_quality_score >= 80.0
    assert all(r.passed for r in report.case_results)


def test_code_task_eval_empty_inputs():
    """2. Empty input validation: Rejects empty IDs, titles, entrypoints, and whitespace submissions."""
    harness = CodeTaskEvalHarness()

    with pytest.raises(ValueError, match="problem_id cannot be empty"):
        harness.register_problem(
            CodeProblem(problem_id="", title="T", language=CodeLanguage.PYTHON, description="D", entrypoint="f")
        )

    with pytest.raises(ValueError, match="problem_id title cannot be empty"):
        harness.register_problem(
            CodeProblem(problem_id="p1", title="   ", language=CodeLanguage.PYTHON, description="D", entrypoint="f")
        )

    with pytest.raises(ValueError, match="entrypoint cannot be empty"):
        harness.register_problem(
            CodeProblem(problem_id="p1", title="T", language=CodeLanguage.PYTHON, description="D", entrypoint="")
        )

    # Empty submission fields
    sub_bad = CodeCandidateSubmission(
        submission_id="",
        problem_id="p1",
        candidate_id="c1",
        language=CodeLanguage.PYTHON,
        source_code="x = 1",
    )
    with pytest.raises(ValueError, match="submission_id cannot be empty"):
        harness.evaluate_submission(sub_bad)


def test_code_task_eval_malformed_syntax():
    """3. Malformed input: Detects Python SyntaxError and TypeScript bracket mismatches cleanly."""
    harness = CodeTaskEvalHarness()

    # Python syntax error
    py_ok, py_err = harness.validate_syntax("def broken_fn(x:\n  return x +", CodeLanguage.PYTHON)
    assert py_ok is False
    assert py_err is not None
    assert "SyntaxError" in py_err

    # TypeScript unmatched closing bracket
    ts_code = "function sum(a: number, b: number): number { return a + b; ]"
    ts_ok, ts_err = harness.validate_syntax(ts_code, CodeLanguage.TYPESCRIPT)
    assert ts_ok is False
    assert "Unmatched closing ']'" in str(ts_err)

    # TypeScript unclosed open bracket
    ts_code_unclosed = "function calc() { if (true) { console.log(1); }"
    ts_ok2, ts_err2 = harness.validate_syntax(ts_code_unclosed, CodeLanguage.TYPESCRIPT)
    assert ts_ok2 is False
    assert "Unclosed" in str(ts_err2)


def test_code_task_eval_boundary_values():
    """4. Boundary values: 0 test cases, 0% vs 100% pass rates, and max complexity thresholds."""
    harness = CodeTaskEvalHarness()
    problem_empty = CodeProblem(
        problem_id="prob-noop",
        title="No-op Problem",
        language=CodeLanguage.PYTHON,
        description="Problem with 0 test cases",
        entrypoint="noop",
        test_cases=[],
    )
    harness.register_problem(problem_empty)

    sub = CodeCandidateSubmission(
        submission_id="sub-boundary",
        problem_id="prob-noop",
        candidate_id="cand-001",
        language=CodeLanguage.PYTHON,
        source_code="def noop(): pass",
    )
    report = harness.evaluate_submission(sub)
    assert report.total_cases == 0
    assert report.passed_cases == 0
    assert report.pass_rate_percent == 100.0  # Vacuously 100% with 0 test cases

    # Test complexity penalty boundary
    complex_code = "def complex_fn(x):\n" + "\n".join(f"    if x == {i}: return {i}" for i in range(25))
    _, quality_score, feedback = harness.analyze_code_quality(
        complex_code, CodeLanguage.PYTHON, max_complexity=10
    )
    assert quality_score < 70.0
    assert any("High cyclomatic complexity" in f for f in feedback)


def test_code_task_eval_runtime_dependency_failure():
    """5. Dependency failure / runtime exception: Captures division by zero, missing function, or exception."""
    harness = CodeTaskEvalHarness()
    problem = CodeProblem(
        problem_id="prob-div",
        title="Safe Division",
        language=CodeLanguage.PYTHON,
        description="Divide a by b.",
        entrypoint="divide",
        test_cases=[
            TestCase(case_id="tc-1", entrypoint="divide", args=[10, 2], expected=5.0),
            TestCase(case_id="tc-2", entrypoint="divide", args=[10, 0], expected=None),
        ],
    )
    harness.register_problem(problem)

    # Buggy submission that raises ZeroDivisionError
    buggy_code = '''def divide(a, b):
    return a / b
'''
    sub = CodeCandidateSubmission(
        submission_id="sub-buggy",
        problem_id="prob-div",
        candidate_id="cand-buggy",
        language=CodeLanguage.PYTHON,
        source_code=buggy_code,
    )
    report = harness.evaluate_submission(sub)
    assert report.syntax_valid is True
    assert report.total_cases == 2
    assert report.passed_cases == 1
    assert report.case_results[0].passed is True
    assert report.case_results[1].passed is False
    assert "ZeroDivisionError" in str(report.case_results[1].error)
    assert report.grade == EvaluationGrade.PARTIAL

    # Missing entrypoint
    missing_entrypoint_code = "def other_function(): return 42"
    sub_missing = CodeCandidateSubmission(
        submission_id="sub-missing",
        problem_id="prob-div",
        candidate_id="cand-missing",
        language=CodeLanguage.PYTHON,
        source_code=missing_entrypoint_code,
    )
    report_missing = harness.evaluate_submission(sub_missing)
    assert report_missing.passed_cases == 0
    assert all("not defined or not callable" in str(r.error) for r in report_missing.case_results)


def test_code_task_eval_invariance_properties():
    """6. Property that must always hold (invariance):

    - pass_rate_percent is strictly bounded within [0.0, 100.0].
    - code_quality_score is strictly bounded within [0.0, 100.0].
    - grade is deterministic and matches score bounds.
    """
    harness = CodeTaskEvalHarness()
    problem = CodeProblem(
        problem_id="prob-inv",
        title="Invariance Check",
        language=CodeLanguage.PYTHON,
        description="Invariance problem",
        entrypoint="identity",
        test_cases=[TestCase(case_id=f"tc-{i}", entrypoint="identity", args=[i], expected=i) for i in range(10)],
    )
    harness.register_problem(problem)

    for i in range(10):
        # Submission that passes only the first i cases
        code = f"""def identity(x):
    if x < {i}:
        return x
    return -1
"""
        sub = CodeCandidateSubmission(
            submission_id=f"sub-inv-{i}",
            problem_id="prob-inv",
            candidate_id="cand-inv",
            language=CodeLanguage.PYTHON,
            source_code=code,
        )
        report: CodeEvaluationReport = harness.evaluate_submission(sub)
        assert 0.0 <= report.pass_rate_percent <= 100.0
        assert 0.0 <= report.code_quality_score <= 100.0
        assert report.passed_cases == i
        if i == 10:
            assert report.grade in (EvaluationGrade.PERFECT, EvaluationGrade.PASS)
        elif i >= 8:
            assert report.grade == EvaluationGrade.PASS
        elif i > 0:
            assert report.grade == EvaluationGrade.PARTIAL
        else:
            assert report.grade == EvaluationGrade.FAIL
