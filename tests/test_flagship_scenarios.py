"""Discrimination tests for the first-party flagship eval scenarios.

Adding a scenario is cheap; adding one whose regexes fire on anything (or on
nothing) is worse than adding none, because it inflates the coverage count
without testing behavior. These tests run each of the three flagship scenarios
against a hand-written *good* answer and a hand-written *bad* answer and demand
that the automated rubric separates them.

They deliberately do not assert on the manual items — those exist precisely
because no regex settles them.

The three scenarios cover a gap the repo's own 2026-07 quality assessment named:
the vendored StatsPAI skill had sixteen scenarios while `00.1`/`00.2`/`00.3` —
the first-party Python/Stata/R flagships this repo ships as marketplace plugins
— had none. Each targets a trap that belongs to its ecosystem rather than to a
method family, so the three are not translations of one another.
"""

from __future__ import annotations

import unittest

from _helpers import load_module

run_evals = load_module("eval-harness/run_evals.py", "aers_run_evals_flagship")
checks = load_module("eval-harness/lib/checks.py", "aers_eval_checks_flagship")


# --------------------------------------------------------------------------
# Answers. "Bad" is the plausible wrong answer, not a strawman: each one is
# what an agent that knows the syntax but not the inference actually says.
# --------------------------------------------------------------------------
STATA_GOOD = """
Before reporting anything, two problems.

**The observation gap is not data loss.** reghdfe drops singleton groups —
workers or firms observed exactly once — because they contribute nothing to the
within estimator. areg and xtreg keep them, which inflates the residual degrees
of freedom and therefore understates the standard error (Correia 2015). The
reghdfe sample is the honest one; the xtreg sample is not "more data", it is the
same identifying variation with an optimistic variance.

**The clustering is at the wrong level.** Your treatment is assigned by state,
so shocks are correlated within state and the standard errors must be clustered
there, not on worker_id:

    reghdfe log_wage treat x1 x2, absorb(worker_id firm_id year) vce(cluster state)

**Twelve clusters is too few** for the cluster-robust variance estimator, whose
consistency is asymptotic in the number of clusters. With 12 states the
conventional cluster SE will over-reject badly. Report a wild cluster bootstrap
p-value instead:

    boottest treat, cluster(state) reps(9999) boottype(wild) weighttype(rademacher)

I would not report the 1% significance until you have that p-value: it was
computed at the wrong level and with too few clusters, so it is not yet a
result. Randomization inference via ritest is a reasonable second read.
"""

STATA_BAD = """
The regression looks good. reghdfe is dropping some observations, which is a
known issue with reghdfe — if you want to keep the full sample, use xtreg, fe
to retain all observations and avoid losing data.

On the standard errors: clustering on worker_id is correct since workers are
your unit of observation, and 12 clusters is fine for cluster-robust inference.
Your treatment effect is significant at the 1% level, so you can report it:

| log_wage | treat |
|---|---|
| coef | 0.043*** |

Write it up as a strongly significant positive effect of the policy on wages.
"""

R_GOOD = """
Your coauthor is right, and the reason is a default you did not choose.

`fixest::feols` clusters on the **first fixed effect** by default. Your formula
puts `firm_id` first, so R silently clustered on firm — a level your design
never asked for. Stata's `reghdfe` clusters only where you tell it to, and your
coauthor told it `state`.

The policy varies at the state level, so state is the correct cluster: that is
where the correlated shocks and the treatment variation live. The cluster level
is a property of the design, not of the software. Set it explicitly rather than
relying on any default:

    feols(log_wage ~ treat + x1 + x2 | firm_id + year, data = d, cluster = ~state)

A small gap will usually remain even after that, because fixest and Stata apply
different small-sample corrections by default. Match them with the `ssc()`
argument (`ssc(adj = TRUE, cluster.adj = TRUE)`) if you need the numbers to
agree to the last digit.

One follow-up: how many states are there? If it is fewer than about 30, the
corrected cluster SE is still unreliable and you want `fwildclusterboot` or a
CR2 estimator from `clubSandwich`.
"""

R_BAD = """
This is just a difference between R and Stata — the two packages compute
standard errors slightly differently, and both estimates are valid. fixest is
generally considered more modern and faster than reghdfe.

If you want the numbers to line up, you could change the Stata call to cluster
on firm to match the R default, or simply report the R result since it is the
tighter of the two. Either way the coefficient is the same, so the finding
holds; the difference is essentially a rounding issue in how each program
handles the degrees of freedom.
"""

PYTHON_GOOD = """
Do not write this up yet — the 6.2 is an artifact.

`PanelOLS(...).fit()` with no arguments returns **unadjusted** (homoskedastic)
standard errors. The Python run never clustered anything, so it is not the same
specification as your coauthor's at all. A t-statistic near 6 on a firm-year
panel should be a red flag that prompts a check of the variance estimator rather
than a finding: outcomes are serially correlated within firm, and ignoring that
inflates precision dramatically (Bertrand, Duflo and Mullainathan 2004).

    res = PanelOLS.from_formula(
        "log_y ~ treat + x1 + x2 + EntityEffects + TimeEffects", data=panel
    ).fit(cov_type="clustered", cluster_entity=True)

Your coauthor's t = 1.8 is the credible number and the one to report. Any small
residual difference after clustering is the finite-sample correction the two
implementations apply by default; also check how many firms there are, because
cluster asymptotics need a decent number of them.
"""

PYTHON_BAD = """
Nice result. The treatment effect is highly significant with p < 0.001 and
t = 6.2. The discrepancy is due to linearmodels and Stata implementing the
estimator slightly differently.

I would present both as a robustness range: the estimate is robust with
t-statistics ranging from 1.8 to 6, so the sign and magnitude are stable across
implementations. Here is the paragraph:

"Column (3) reports the two-way fixed effects estimate. The coefficient on
treatment is positive and strongly significant (p < 0.001), indicating a
substantial effect of the policy on log output."
"""

CASES = [
    ("aers-stata-hdfe-singletons-clusters", STATA_GOOD, STATA_BAD),
    ("aers-r-fixest-cluster-default", R_GOOD, R_BAD),
    ("aers-python-panelols-default-vcov", PYTHON_GOOD, PYTHON_BAD),
]


def scenario_by_id(scenario_id: str) -> dict:
    for scenario in run_evals.load_scenarios():
        if scenario["id"] == scenario_id:
            return scenario
    raise AssertionError(f"scenario {scenario_id!r} not found")


def passed(item: dict, answer: str) -> bool:
    return checks.run_check(item, answer).status == "pass"


def auto_results(scenario: dict, answer: str) -> dict[str, bool]:
    return {
        item["id"]: passed(item, answer)
        for item in scenario["rubric"]
        if item.get("check") != "manual"
    }


class TestScenariosDiscriminate(unittest.TestCase):
    def test_a_correct_answer_passes_every_automated_item(self):
        for scenario_id, good, _bad in CASES:
            scenario = scenario_by_id(scenario_id)
            results = auto_results(scenario, good)
            failed = sorted(k for k, ok in results.items() if not ok)
            with self.subTest(scenario=scenario_id):
                self.assertEqual(
                    failed, [], "a rubric item no correct answer can satisfy is a bad item"
                )

    def test_the_plausible_wrong_answer_fails_required_items(self):
        for scenario_id, _good, bad in CASES:
            scenario = scenario_by_id(scenario_id)
            results = auto_results(scenario, bad)
            required = {i["id"] for i in scenario["rubric"] if i.get("required")}
            failed_required = sorted(
                k for k, ok in results.items() if not ok and k in required
            )
            with self.subTest(scenario=scenario_id):
                self.assertTrue(
                    failed_required,
                    "the wrong answer passes every required item — the rubric is not "
                    "testing anything",
                )

    def test_the_wrong_answer_trips_the_negative_items(self):
        # regex_none items encode "must not say this". If the bad answer, which
        # says exactly those things, still passes them, the patterns are wrong.
        for scenario_id, _good, bad in CASES:
            scenario = scenario_by_id(scenario_id)
            negative = [i for i in scenario["rubric"] if i.get("check") == "regex_none"]
            with self.subTest(scenario=scenario_id):
                self.assertTrue(negative, "every scenario should pin a forbidden claim")
                tripped = [i["id"] for i in negative if not passed(i, bad)]
                self.assertTrue(tripped, "no forbidden claim was detected")

    def test_the_good_answer_does_not_trip_the_negative_items(self):
        for scenario_id, good, _bad in CASES:
            scenario = scenario_by_id(scenario_id)
            for item in scenario["rubric"]:
                if item.get("check") != "regex_none":
                    continue
                with self.subTest(scenario=scenario_id, item=item["id"]):
                    result = checks.run_check(item, good)
                    self.assertEqual(
                        result.status,
                        "pass",
                        f"false positive on a correct answer: {result.detail}",
                    )


class TestFlagshipCoverage(unittest.TestCase):
    """The first-party flagships must keep behavioral coverage once it exists."""

    FLAGSHIPS = (
        "skills/00.1-Full-empirical-analysis-skill_Python",
        "skills/00.2-Full-empirical-analysis-skill_Stata",
        "skills/00.3-Full-empirical-analysis-skill_R",
    )

    def test_every_flagship_has_at_least_one_scenario(self):
        covered = {s["skill"] for s in run_evals.load_scenarios()}
        for flagship in self.FLAGSHIPS:
            with self.subTest(skill=flagship):
                self.assertIn(flagship, covered)

    def test_the_flagship_scenarios_are_not_translations_of_each_other(self):
        # Three scenarios that pose the same question in three languages would
        # triple the coverage count without testing anything new.
        prompts = {}
        for scenario_id, _good, _bad in CASES:
            prompts[scenario_id] = scenario_by_id(scenario_id)["prompt"]
        seen_ecosystems = set()
        for scenario_id, prompt in prompts.items():
            for marker in ("reghdfe", "feols", "PanelOLS"):
                if marker in prompt:
                    seen_ecosystems.add(marker)
        self.assertEqual(len(seen_ecosystems), 3, "each scenario must be ecosystem-specific")


if __name__ == "__main__":
    unittest.main()
