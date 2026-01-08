#!/usr/bin/env python3
"""
Data Generation Script for FluxEM Multi-Domain Tool-Calling Training.

Generates training data for all 22 domains with tool-calling format:
{
    "id": "sample_001",
    "text": "What is 5 factorial?",
    "domain": "combinatorics",
    "tool_call": {"tool": "combinatorics_factorial", "args": [5]},
    "tool_result": 120,
    "target_text": "120",
    "target_value": 120,
    "difficulty": "easy"
}
"""

import argparse
import json
import math
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml


# =============================================================================
# Domain-Specific Sample Generators
# =============================================================================

def generate_combinatorics_sample() -> Dict[str, Any]:
    """Generate a combinatorics sample."""
    kinds = ["factorial", "ncr", "npr", "multiset"]
    kind = random.choice(kinds)

    if kind == "factorial":
        n = random.randint(1, 12)
        prompts = [
            f"What is {n} factorial?",
            f"Calculate {n}!",
            f"Compute the factorial of {n}",
        ]
        result = math.factorial(n)
        tool_call = {"tool": "combinatorics_factorial", "args": [n]}
        difficulty = "easy" if n <= 7 else "medium"

    elif kind == "ncr":
        n = random.randint(3, 15)
        k = random.randint(1, min(n, 10))
        prompts = [
            f"How many ways to choose {k} items from {n}?",
            f"What is C({n}, {k})?",
            f"Calculate {n} choose {k}",
        ]
        result = math.comb(n, k)
        tool_call = {"tool": "combinatorics_ncr", "args": [n, k]}
        difficulty = "easy" if n <= 8 else "medium"

    elif kind == "npr":
        n = random.randint(3, 12)
        k = random.randint(1, min(n, 8))
        prompts = [
            f"How many permutations of {k} items from {n}?",
            f"What is P({n}, {k})?",
            f"Calculate {n} permute {k}",
        ]
        result = math.perm(n, k)
        tool_call = {"tool": "combinatorics_npr", "args": [n, k]}
        difficulty = "medium"

    else:  # multiset
        n = random.randint(3, 8)
        k = random.randint(2, 5)
        prompts = [
            f"How many ways to select {k} items with repetition from {n} types?",
            f"Combinations with repetition: n={n}, k={k}",
        ]
        result = math.comb(n + k - 1, k)
        tool_call = {"tool": "combinatorics_multiset", "args": [n, k]}
        difficulty = "hard"

    return {
        "text": random.choice(prompts),
        "domain": "combinatorics",
        "tool_call": tool_call,
        "tool_result": result,
        "target_text": str(result),
        "target_value": result,
        "difficulty": difficulty,
    }


def generate_probability_sample() -> Dict[str, Any]:
    """Generate a probability sample."""
    kinds = ["bernoulli", "binomial", "bayes"]
    kind = random.choice(kinds)

    if kind == "bernoulli":
        p = round(random.uniform(0.1, 0.9), 2)
        x = random.choice([0, 1])
        prompts = [
            f"What is the Bernoulli PMF for p={p} and x={x}?",
            f"Bernoulli probability: p={p}, outcome={x}",
        ]
        result = p if x == 1 else 1.0 - p
        tool_call = {"tool": "probability_bernoulli_pmf", "args": [p, x]}
        difficulty = "easy"

    elif kind == "binomial":
        n = random.randint(5, 15)
        k = random.randint(0, n)
        p = round(random.uniform(0.2, 0.8), 2)
        prompts = [
            f"In {n} trials with success probability {p}, what's P(exactly {k} successes)?",
            f"Binomial PMF: n={n}, k={k}, p={p}",
        ]
        result = math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
        tool_call = {"tool": "probability_binomial_pmf", "args": [n, k, p]}
        difficulty = "medium"

    else:  # bayes
        p_a = round(random.uniform(0.01, 0.2), 3)
        p_b_given_a = round(random.uniform(0.8, 0.99), 2)
        p_b_given_not_a = round(random.uniform(0.01, 0.2), 2)
        prompts = [
            f"If P(A)={p_a}, P(B|A)={p_b_given_a}, P(B|not A)={p_b_given_not_a}, what's P(A|B)?",
            f"Apply Bayes rule: prior={p_a}, sensitivity={p_b_given_a}, false_positive={p_b_given_not_a}",
        ]
        p_b = p_b_given_a * p_a + p_b_given_not_a * (1 - p_a)
        result = (p_b_given_a * p_a) / p_b if p_b > 0 else 0
        tool_call = {"tool": "probability_bayes_rule", "args": [p_a, p_b_given_a, p_b_given_not_a]}
        difficulty = "hard"

    return {
        "text": random.choice(prompts),
        "domain": "probability",
        "tool_call": tool_call,
        "tool_result": round(result, 6),
        "target_text": f"{result:.4f}",
        "target_value": round(result, 6),
        "difficulty": difficulty,
    }


def generate_statistics_sample() -> Dict[str, Any]:
    """Generate a statistics sample."""
    kinds = ["mean", "median", "variance", "corr"]
    kind = random.choice(kinds)

    n = random.randint(4, 10)
    values = [random.randint(1, 50) for _ in range(n)]

    if kind == "mean":
        prompts = [
            f"What is the mean of {values}?",
            f"Calculate the average of {values}",
        ]
        result = sum(values) / len(values)
        tool_call = {"tool": "statistics_mean", "args": values}
        difficulty = "easy"

    elif kind == "median":
        prompts = [
            f"What is the median of {values}?",
            f"Find the middle value of {values}",
        ]
        sorted_vals = sorted(values)
        mid = len(sorted_vals) // 2
        if len(sorted_vals) % 2 == 0:
            result = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
        else:
            result = sorted_vals[mid]
        tool_call = {"tool": "statistics_median", "args": values}
        difficulty = "easy"

    elif kind == "variance":
        prompts = [
            f"What is the sample variance of {values}?",
            f"Compute variance for {values}",
        ]
        mean = sum(values) / len(values)
        result = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        tool_call = {"tool": "statistics_variance", "args": values}
        difficulty = "medium"

    else:  # corr
        x = [random.randint(1, 20) for _ in range(n)]
        y = [v + random.randint(-5, 5) for v in x]  # Correlated with noise
        prompts = [
            f"What is the correlation between {x} and {y}?",
            f"Calculate Pearson r for X={x} and Y={y}",
        ]
        mean_x, mean_y = sum(x) / n, sum(y) / n
        cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / (n - 1)
        std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) / (n - 1))
        std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y) / (n - 1))
        result = cov / (std_x * std_y) if std_x * std_y > 0 else 0
        tool_call = {"tool": "statistics_corr", "args": [x, y]}
        difficulty = "hard"

    return {
        "text": random.choice(prompts),
        "domain": "statistics",
        "tool_call": tool_call,
        "tool_result": round(result, 6),
        "target_text": f"{result:.4f}",
        "target_value": round(result, 6),
        "difficulty": difficulty,
    }


def generate_information_theory_sample() -> Dict[str, Any]:
    """Generate an information theory sample."""
    kinds = ["entropy", "cross_entropy", "kl"]
    kind = random.choice(kinds)

    # Generate probability distribution
    n = random.randint(2, 6)
    raw = [random.random() for _ in range(n)]
    total = sum(raw)
    probs = [round(p / total, 4) for p in raw]
    # Ensure sum is exactly 1
    probs[-1] = round(1.0 - sum(probs[:-1]), 4)

    if kind == "entropy":
        prompts = [
            f"What is the Shannon entropy of {probs}?",
            f"Calculate entropy for distribution {probs}",
        ]
        result = -sum(p * math.log2(p) for p in probs if p > 0)
        tool_call = {"tool": "info_entropy", "args": probs}
        difficulty = "medium"

    elif kind == "cross_entropy":
        raw2 = [random.random() for _ in range(n)]
        total2 = sum(raw2)
        q = [round(p / total2, 4) for p in raw2]
        q[-1] = round(1.0 - sum(q[:-1]), 4)
        prompts = [
            f"What is the cross-entropy H(p, q) for p={probs} and q={q}?",
            f"Calculate cross-entropy between {probs} and {q}",
        ]
        result = -sum(p * math.log2(q_i) for p, q_i in zip(probs, q) if q_i > 0)
        tool_call = {"tool": "info_cross_entropy", "args": [probs, q]}
        difficulty = "hard"

    else:  # kl
        raw2 = [random.random() for _ in range(n)]
        total2 = sum(raw2)
        q = [round(p / total2, 4) for p in raw2]
        q[-1] = round(1.0 - sum(q[:-1]), 4)
        prompts = [
            f"What is the KL divergence D_KL(p || q) for p={probs} and q={q}?",
            f"Calculate KL divergence from {probs} to {q}",
        ]
        result = sum(p * math.log2(p / q_i) for p, q_i in zip(probs, q) if p > 0 and q_i > 0)
        tool_call = {"tool": "info_kl_divergence", "args": [probs, q]}
        difficulty = "hard"

    return {
        "text": random.choice(prompts),
        "domain": "information_theory",
        "tool_call": tool_call,
        "tool_result": round(result, 6),
        "target_text": f"{result:.4f}",
        "target_value": round(result, 6),
        "difficulty": difficulty,
    }


def generate_signal_processing_sample() -> Dict[str, Any]:
    """Generate a signal processing sample."""
    kinds = ["convolution", "moving_average", "dft"]
    kind = random.choice(kinds)

    if kind == "convolution":
        signal = [random.randint(-5, 5) for _ in range(random.randint(3, 6))]
        kernel = [random.randint(1, 3) for _ in range(random.randint(2, 3))]
        prompts = [
            f"Convolve {signal} with {kernel}",
            f"Compute discrete convolution of {signal} and {kernel}",
        ]
        # Manual convolution
        result = []
        for i in range(len(signal) + len(kernel) - 1):
            acc = 0
            for j in range(len(kernel)):
                if 0 <= i - j < len(signal):
                    acc += signal[i - j] * kernel[j]
            result.append(acc)
        tool_call = {"tool": "signal_convolution", "args": [signal, kernel]}
        difficulty = "medium"

    elif kind == "moving_average":
        signal = [random.randint(1, 20) for _ in range(random.randint(5, 10))]
        window = random.randint(2, 4)
        prompts = [
            f"Compute the {window}-point moving average of {signal}",
            f"Apply moving average filter (window={window}) to {signal}",
        ]
        result = []
        for i in range(len(signal) - window + 1):
            result.append(sum(signal[i:i+window]) / window)
        tool_call = {"tool": "signal_moving_average", "args": [signal, window]}
        difficulty = "medium"

    else:  # dft
        signal = [random.choice([-1, 0, 1]) for _ in range(4)]
        prompts = [
            f"What are the DFT magnitudes of {signal}?",
            f"Compute DFT magnitude spectrum for {signal}",
        ]
        n = len(signal)
        result = []
        for k in range(n):
            real = sum(signal[t] * math.cos(2 * math.pi * k * t / n) for t in range(n))
            imag = sum(-signal[t] * math.sin(2 * math.pi * k * t / n) for t in range(n))
            result.append(round(math.sqrt(real**2 + imag**2), 4))
        tool_call = {"tool": "signal_dft_magnitude", "args": signal}
        difficulty = "hard"

    return {
        "text": random.choice(prompts),
        "domain": "signal_processing",
        "tool_call": tool_call,
        "tool_result": result,
        "target_text": str(result),
        "target_value": result,
        "difficulty": difficulty,
    }


def generate_calculus_sample() -> Dict[str, Any]:
    """Generate a calculus sample."""
    kinds = ["derivative", "integral", "evaluate"]
    kind = random.choice(kinds)

    # Generate polynomial coefficients [a0, a1, a2, ...]
    degree = random.randint(2, 4)
    coeffs = [random.randint(-5, 5) for _ in range(degree + 1)]
    while all(c == 0 for c in coeffs):
        coeffs = [random.randint(-5, 5) for _ in range(degree + 1)]

    if kind == "derivative":
        prompts = [
            f"What is the derivative of the polynomial with coefficients {coeffs}?",
            f"Differentiate the polynomial {coeffs} (a0 + a1*x + a2*x^2 + ...)",
        ]
        result = [coeffs[i] * i for i in range(1, len(coeffs))]
        tool_call = {"tool": "calculus_derivative", "args": coeffs}
        difficulty = "medium"

    elif kind == "integral":
        prompts = [
            f"What is the integral of the polynomial with coefficients {coeffs}?",
            f"Integrate the polynomial {coeffs} (constant=0)",
        ]
        result = [0.0] + [coeffs[i] / (i + 1) for i in range(len(coeffs))]
        tool_call = {"tool": "calculus_integral", "args": coeffs}
        difficulty = "medium"

    else:  # evaluate
        x = random.randint(-3, 3)
        prompts = [
            f"Evaluate the polynomial {coeffs} at x={x}",
            f"What is p({x}) for p(x) with coefficients {coeffs}?",
        ]
        result = sum(coeffs[i] * (x ** i) for i in range(len(coeffs)))
        tool_call = {"tool": "calculus_evaluate", "args": [coeffs, x]}
        difficulty = "easy"

    return {
        "text": random.choice(prompts),
        "domain": "calculus",
        "tool_call": tool_call,
        "tool_result": result,
        "target_text": str(result),
        "target_value": result,
        "difficulty": difficulty,
    }


def generate_temporal_sample() -> Dict[str, Any]:
    """Generate a temporal sample."""
    kinds = ["add_days", "diff_days", "day_of_week"]
    kind = random.choice(kinds)

    # Generate a random date in 2024-2025
    base_date = date(2024, random.randint(1, 12), random.randint(1, 28))

    if kind == "add_days":
        days = random.randint(1, 90)
        prompts = [
            f"What date is {days} days after {base_date.isoformat()}?",
            f"Add {days} days to {base_date.isoformat()}",
        ]
        result = (base_date + timedelta(days=days)).isoformat()
        tool_call = {"tool": "temporal_add_days", "args": [base_date.isoformat(), days]}
        difficulty = "easy"

    elif kind == "diff_days":
        end_date = base_date + timedelta(days=random.randint(1, 180))
        prompts = [
            f"How many days between {base_date.isoformat()} and {end_date.isoformat()}?",
            f"Calculate the difference in days from {base_date.isoformat()} to {end_date.isoformat()}",
        ]
        result = (end_date - base_date).days
        tool_call = {"tool": "temporal_diff_days", "args": [base_date.isoformat(), end_date.isoformat()]}
        difficulty = "easy"

    else:  # day_of_week
        prompts = [
            f"What day of the week is {base_date.isoformat()}?",
            f"What day is {base_date.isoformat()}?",
        ]
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        result = days[base_date.weekday()]
        tool_call = {"tool": "temporal_day_of_week", "args": [base_date.isoformat()]}
        difficulty = "easy"

    return {
        "text": random.choice(prompts),
        "domain": "temporal",
        "tool_call": tool_call,
        "tool_result": result,
        "target_text": str(result),
        "target_value": result,
        "difficulty": difficulty,
    }


def generate_finance_sample() -> Dict[str, Any]:
    """Generate a finance sample."""
    kinds = ["compound", "npv", "payment"]
    kind = random.choice(kinds)

    if kind == "compound":
        principal = random.choice([1000, 5000, 10000])
        rate = round(random.uniform(0.02, 0.10), 3)
        years = random.randint(1, 10)
        times_per_year = random.choice([1, 4, 12])
        prompts = [
            f"What is ${principal} at {rate*100:.1f}% for {years} years (compounded {'annually' if times_per_year == 1 else str(times_per_year) + 'x/year'})?",
            f"Calculate compound interest: principal={principal}, rate={rate}, years={years}, periods/year={times_per_year}",
        ]
        result = principal * (1 + rate / times_per_year) ** (times_per_year * years)
        tool_call = {"tool": "finance_compound_interest", "args": [principal, rate, years, times_per_year]}
        difficulty = "medium"

    elif kind == "npv":
        rate = round(random.uniform(0.05, 0.15), 2)
        initial = -random.choice([100, 500, 1000])
        n = random.randint(2, 5)
        cashflows = [initial] + [random.randint(20, 100) for _ in range(n)]
        prompts = [
            f"What is the NPV at {rate*100:.0f}% for cashflows {cashflows}?",
            f"Calculate net present value: rate={rate}, cashflows={cashflows}",
        ]
        result = sum(cf / (1 + rate) ** i for i, cf in enumerate(cashflows))
        tool_call = {"tool": "finance_npv", "args": [rate, cashflows]}
        difficulty = "hard"

    else:  # payment
        principal = random.choice([10000, 50000, 100000])
        rate = round(random.uniform(0.04, 0.08), 3)
        years = random.choice([5, 10, 15, 30])
        periods = 12  # Monthly
        prompts = [
            f"What is the monthly payment on a ${principal} loan at {rate*100:.1f}% for {years} years?",
            f"Calculate loan payment: principal={principal}, rate={rate}, years={years}",
        ]
        r = rate / periods
        n = years * periods
        if r > 0:
            result = principal * r * (1 + r) ** n / ((1 + r) ** n - 1)
        else:
            result = principal / n
        tool_call = {"tool": "finance_payment", "args": [principal, rate, years, periods]}
        difficulty = "hard"

    return {
        "text": random.choice(prompts),
        "domain": "finance",
        "tool_call": tool_call,
        "tool_result": round(result, 2),
        "target_text": f"{result:.2f}",
        "target_value": round(result, 2),
        "difficulty": difficulty,
    }


def generate_optimization_sample() -> Dict[str, Any]:
    """Generate an optimization sample."""
    kinds = ["least_squares", "gradient_step", "project_box"]
    kind = random.choice(kinds)

    if kind == "least_squares":
        a = [[random.randint(-5, 5) for _ in range(2)] for _ in range(2)]
        # Ensure non-singular
        while a[0][0] * a[1][1] - a[0][1] * a[1][0] == 0:
            a = [[random.randint(-5, 5) for _ in range(2)] for _ in range(2)]
        b = [random.randint(-10, 10) for _ in range(2)]
        prompts = [
            f"Solve the 2x2 system {a} x = {b}",
            f"Find x such that Ax = b where A={a}, b={b}",
        ]
        det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
        x0 = (a[1][1] * b[0] - a[0][1] * b[1]) / det
        x1 = (a[0][0] * b[1] - a[1][0] * b[0]) / det
        result = [round(x0, 4), round(x1, 4)]
        tool_call = {"tool": "optimization_least_squares_2x2", "args": [a, b]}
        difficulty = "medium"

    elif kind == "gradient_step":
        x = [round(random.uniform(-5, 5), 2) for _ in range(2)]
        grad = [round(random.uniform(-1, 1), 3) for _ in range(2)]
        lr = round(random.uniform(0.1, 0.5), 2)
        prompts = [
            f"Take a gradient step from {x} with gradient {grad} and lr={lr}",
            f"Apply gradient descent: x={x}, grad={grad}, learning_rate={lr}",
        ]
        result = [round(x[i] - lr * grad[i], 4) for i in range(2)]
        tool_call = {"tool": "optimization_gradient_step", "args": [x, grad, lr]}
        difficulty = "easy"

    else:  # project_box
        x = [round(random.uniform(-3, 3), 2) for _ in range(3)]
        lower = -1.0
        upper = 1.0
        prompts = [
            f"Project {x} into the box [{lower}, {upper}]",
            f"Apply box constraint [{lower}, {upper}] to vector {x}",
        ]
        result = [max(lower, min(upper, xi)) for xi in x]
        tool_call = {"tool": "optimization_project_box", "args": [x, lower, upper]}
        difficulty = "easy"

    return {
        "text": random.choice(prompts),
        "domain": "optimization",
        "tool_call": tool_call,
        "tool_result": result,
        "target_text": str(result),
        "target_value": result,
        "difficulty": difficulty,
    }


def generate_control_systems_sample() -> Dict[str, Any]:
    """Generate a control systems sample."""
    kinds = ["state_update", "is_stable"]
    kind = random.choice(kinds)

    if kind == "state_update":
        # Generate a stable 2x2 system
        a = [[round(random.uniform(-0.9, 0.9), 2), round(random.uniform(-0.3, 0.3), 2)],
             [round(random.uniform(-0.3, 0.3), 2), round(random.uniform(-0.9, 0.9), 2)]]
        b = [[1.0, 0.0], [0.0, 1.0]]
        x = [round(random.uniform(-5, 5), 2), round(random.uniform(-5, 5), 2)]
        u = [round(random.uniform(-1, 1), 2), round(random.uniform(-1, 1), 2)]
        prompts = [
            f"Compute state update x' = Ax + Bu for A={a}, B={b}, x={x}, u={u}",
            f"Apply discrete state equation with A={a}, x={x}, u={u}",
        ]
        x_new = [
            a[0][0] * x[0] + a[0][1] * x[1] + b[0][0] * u[0] + b[0][1] * u[1],
            a[1][0] * x[0] + a[1][1] * x[1] + b[1][0] * u[0] + b[1][1] * u[1],
        ]
        result = [round(xi, 4) for xi in x_new]
        tool_call = {"tool": "control_state_update", "args": {"A": a, "B": b, "x": x, "u": u}}
        difficulty = "medium"

    else:  # is_stable
        # Generate either stable or unstable system
        stable = random.choice([True, False])
        if stable:
            a = [[round(random.uniform(-0.8, 0.8), 2), 0.0],
                 [0.0, round(random.uniform(-0.8, 0.8), 2)]]
        else:
            a = [[round(random.uniform(1.1, 1.5), 2), 0.0],
                 [0.0, round(random.uniform(-0.8, 0.8), 2)]]
        prompts = [
            f"Is the 2x2 matrix {a} stable (discrete-time)?",
            f"Check stability of system matrix {a}",
        ]
        result = stable
        tool_call = {"tool": "control_is_stable_2x2", "args": a}
        difficulty = "medium"

    return {
        "text": random.choice(prompts),
        "domain": "control_systems",
        "tool_call": tool_call,
        "tool_result": result,
        "target_text": str(result),
        "target_value": result,
        "difficulty": difficulty,
    }


# =============================================================================
# Generator Registry
# =============================================================================

DOMAIN_GENERATORS = {
    "combinatorics": generate_combinatorics_sample,
    "probability": generate_probability_sample,
    "statistics": generate_statistics_sample,
    "information_theory": generate_information_theory_sample,
    "signal_processing": generate_signal_processing_sample,
    "calculus": generate_calculus_sample,
    "temporal": generate_temporal_sample,
    "finance": generate_finance_sample,
    "optimization": generate_optimization_sample,
    "control_systems": generate_control_systems_sample,
}


def generate_sample(domain_distribution: Dict[str, float]) -> Dict[str, Any]:
    """Generate a sample from weighted domain distribution."""
    domains = list(domain_distribution.keys())
    weights = list(domain_distribution.values())

    # Filter to generators we have
    available = [(d, w) for d, w in zip(domains, weights) if d in DOMAIN_GENERATORS]
    if not available:
        raise ValueError("No available generators for given domain distribution")

    domains, weights = zip(*available)
    total = sum(weights)
    weights = [w / total for w in weights]

    domain = random.choices(domains, weights=weights, k=1)[0]
    sample = DOMAIN_GENERATORS[domain]()
    sample["id"] = f"sample_{random.randint(0, 999999):06d}"

    return sample


def generate_dataset(
    config_path: str,
    output_dir: str,
    split: str = "train",
) -> None:
    """Generate a dataset based on config file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_config = config.get("data", {})
    size = data_config.get(f"{split}_size", 1000)
    domain_dist = data_config.get("domain_distribution", {})

    # Filter to new domains we have generators for
    domain_dist = {k: v for k, v in domain_dist.items() if k in DOMAIN_GENERATORS}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{split}.jsonl"

    print(f"Generating {size} samples for {split}...")
    print(f"Domains: {list(domain_dist.keys())}")

    domain_counts = {d: 0 for d in DOMAIN_GENERATORS}

    with open(output_file, "w") as f:
        for i in range(size):
            try:
                sample = generate_sample(domain_dist)
                domain_counts[sample["domain"]] = domain_counts.get(sample["domain"], 0) + 1
                f.write(json.dumps(sample) + "\n")
            except Exception as e:
                print(f"Warning: Failed to generate sample {i}: {e}")

            if (i + 1) % 1000 == 0:
                print(f"  Generated {i + 1}/{size} samples...")

    print(f"Done! Wrote {size} samples to {output_file}")
    print("Domain distribution:")
    for domain, count in sorted(domain_counts.items()):
        if count > 0:
            print(f"  {domain}: {count} ({count/size*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate FluxEM tool-calling training data"
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/toolcalling_multidomain.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/data/toolcalling_multidomain",
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Which splits to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    for split in args.splits:
        generate_dataset(args.config, args.output_dir, split)


if __name__ == "__main__":
    main()
