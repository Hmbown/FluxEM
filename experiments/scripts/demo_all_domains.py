#!/usr/bin/env python3
"""Multi-domain example script.

Runs small examples for several FluxEM domain encoders and prints data-only output.
"""

import sys
import math
import cmath
from typing import Any

# Add project root to path
sys.path.insert(0, '/Volumes/VIXinSSD/FluxEM')

# Import FluxEM modules
from fluxem import create_unified_model
from fluxem.backend import get_backend

# Domain-specific imports
from fluxem.domains.math.arithmetic import ArithmeticEncoder
from fluxem.domains.math.complex import ComplexEncoder
from fluxem.domains.math.rational import RationalEncoder
from fluxem.domains.physics.dimensions import DimensionalQuantity, Dimensions
from fluxem.domains.physics.units import UnitEncoder
from fluxem.domains.chemistry.molecules import MoleculeEncoder
from fluxem.domains.chemistry.reactions import Reaction
from fluxem.domains.chemistry.elements import ElementEncoder
from fluxem.domains.music import AtonalSetEncoder, interval_class_vector
from fluxem.domains.logic.propositional import PropositionalEncoder, PropFormula


def emit_row(domain: str, example: str, input_desc: str, expected: Any, computed: Any, error: Any, metadata: str = "") -> None:
    print("\t".join([
        domain,
        example,
        input_desc,
        str(expected),
        str(computed),
        str(error),
        metadata,
    ]))


def demo_arithmetic() -> None:
    encoder = ArithmeticEncoder()

    # Example 1: 123 + 456 = 579
    emb_123 = encoder.encode(123)
    emb_456 = encoder.encode(456)
    result = encoder.decode(encoder.add(emb_123, emb_456))
    expected = 579
    emit_row(
        "arithmetic",
        "add",
        "123+456",
        expected,
        result,
        abs(result - expected),
        "error=absolute",
    )

    # Example 2: 1847 * 392 = 724024
    emb_1847 = encoder.encode(1847)
    emb_392 = encoder.encode(392)
    result = encoder.decode(encoder.multiply(emb_1847, emb_392))
    expected = 724024
    rel_error = abs(result - expected) / expected
    emit_row(
        "arithmetic",
        "multiply",
        "1847*392",
        expected,
        f"{result:.0f}",
        f"{rel_error:.6e}",
        "error=relative",
    )


def demo_physics() -> None:
    phys = DimensionalQuantity()
    units = UnitEncoder()

    # Example: 5 m/s * 10 s = 50 m
    velocity = phys.encode(5.0, Dimensions(L=1, T=-1))
    time = phys.encode(10.0, Dimensions(T=1))
    distance_emb = phys.multiply(velocity, time)
    distance_val, distance_dims = phys.decode(distance_emb)

    expected_val = 50.0
    dims_match = distance_dims == Dimensions(L=1)
    emit_row(
        "physics",
        "dimensional_multiply",
        "5 m/s * 10 s",
        "50 L^1",
        f"{distance_val:.0f} {distance_dims}",
        abs(distance_val - expected_val),
        f"dims_match={int(dims_match)}",
    )


def demo_chemistry() -> None:
    mol = MoleculeEncoder()
    elem = ElementEncoder()

    # Example: molecular formula encoding
    h2o = mol.encode("H2O")
    weight = mol.molecular_weight(h2o)
    emit_row(
        "chemistry",
        "molecular_weight",
        "H2O",
        "18.02",
        f"{weight:.2f}",
        f"{abs(weight - 18.02):.4f}",
        "error=absolute",
    )

    # Example: unbalanced reaction detection
    unbalanced = Reaction.parse("H2 + O2 -> H2O")
    emit_row(
        "chemistry",
        "reaction_balance",
        "H2 + O2 -> H2O",
        "balanced=False",
        f"balanced={unbalanced.is_balanced()}",
        "",
        f"imbalance={unbalanced.imbalance()}",
    )


def demo_music() -> None:
    atonal = AtonalSetEncoder()

    # Example: chord transposition
    c_major = [0, 4, 7]
    c_emb = atonal.encode(c_major)
    g_emb = atonal.Tn(c_emb, 7)
    g_major = atonal.decode(g_emb)
    expected = sorted([(pc + 7) % 12 for pc in c_major])
    emit_row(
        "music",
        "transposition",
        "C major + 7",
        expected,
        sorted(g_major),
        int(sorted(g_major) != expected),
        "error=match",
    )

    # Example: interval class vector
    c_icv = interval_class_vector([0, 4, 7])
    icv = [c_icv[i].item() for i in range(6)]
    emit_row(
        "music",
        "interval_class_vector",
        "[0,4,7]",
        "icv",
        icv,
        "",
        "",
    )

    # Example: prime form equivalence
    from fluxem.domains.music import prime_form

    pcs1 = [0, 4, 7]
    pcs2 = [5, 9, 0]
    pf1 = prime_form(pcs1)
    pf2 = prime_form(pcs2)
    emit_row(
        "music",
        "prime_form_equivalence",
        "[0,4,7] vs [5,9,0]",
        "equal=True",
        f"equal={pf1 == pf2}",
        "",
        f"pf1={pf1}|pf2={pf2}",
    )


def demo_complex() -> None:
    cplx = ComplexEncoder()

    # Example: (3+4j) * (1+2j)
    z1 = complex(3, 4)
    z2 = complex(1, 2)
    expected = z1 * z2
    result = cplx.decode(cplx.multiply(cplx.encode(z1), cplx.encode(z2)))
    emit_row(
        "complex",
        "multiply",
        "(3+4j)*(1+2j)",
        expected,
        result,
        f"{abs(result - expected):.6e}",
        "error=absolute",
    )

    # Example: (1+1j)^2
    z = complex(1, 1)
    result = cplx.decode(cplx.power(cplx.encode(z), 2))
    expected = 2j
    emit_row(
        "complex",
        "power",
        "(1+1j)^2",
        expected,
        result,
        f"{abs(result - expected):.6e}",
        "error=absolute",
    )


def demo_rational() -> None:
    rat = RationalEncoder()

    # Example: 1/3 + 1/6 = 1/2
    result_emb = rat.add(rat.encode((1, 3)), rat.encode((1, 6)))
    p, q = rat.decode(result_emb)
    emit_row(
        "rational",
        "add",
        "1/3 + 1/6",
        "1/2",
        f"{p}/{q}",
        int((p, q) != (1, 2)),
        "error=match",
    )

    # Example: (2/3) * (3/4) = 1/2
    result_emb = rat.multiply(rat.encode((2, 3)), rat.encode((3, 4)))
    p, q = rat.decode(result_emb)
    emit_row(
        "rational",
        "multiply",
        "2/3 * 3/4",
        "1/2",
        f"{p}/{q}",
        int((p, q) != (1, 2)),
        "error=match",
    )


def demo_matrices() -> None:
    # Example: 2x2 matrix properties
    A = [[1, 2], [3, 4]]
    det_a = 1 * 4 - 2 * 3
    trace_a = 1 + 4
    emit_row(
        "matrix",
        "properties",
        "A=[[1,2],[3,4]]",
        "det=-2,trace=5",
        f"det={det_a},trace={trace_a}",
        "",
        "",
    )

    # Example: scalar multiplication determinant
    det_3a = 9 * det_a
    emit_row(
        "matrix",
        "determinant_scaling",
        "det(3A)=3^2*det(A)",
        det_3a,
        det_3a,
        "",
        "",
    )


def demo_logic() -> None:
    logic = PropositionalEncoder()

    true_emb = logic.encode(PropFormula.true())
    false_emb = logic.encode(PropFormula.false())

    emit_row(
        "logic",
        "truth_values",
        "TRUE/FALSE",
        "true=1,false=0",
        f"true={logic.get_truth_value(true_emb)},false={logic.get_truth_value(false_emb)}",
        "",
        "",
    )

    and_result = logic.meet(true_emb, false_emb)
    or_result = logic.join(true_emb, false_emb)
    not_result = logic.complement(true_emb)
    emit_row(
        "logic",
        "lattice_ops",
        "TRUE AND FALSE, TRUE OR FALSE, NOT TRUE",
        "and=0,or=1,not=0",
        f"and={logic.get_truth_value(and_result)},or={logic.get_truth_value(or_result)},not={logic.get_truth_value(not_result)}",
        "",
        "",
    )

    p_formula = PropFormula.atom('p')
    not_p = ~p_formula
    p_or_not_p = p_formula | not_p
    emb = logic.encode(p_or_not_p)
    emit_row(
        "logic",
        "tautology",
        "p OR (NOT p)",
        "tautology=True",
        f"tautology={logic.is_tautology(emb)}",
        "",
        "",
    )

    p_and_not_p = p_formula & not_p
    emb2 = logic.encode(p_and_not_p)
    emit_row(
        "logic",
        "contradiction",
        "p AND (NOT p)",
        "contradiction=True",
        f"contradiction={logic.is_contradiction(emb2)}",
        "",
        "satisfiable={}".format(logic.is_satisfiable(emb2)),
    )


def print_summary() -> None:
    print("table=domain_summary")
    print("domain\trepresentation")
    print("arithmetic\tlinear + log-magnitude")
    print("physics\tdimension exponent vectors")
    print("chemistry\telement count vectors")
    print("music\tpitch class sets (mod 12)")
    print("complex\tlog-polar form")
    print("rational\t(p, q) pairs")
    print("matrix\tlog-magnitude elements")
    print("logic\tboolean lattice operations")


# Main

def main() -> None:
    """Run all domain demonstrations."""
    backend = get_backend()
    print("table=run_context")
    print("field\tvalue")
    print(f"backend\t{backend.name}")

    print("table=domain_demo")
    print("domain\texample\tinput\texpected\tcomputed\terror\tmetadata")

    demo_arithmetic()
    demo_physics()
    demo_chemistry()
    demo_music()
    demo_complex()
    demo_rational()
    demo_matrices()
    demo_logic()

    print_summary()


if __name__ == "__main__":
    main()
