"""
FluxEM Basic Usage Example

Demonstrates algebraic embeddings for arithmetic with float precision.
"""

from fluxem import create_unified_model, create_extended_ops


def main():
    # Create the unified arithmetic model
    model = create_unified_model()

    print("=== FluxEM: Algebraic Embeddings for Arithmetic ===\n")

    # Basic operations (no spaces in expressions)
    print("Basic Operations:")
    print(f"  1847 * 392   = {model.compute('1847*392')}")
    print(f"  123456 + 789 = {model.compute('123456+789')}")
    print(f"  56088 / 123  = {model.compute('56088/123')}")
    print(f"  10000 - 7777 = {model.compute('10000-7777')}")

    # Extended operations
    ops = create_extended_ops()

    print("\nExtended Operations:")
    print(f"  2^16   = {ops.power(2, 16)}")
    print(f"  sqrt(256) = {ops.sqrt(256)}")
    print(f"  exp(1) = {ops.exp(1.0):.6f}")
    print(f"  ln(e)  = {ops.ln(2.718281828):.6f}")

    # Demonstrate the mapping from arithmetic to geometry in embedding space
    print("\n=== The Key Insight ===")
    print("These results are within floating-point tolerance because arithmetic operations become")
    print("geometric operations in embedding space:")
    print("  - Addition: vector addition")
    print("  - Multiplication: addition in log-space")
    print("\nNo gradient-based training. Parameter-free arithmetic.")


if __name__ == "__main__":
    main()
