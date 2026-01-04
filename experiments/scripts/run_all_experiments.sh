#!/bin/bash
#
# Run all FluxEM experiments in sequence.
#
# This script executes the full experiment pipeline:
# 1. Data generation for all configs
# 2. Token-only baseline training
# 3. Hybrid model training
# 4. Embedding comparison (if available)
# 5. Ablation study (if available)
# 6. Evaluation
# 7. Visualization
#
# Usage:
#   ./run_all_experiments.sh [--config CONFIG] [--skip-training] [--verbose]
#
# Options:
#   --config CONFIG    Run only specified config (default: all configs)
#   --skip-training    Skip training steps (only run eval and viz)
#   --skip-data        Skip data generation (use existing data)
#   --verbose          Show detailed output
#   --help             Show this help message
#

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXPERIMENTS_DIR="$PROJECT_ROOT/experiments"
CONFIGS_DIR="$EXPERIMENTS_DIR/configs"
SCRIPTS_DIR="$EXPERIMENTS_DIR/scripts"

# Default options
SPECIFIC_CONFIG=""
SKIP_TRAINING=false
SKIP_DATA=false
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo ""
}

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

run_command() {
    local cmd="$1"
    local description="$2"

    print_step "$description"

    if [ "$VERBOSE" = true ]; then
        eval "$cmd"
    else
        eval "$cmd" > /dev/null 2>&1 || {
            print_error "Command failed: $cmd"
            return 1
        }
    fi

    print_success "$description - completed"
}

show_help() {
    head -30 "$0" | grep -E '^#' | sed 's/^# //' | sed 's/^#//'
    exit 0
}

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            SPECIFIC_CONFIG="$2"
            shift 2
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            ;;
    esac
done

# =============================================================================
# Setup
# =============================================================================

print_header "FluxEM Experiment Pipeline"

echo "Project root: $PROJECT_ROOT"
echo "Experiments: $EXPERIMENTS_DIR"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Check Python availability
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found. Please install Python 3."
    exit 1
fi

# Check if fluxem is importable
if ! python3 -c "import fluxem" 2>/dev/null; then
    print_warning "FluxEM not installed. Installing in development mode..."
    pip install -e . > /dev/null 2>&1 || {
        print_error "Failed to install FluxEM. Please run: pip install -e ."
        exit 1
    }
fi

# Check for PyYAML
if ! python3 -c "import yaml" 2>/dev/null; then
    print_warning "PyYAML not installed. Installing..."
    pip install pyyaml > /dev/null 2>&1
fi

# Get list of configs
if [ -n "$SPECIFIC_CONFIG" ]; then
    if [ -f "$CONFIGS_DIR/$SPECIFIC_CONFIG" ]; then
        CONFIGS=("$CONFIGS_DIR/$SPECIFIC_CONFIG")
    elif [ -f "$CONFIGS_DIR/${SPECIFIC_CONFIG}.yaml" ]; then
        CONFIGS=("$CONFIGS_DIR/${SPECIFIC_CONFIG}.yaml")
    else
        print_error "Config not found: $SPECIFIC_CONFIG"
        exit 1
    fi
else
    CONFIGS=("$CONFIGS_DIR"/*.yaml)
fi

echo "Configs to process: ${#CONFIGS[@]}"
for cfg in "${CONFIGS[@]}"; do
    echo "  - $(basename "$cfg")"
done
echo ""

# =============================================================================
# Step 1: Data Generation
# =============================================================================

if [ "$SKIP_DATA" = false ]; then
    print_header "Step 1: Data Generation"

    for config in "${CONFIGS[@]}"; do
        config_name=$(basename "$config" .yaml)
        print_step "Generating data for $config_name..."

        python3 "$SCRIPTS_DIR/generate_data.py" --config "$config" || {
            print_error "Data generation failed for $config_name"
            exit 1
        }

        print_success "Data generated for $config_name"
    done
else
    print_warning "Skipping data generation (--skip-data)"
fi

# =============================================================================
# Step 2: Token-Only Training
# =============================================================================

if [ "$SKIP_TRAINING" = false ]; then
    print_header "Step 2: Token-Only Baseline Training"

    if [ -f "$SCRIPTS_DIR/train_token_only.py" ]; then
        # Check if PyTorch is available
        if python3 -c "import torch" 2>/dev/null; then
            for config in "${CONFIGS[@]}"; do
                config_name=$(basename "$config" .yaml)
                print_step "Training token-only model for $config_name..."

                python3 "$SCRIPTS_DIR/train_token_only.py" --config "$config" || {
                    print_warning "Token-only training failed for $config_name (continuing...)"
                }
            done
        else
            print_warning "PyTorch not installed. Skipping token-only training."
            print_warning "Install with: pip install torch"
        fi
    else
        print_warning "train_token_only.py not found. Skipping."
    fi
else
    print_warning "Skipping training (--skip-training)"
fi

# =============================================================================
# Step 3: Hybrid Model Training
# =============================================================================

if [ "$SKIP_TRAINING" = false ]; then
    print_header "Step 3: Hybrid Model Training"

    if [ -f "$SCRIPTS_DIR/train_hybrid.py" ]; then
        if python3 -c "import torch" 2>/dev/null; then
            for config in "${CONFIGS[@]}"; do
                config_name=$(basename "$config" .yaml)
                print_step "Training hybrid model for $config_name..."

                python3 "$SCRIPTS_DIR/train_hybrid.py" --config "$config" || {
                    print_warning "Hybrid training failed for $config_name (continuing...)"
                }
            done
        else
            print_warning "PyTorch not installed. Skipping hybrid training."
        fi
    else
        print_warning "train_hybrid.py not found. Skipping."
    fi
fi

# =============================================================================
# Step 4: Embedding Comparison (Optional)
# =============================================================================

print_header "Step 4: Embedding Comparison"

if [ -f "$SCRIPTS_DIR/compare_embeddings.py" ]; then
    for config in "${CONFIGS[@]}"; do
        config_name=$(basename "$config" .yaml)
        print_step "Comparing embeddings for $config_name..."

        python3 "$SCRIPTS_DIR/compare_embeddings.py" --config "$config" 2>/dev/null || {
            print_warning "Embedding comparison not available for $config_name"
        }
    done
else
    print_warning "compare_embeddings.py not found. Skipping."
fi

# =============================================================================
# Step 5: Ablation Study (Optional)
# =============================================================================

print_header "Step 5: Ablation Study"

if [ -f "$SCRIPTS_DIR/ablation_study.py" ]; then
    for config in "${CONFIGS[@]}"; do
        config_name=$(basename "$config" .yaml)
        print_step "Running ablation study for $config_name..."

        python3 "$SCRIPTS_DIR/ablation_study.py" --config "$config" 2>/dev/null || {
            print_warning "Ablation study not available for $config_name"
        }
    done
else
    print_warning "ablation_study.py not found. Skipping."
fi

# =============================================================================
# Step 6: Evaluation
# =============================================================================

print_header "Step 6: Evaluation"

if [ -f "$SCRIPTS_DIR/eval.py" ]; then
    for config in "${CONFIGS[@]}"; do
        config_name=$(basename "$config" .yaml)
        print_step "Evaluating models for $config_name..."

        python3 "$SCRIPTS_DIR/eval.py" --config "$config" || {
            print_warning "Evaluation failed for $config_name (continuing...)"
        }
    done
else
    print_warning "eval.py not found. Skipping."
fi

# =============================================================================
# Step 7: Visualization
# =============================================================================

print_header "Step 7: Visualization"

if [ -f "$SCRIPTS_DIR/visualize_results.py" ]; then
    print_step "Generating visualizations..."

    python3 "$SCRIPTS_DIR/visualize_results.py" \
        --results-dir "$EXPERIMENTS_DIR/results" \
        --format all || {
        print_warning "Visualization failed (continuing...)"
    }
else
    print_warning "visualize_results.py not found. Skipping."
fi

# =============================================================================
# Summary
# =============================================================================

print_header "Pipeline Complete"

echo "Results saved to: $EXPERIMENTS_DIR/results/"
echo ""

# List output files
if [ -d "$EXPERIMENTS_DIR/results" ]; then
    echo "Generated files:"
    find "$EXPERIMENTS_DIR/results" -type f -name "*.json" -o -name "*.txt" -o -name "*.md" -o -name "*.png" 2>/dev/null | while read -r file; do
        echo "  - ${file#$PROJECT_ROOT/}"
    done
fi

echo ""
echo -e "${GREEN}All experiments completed successfully!${NC}"
echo ""
echo "Next steps:"
echo "  1. Review results in experiments/results/"
echo "  2. Check figures in experiments/results/figures/"
echo "  3. Copy markdown tables to README.md"
echo ""
