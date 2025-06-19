#!/usr/bin/env bash


set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

check_python_version() {
    log_info "Checking Python version..."
    
    if ! command_exists python3; then
        log_error "Python 3 is not installed. Please install Python 3.9 or higher."
        exit 1
    fi
    
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    required_version="3.9"
    
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
        log_error "Python ${python_version} is installed, but Python ${required_version}+ is required."
        log_error "Please install Python ${required_version} or higher."
        exit 1
    fi
    
    log_success "Python ${python_version} is installed and meets requirements (>= ${required_version})"
}

install_uv() {
    log_info "Checking for uv package manager..."
    
    if command_exists uv; then
        uv_version=$(uv --version | cut -d' ' -f2)
        log_success "uv ${uv_version} is already installed"
        return 0
    fi
    
    log_info "Installing uv package manager..."
    
    if command_exists curl; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command_exists wget; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        log_error "Neither curl nor wget is available. Please install one of them first."
        exit 1
    fi
    
    if [[ -f "$HOME/.bashrc" ]]; then
        source "$HOME/.bashrc"
    fi
    
    export PATH="$HOME/.local/bin:$PATH"
    
    if command_exists uv; then
        uv_version=$(uv --version | cut -d' ' -f2)
        log_success "uv ${uv_version} installed successfully"
    else
        log_error "Failed to install uv. Please install it manually."
        exit 1
    fi
}

install_dependencies() {
    log_info "Installing project dependencies with uv..."
    
    if [[ ! -f "pyproject.toml" ]]; then
        log_error "pyproject.toml not found. Are you in the project root directory?"
        exit 1
    fi
    
    if uv sync; then
        log_success "Dependencies installed successfully"
    else
        log_error "Failed to install dependencies"
        exit 1
    fi
    
    if [[ -d ".venv" ]]; then
        log_success "Virtual environment created at .venv"
    else
        log_warning "Virtual environment not found at .venv"
    fi
}

setup_precommit() {
    log_info "Setting up pre-commit hooks..."
    
    if [[ ! -f ".pre-commit-config.yaml" ]]; then
        log_warning ".pre-commit-config.yaml not found, skipping pre-commit setup"
        return 0
    fi
    
    if source .venv/bin/activate && pre-commit install; then
        log_success "Pre-commit hooks installed successfully"
    else
        log_error "Failed to install pre-commit hooks"
        exit 1
    fi
}

verify_installation() {
    log_info "Verifying installation..."
    
    if [[ -f ".venv/bin/python" ]]; then
        log_success "Virtual environment is properly set up"
    else
        log_error "Virtual environment not found or incomplete"
        exit 1
    fi
    
    source .venv/bin/activate
    
    local tools=("pytest" "black" "flake8" "mypy" "pre-commit")
    local missing_tools=()
    
    for tool in "${tools[@]}"; do
        if command_exists "$tool"; then
            log_success "$tool is available"
        else
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing development tools: ${missing_tools[*]}"
        exit 1
    fi
    
    log_success "All development tools are properly installed"
}

print_usage() {
    log_info "Development environment setup complete!"
    echo
    echo "Next steps:"
    echo "1. Activate the virtual environment:"
    echo "   source .venv/bin/activate"
    echo
    echo "2. Available development commands (using just):"
    echo "   just help                 # Show all available commands"
    echo "   just install-dev          # Install dev dependencies"
    echo "   just test                 # Run tests"
    echo "   just lint                 # Run linting"
    echo "   just format               # Format code"
    echo "   just all                  # Run format, lint, and type-check"
    echo
    echo "3. Or use traditional commands:"
    echo "   pytest                    # Run tests"
    echo "   black src/ tests/         # Format code"
    echo "   flake8 src/ tests/        # Lint code"
    echo "   mypy src/                 # Type check"
    echo
    echo "4. Install just task runner (optional but recommended):"
    echo "   curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/.local/bin"
    echo
}

main() {
    log_info "Starting Astrabot development environment bootstrap..."
    echo
    
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    
    log_info "Working directory: $(pwd)"
    echo
    
    check_python_version
    echo
    
    install_uv
    echo
    
    install_dependencies
    echo
    
    setup_precommit
    echo
    
    verify_installation
    echo
    
    print_usage
    
    log_success "Bootstrap completed successfully! 🎉"
}

trap 'log_error "Bootstrap interrupted"; exit 1' INT TERM

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
