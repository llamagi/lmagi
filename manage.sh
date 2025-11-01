#!/bin/bash
# manage.sh - Environment management helper for lmagi
# (c) Gregory L. Magnusson MIT license 2024

VENV_DIR="venv"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if virtual environment exists
check_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        print_error "Virtual environment not found!"
        echo "Run './setup.sh' first to create the environment."
        return 1
    fi
    return 0
}

# Activate virtual environment
activate_env() {
    if check_venv; then
        source "$VENV_DIR/bin/activate"
        print_success "Virtual environment activated"
        python --version
    fi
}

# Run the main application
run_app() {
    if check_venv; then
        source "$VENV_DIR/bin/activate"
        print_header "Starting lmagi..."
        python lmagi.py
    fi
}

# Run with Ollama
run_ollama() {
    if check_venv; then
        source "$VENV_DIR/bin/activate"
        print_header "Starting lmagi with Ollama..."
        python lmagi.py
    fi
}

# Update dependencies
update_deps() {
    if check_venv; then
        source "$VENV_DIR/bin/activate"
        print_header "Updating dependencies..."
        pip install --upgrade pip
        pip install -r requirements.txt --upgrade
        print_success "Dependencies updated"
    fi
}

# Check environment
check_env() {
    print_header "Environment Status"

    # Check Python
    if command -v python3 &> /dev/null; then
        print_success "Python: $(python3 --version)"
    else
        print_error "Python not found"
    fi

    # Check venv
    if [ -d "$VENV_DIR" ]; then
        print_success "Virtual environment: exists"
    else
        print_error "Virtual environment: not found"
    fi

    # Check .env
    if [ -f ".env" ]; then
        print_success ".env file: exists"
        echo "  Configured APIs:"
        grep -E "^[A-Z_]+_API_KEY=" .env | sed 's/=.*/=***/' | sed 's/^/  - /'
    else
        print_warning ".env file: not found"
    fi

    # Check directories
    echo ""
    echo "Memory directories:"
    for dir in memory/stm memory/logs memory/truth; do
        if [ -d "$dir" ]; then
            print_success "$dir"
        else
            print_warning "$dir (missing)"
        fi
    done

    # Check installed packages
    if check_venv; then
        source "$VENV_DIR/bin/activate"
        echo ""
        echo "Key packages installed:"
        pip list | grep -E "openai|groq|together|ai71|nicegui|aiohttp" | sed 's/^/  /'
    fi
}

# Clean environment
clean_env() {
    print_header "Cleaning environment..."

    echo "This will remove:"
    echo "  - Virtual environment ($VENV_DIR)"
    echo "  - Python cache files (__pycache__, *.pyc)"
    echo ""
    read -p "Continue? (y/N) " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
        find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
        find . -type f -name "*.pyc" -delete 2>/dev/null
        print_success "Environment cleaned"
        echo "Run './setup.sh' to recreate the environment"
    else
        print_warning "Cleaning cancelled"
    fi
}

# Test installation
test_install() {
    if check_venv; then
        source "$VENV_DIR/bin/activate"
        print_header "Testing installation..."

        python3 << 'EOF'
import sys
print(f"Python version: {sys.version}")
print("\nTesting imports...")

packages = [
    "openai",
    "groq",
    "together",
    "ai71",
    "nicegui",
    "aiohttp",
    "ujson",
    "psutil",
    "dotenv"
]

failed = []
for pkg in packages:
    try:
        __import__(pkg.replace("-", "_"))
        print(f"  ✓ {pkg}")
    except ImportError:
        print(f"  ✗ {pkg} - FAILED")
        failed.append(pkg)

if failed:
    print(f"\n{len(failed)} package(s) failed to import")
    sys.exit(1)
else:
    print("\n✓ All packages imported successfully!")
EOF

        if [ $? -eq 0 ]; then
            print_success "Installation test passed"
        else
            print_error "Installation test failed"
        fi
    fi
}

# Show help
show_help() {
    print_header "lmagi Management Tool"
    echo ""
    echo "Usage: ./manage.sh [command]"
    echo ""
    echo "Commands:"
    echo "  run          - Run the main application"
    echo "  ollama       - Run with Ollama integration"
    echo "  check        - Check environment status"
    echo "  update       - Update dependencies"
    echo "  test         - Test package installation"
    echo "  clean        - Clean environment (removes venv)"
    echo "  help         - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./manage.sh run"
    echo "  ./manage.sh check"
    echo "  ./manage.sh update"
}

# Main script
case "$1" in
    run)
        run_app
        ;;
    ollama)
        run_ollama
        ;;
    check)
        check_env
        ;;
    update)
        update_deps
        ;;
    test)
        test_install
        ;;
    clean)
        clean_env
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        if [ -z "$1" ]; then
            show_help
        else
            print_error "Unknown command: $1"
            echo ""
            show_help
        fi
        exit 1
        ;;
esac
