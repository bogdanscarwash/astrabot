#!/usr/bin/env python3
"""
Test runner script for Astrabot
Provides a simple interface for running different types of tests
"""

import sys
import subprocess
import argparse
import os


def run_command(cmd):
    """Run a command and return the result"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Astrabot Test Runner')
    parser.add_argument(
        'test_type',
        choices=['all', 'unit', 'integration', 'coverage', 'quick', 'schemas', 'utils'],
        help='Type of tests to run'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--failfast', '-x',
        action='store_true',
        help='Stop on first failure'
    )
    
    args = parser.parse_args()
    
    # Build pytest command
    cmd_parts = ['pytest']
    
    if args.verbose:
        cmd_parts.append('-v')
    
    if args.failfast:
        cmd_parts.append('-x')
    
    # Add specific test selection
    if args.test_type == 'unit':
        cmd_parts.extend(['-m', 'unit'])
    elif args.test_type == 'integration':
        cmd_parts.extend(['-m', 'integration'])
    elif args.test_type == 'coverage':
        cmd_parts.extend(['--cov=.', '--cov-report=html', '--cov-report=term'])
    elif args.test_type == 'quick':
        cmd_parts.append('-q')
    elif args.test_type == 'schemas':
        cmd_parts.append('tests/test_structured_schemas.py')
    elif args.test_type == 'utils':
        cmd_parts.append('tests/test_conversation_utilities.py')
    
    # Run the tests
    cmd = ' '.join(cmd_parts)
    exit_code = run_command(cmd)
    
    # Show coverage report location if coverage was run
    if args.test_type == 'coverage' and exit_code == 0:
        print("\n✅ Coverage report generated!")
        print("📊 View HTML report: open htmlcov/index.html")
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()