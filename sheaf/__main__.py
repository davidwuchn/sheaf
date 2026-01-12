#!/usr/bin/env python3
# Copyright (c) 2025 Damien Boureille
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
Sheaf CLI - Command-line interface for the Sheaf language.
"""

import os
import sys


def main():
    args = sys.argv[1:]

    # No arguments -> launch REPL
    if len(args) == 0:
        from sheaf.repl.__main__ import main as repl_main

        repl_main()
        return

    # Help
    if args[0] in ("-h", "--help", "help"):
        print("""
Sheaf - A Functional Language for Differentiable Computation

Usage:
    sheaf                 Launch interactive console (REPL)
    sheaf <file.shf>      Execute a Sheaf file
    sheaf --help          Show this help message
""")
        return

    # Execute file
    filename = args[0]

    if not os.path.exists(filename):
        print(f"Error: File not found: {filename}", file=sys.stderr)
        sys.exit(1)

    # Load and execute Sheaf file
    if not filename.endswith(".shf"):
        print(f"Warning: {filename} doesn't have .shf extension", file=sys.stderr)

    from sheaf import Sheaf

    compiler = Sheaf()

    try:
        compiler.load_file(filename)
        # If there's a main function, call it
        if "main" in compiler.registry:
            result = compiler.registry["main"]()
            if result is not None:
                print(result)
    except Exception as e:
        # Errors are already formatted by the error handler
        sys.exit(1)


if __name__ == "__main__":
    main()
