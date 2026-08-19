"""Rust test helpers for the Sheaf Cargo package."""

load("@crates//:defs.bzl", "aliases", "all_crate_deps", "crate_edition")
load("@rules_rust//rust:defs.bzl", "rust_test")

_CARGO_PACKAGE = "sheaf"


def sheaf_integration_test(name, data = []):
    rust_test(
        name = name,
        srcs = [name + ".rs"],
        aliases = aliases(package_name = _CARGO_PACKAGE),
        data = data,
        edition = crate_edition(package_name = _CARGO_PACKAGE),
        env = {"RUST_TEST_THREADS": "1"},
        rustc_env = {
            "CARGO_MANIFEST_DIR": "sheaf",
        },
        deps = ["//sheaf:compiler_lib"] + all_crate_deps(
            normal = True,
            package_name = _CARGO_PACKAGE,
        ),
    )
