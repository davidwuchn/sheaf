"""Repository rule exposing a prebuilt IREE runtime to Bazel."""

_REQUIRED_LIBRARIES = [
    "libiree_runtime_unified.a",
    "libflatcc_parsing.a",
    "libflatcc_runtime.a",
]

_BUILD_FILE = """
load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_import(
    name = "runtime_unified",
    static_library = "libiree_runtime_unified.a",
    alwayslink = True,
)

cc_import(
    name = "flatcc_parsing",
    static_library = "libflatcc_parsing.a",
)

cc_import(
    name = "flatcc_runtime",
    static_library = "libflatcc_runtime.a",
)

cc_library(
    name = "iree_runtime",
    deps = [
        ":flatcc_parsing",
        ":flatcc_runtime",
        ":runtime_unified",
    ],
    linkopts = [
        "-framework Accelerate",
        "-framework CoreFoundation",
        "-framework Foundation",
        "-framework Metal",
        "-framework IOKit",
        "-lc++",
    ],
)
"""

def _resolve_library_directory(repository_ctx):
    explicit = repository_ctx.os.environ.get("IREE_RUNTIME_LIB_DIR")
    if explicit:
        return repository_ctx.path(explicit)

    cargo_toml = repository_ctx.path(Label("//sheaf:Cargo.toml"))
    return cargo_toml.dirname.get_child("iree-runtime")

def _iree_runtime_repository_impl(repository_ctx):
    library_directory = _resolve_library_directory(repository_ctx)
    missing = []

    for library in _REQUIRED_LIBRARIES:
        source = library_directory.get_child(library)
        if source.exists:
            repository_ctx.symlink(source, library)
        else:
            missing.append(str(source))

    if missing:
        fail(
            "IREE runtime libraries not found:\n  " +
            "\n  ".join(missing) +
            "\nSet IREE_RUNTIME_LIB_DIR or populate sheaf/iree-runtime.",
        )

    repository_ctx.file("BUILD.bazel", _BUILD_FILE)

iree_runtime_repository = repository_rule(
    implementation = _iree_runtime_repository_impl,
    environ = ["IREE_RUNTIME_LIB_DIR"],
    local = True,
)
