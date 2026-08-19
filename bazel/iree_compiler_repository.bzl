"""Repository rule exposing the local, version-matched IREE compiler."""

def _iree_compiler_repository_impl(repository_ctx):
    configured = repository_ctx.os.environ.get("IREE_COMPILE")
    if configured:
        compiler = repository_ctx.path(configured)
    else:
        home = repository_ctx.os.environ.get("HOME")
        if not home:
            fail("HOME is not set and IREE_COMPILE was not provided")
        compiler = repository_ctx.path(home).get_child(".sheaf").get_child("toolchain").get_child("iree-compile")

    if not compiler.exists:
        fail("iree-compile not found at {}. Set IREE_COMPILE explicitly.".format(compiler))

    repository_ctx.symlink(compiler, "iree-compile")
    repository_ctx.file(
        "BUILD.bazel",
        """
package(default_visibility = ["//visibility:public"])

exports_files(["iree-compile"])
""",
    )

iree_compiler_repository = repository_rule(
    implementation = _iree_compiler_repository_impl,
    environ = [
        "HOME",
        "IREE_COMPILE",
    ],
    local = True,
)
