"""Repository rule configuring IREE CUDA headers from an installed toolkit."""

def _cuda_root(repository_ctx):
    configured = repository_ctx.os.environ.get("IREE_CUDA_TOOLKIT_ROOT")
    if configured:
        return repository_ctx.path(configured)

    nvcc = repository_ctx.which("nvcc")
    if nvcc:
        return nvcc.dirname.dirname

    return None

def _iree_cuda_repository_impl(repository_ctx):
    cuda_root = _cuda_root(repository_ctx)
    enabled = cuda_root != None
    libdevice_rel_path = "iree_local/libdevice.bc"

    if enabled:
        include_dir = cuda_root.get_child("include")
        libdevice = cuda_root.get_child("nvvm").get_child("libdevice").get_child("libdevice.10.bc")
        if not include_dir.exists:
            fail("CUDA include directory not found: {}".format(include_dir))
        if not libdevice.exists:
            fail("CUDA libdevice not found: {}".format(libdevice))
        repository_ctx.symlink(include_dir, "include")
        repository_ctx.symlink(libdevice, libdevice_rel_path)

    repository_ctx.template(
        "BUILD.bazel",
        repository_ctx.attr.build_template,
        substitutions = {
            "%ENABLED%": "True" if enabled else "False",
            "%IREE_REPO_ALIAS%": "@iree",
            "%LIBDEVICE_REL_PATH%": libdevice_rel_path if enabled else "BUILD.bazel",
        },
    )

iree_cuda_repository = repository_rule(
    implementation = _iree_cuda_repository_impl,
    attrs = {
        "build_template": attr.label(
            allow_single_file = True,
            default = "@iree//:build_tools/third_party/cuda/BUILD.template",
        ),
    },
    environ = [
        "IREE_CUDA_TOOLKIT_ROOT",
        "PATH",
    ],
    local = True,
)
