"""Repository rule fetching the version-matched IREE compiler."""

def _iree_version(cargo_toml):
    for line in cargo_toml.splitlines():
        if line.startswith("iree-version"):
            parts = line.split("=", 1)
            if len(parts) == 2:
                return parts[1].strip().strip("\"")
    fail("iree-version not found in sheaf/Cargo.toml")

def _wheel_platform(repository_ctx):
    os_name = repository_ctx.os.name
    arch = repository_ctx.os.arch
    if os_name == "mac os x" and arch in ["aarch64", "arm64"]:
        return "macosx_13_0_universal2", "libIREECompiler.dylib"
    if os_name == "linux" and arch in ["aarch64", "arm64"]:
        return "manylinux_2_28_aarch64", "libIREECompiler.so"
    if os_name == "linux" and arch in ["amd64", "x86_64"]:
        return "manylinux_2_28_x86_64", "libIREECompiler.so"
    fail("unsupported IREE compiler platform: {} {}".format(os_name, arch))

def _resolve_wheel(repository_ctx, version):
    platform_tag, library = _wheel_platform(repository_ctx)
    metadata_path = "iree-compiler-metadata.json"
    repository_ctx.download(
        url = "https://pypi.org/pypi/iree-base-compiler/{}/json".format(version),
        output = metadata_path,
    )
    metadata = json.decode(repository_ctx.read(metadata_path))
    repository_ctx.delete(metadata_path)

    for artifact in metadata.get("urls", []):
        filename = artifact.get("filename", "")
        if filename.endswith(".whl") and platform_tag in filename:
            sha256 = artifact.get("digests", {}).get("sha256")
            url = artifact.get("url")
            if not url or not sha256:
                fail("PyPI metadata is missing the URL or SHA-256 for {}".format(filename))
            return url, sha256, library

    fail("no IREE compiler wheel found for version {} and platform {}".format(version, platform_tag))

def _iree_compiler_repository_impl(repository_ctx):
    configured = repository_ctx.os.environ.get("IREE_COMPILE")
    if configured:
        compiler = repository_ctx.path(configured)
        if not compiler.exists:
            fail("IREE_COMPILE does not exist: {}".format(compiler))
        repository_ctx.symlink(compiler, "bin/iree-compile")
        support_files = []
        for name in ["iree-lld", "libIREECompiler.dylib", "libIREECompiler.so"]:
            source = compiler.dirname.get_child(name)
            if source.exists:
                repository_ctx.symlink(source, "bin/{}".format(name))
                support_files.append(name)
    else:
        version = _iree_version(repository_ctx.read(repository_ctx.attr.cargo_toml))
        url, sha256, library = _resolve_wheel(repository_ctx, version)
        repository_ctx.download_and_extract(
            url = url,
            sha256 = sha256,
            type = "zip",
        )
        source_dir = repository_ctx.path("iree/compiler/_mlir_libs")
        repository_ctx.symlink(source_dir.get_child("iree-compile"), "bin/iree-compile")
        repository_ctx.symlink(source_dir.get_child("iree-lld"), "bin/iree-lld")
        repository_ctx.symlink(source_dir.get_child(library), "bin/{}".format(library))
        support_files = ["iree-lld", library]

    repository_ctx.file(
        "tool.bzl",
        """def _iree_compiler_tool_impl(ctx):
    executable = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.symlink(
        output = executable,
        target_file = ctx.file.binary,
        is_executable = True,
    )
    return [DefaultInfo(
        executable = executable,
        files = depset([executable]),
        runfiles = ctx.runfiles(files = [ctx.file.binary] + ctx.files.support),
    )]

iree_compiler_tool = rule(
    implementation = _iree_compiler_tool_impl,
    executable = True,
    attrs = {
        "binary": attr.label(allow_single_file = True, mandatory = True),
        "support": attr.label_list(allow_files = True),
    },
)
""",
    )
    repository_ctx.file(
        "BUILD.bazel",
        """load(":tool.bzl", "iree_compiler_tool")

package(default_visibility = ["//visibility:public"])

exports_files(glob(["bin/*"]))

iree_compiler_tool(
    name = "iree-compile",
    binary = "bin/iree-compile",
    support = [{}],
)
""".format(
            ", ".join(["\"bin/{}\"".format(name) for name in support_files]),
        ),
    )

iree_compiler_repository = repository_rule(
    implementation = _iree_compiler_repository_impl,
    attrs = {
        "cargo_toml": attr.label(
            allow_single_file = True,
            mandatory = True,
        ),
    },
    environ = ["IREE_COMPILE"],
)
