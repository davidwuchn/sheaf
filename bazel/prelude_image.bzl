"""Build rule for deterministic Sheaf prelude images."""

def _prelude_image_impl(ctx):
    output = ctx.actions.declare_directory(ctx.label.name + ".out")
    arguments = ctx.actions.args()
    arguments.add(output.path)

    for module in ctx.files.modules:
        arguments.add("--module")
        arguments.add(module.basename)
        arguments.add(module.path)

    for source in sorted(ctx.files.frontend_srcs, key = lambda file: file.short_path):
        arguments.add("--frontend-source")
        arguments.add(source.short_path)
        arguments.add(source.path)

    ctx.actions.run(
        executable = ctx.executable.compiler,
        inputs = depset(ctx.files.modules + ctx.files.frontend_srcs),
        outputs = [output],
        arguments = [arguments],
        mnemonic = "SheafPrelude",
        progress_message = "Compiling Sheaf prelude image",
    )
    return [DefaultInfo(files = depset([output]))]

prelude_image = rule(
    implementation = _prelude_image_impl,
    attrs = {
        "compiler": attr.label(
            cfg = "exec",
            executable = True,
            mandatory = True,
        ),
        "frontend_srcs": attr.label_list(
            allow_files = [".rs"],
            mandatory = True,
        ),
        "modules": attr.label_list(
            allow_files = [".shf"],
            mandatory = True,
        ),
    },
)
