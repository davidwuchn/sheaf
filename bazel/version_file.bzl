"""Rule generating the Cargo-compatible Sheaf version source."""

def _version_file_impl(ctx):
    output = ctx.actions.declare_directory(ctx.label.name + "_out")
    ctx.actions.run_shell(
        inputs = [ctx.file.cargo_toml],
        outputs = [output],
        arguments = [
            ctx.file.cargo_toml.path,
            output.path,
            ctx.var.get("SHEAF_BUILD_VERSION", ""),
        ],
        command = """
set -eu
version="$3"
if [ -z "$version" ]; then
    version=$(awk '
        /^version = / {
            gsub(/\"/, "", $3)
            print $3
            exit
        }
    ' "$1")
fi
test -n "$version"
mkdir -p "$2"
printf 'pub const SHEAF_VERSION: &str = "%s";\\n' "$version" > "$2/generated_version.rs"
""",
        mnemonic = "SheafVersion",
        progress_message = "Generating Sheaf version source",
    )
    return [DefaultInfo(files = depset([output]))]

version_file = rule(
    implementation = _version_file_impl,
    attrs = {
        "cargo_toml": attr.label(
            allow_single_file = True,
            mandatory = True,
        ),
    },
)

def _cargo_rustc_env_impl(ctx):
    output = ctx.actions.declare_file(ctx.label.name + ".env")
    ctx.actions.run_shell(
        inputs = [ctx.file.cargo_toml],
        outputs = [output],
        arguments = [ctx.file.cargo_toml.path, output.path],
        command = """
set -eu
package_version=$(awk '
    /^version = / {
        gsub(/\"/, "", $3)
        print $3
        exit
    }
' "$1")
iree_version=$(awk '
    /^iree-version = / {
        gsub(/\"/, "", $3)
        print $3
        exit
    }
' "$1")
test -n "$package_version"
test -n "$iree_version"
printf 'IREE_VERSION=%s\\nSHEAF_COMPILER_VERSION=%s\\n' \
    "$iree_version" "$package_version" > "$2"
""",
        mnemonic = "SheafCargoEnv",
        progress_message = "Generating Sheaf rustc environment",
    )
    return [DefaultInfo(files = depset([output]))]

cargo_rustc_env = rule(
    implementation = _cargo_rustc_env_impl,
    attrs = {
        "cargo_toml": attr.label(
            allow_single_file = True,
            mandatory = True,
        ),
    },
)
