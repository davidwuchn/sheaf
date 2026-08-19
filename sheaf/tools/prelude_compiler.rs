// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

use sheaf_frontend::CompilerContext;
use sheaf_frontend::core::prelude::{PreludeIdentity, content_identity, encode};
use std::path::PathBuf;

struct Input {
    name: String,
    path: PathBuf,
    bytes: Vec<u8>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let output_directory = PathBuf::from(arguments.next().ok_or("missing output directory")?);
    let mut modules = Vec::new();
    let mut frontend_sources = Vec::new();

    while let Some(kind) = arguments.next() {
        let name = arguments.next().ok_or("missing input name")?;
        let path = PathBuf::from(arguments.next().ok_or("missing input path")?);
        let bytes = std::fs::read(&path)?;
        let input = Input { name, path, bytes };
        match kind.as_str() {
            "--module" => modules.push(input),
            "--frontend-source" => frontend_sources.push(input),
            _ => return Err(format!("unknown argument '{kind}'").into()),
        }
    }

    if modules.is_empty() {
        return Err("no standard-library modules were provided".into());
    }
    if frontend_sources.is_empty() {
        return Err("no frontend sources were provided".into());
    }

    let mut context = CompilerContext::new_without_prelude();
    for module in &modules {
        let source = std::str::from_utf8(&module.bytes)
            .map_err(|error| format!("{} is not UTF-8: {error}", module.path.display()))?;
        context.compile_prelude_module(&module.name, source)?;
    }

    frontend_sources.sort_by(|left, right| left.name.cmp(&right.name));
    let identity = PreludeIdentity {
        compiler_version: env!("SHEAF_COMPILER_VERSION").to_string(),
        frontend_identity: content_identity(
            frontend_sources.iter().map(|input| (input.name.as_str(), input.bytes.as_slice())),
        ),
        source_hash: content_identity(
            modules.iter().map(|input| (input.name.as_str(), input.bytes.as_slice())),
        ),
    };
    let image = encode(&context, &identity)?;
    std::fs::create_dir_all(&output_directory)?;
    std::fs::write(output_directory.join("prelude.bin"), image)?;
    let metadata = format!(
        "const PRELUDE_FRONTEND_IDENTITY: &str = {:?};\nconst PRELUDE_SOURCE_HASH: &str = {:?};\n",
        identity.frontend_identity,
        identity.source_hash,
    );
    std::fs::write(output_directory.join("metadata.rs"), metadata)?;
    Ok(())
}
