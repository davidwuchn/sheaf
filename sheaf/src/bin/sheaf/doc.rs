// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Built-in documentation system.
//!
//! Parses the embedded reference.md and provides:
//! - `doc_list`: categorized index of all functions
//! - `doc_show`: detailed signature + description + examples for one function

const REFERENCE: &str = include_str!("../../../assets/reference.md");

/// A parsed category with its functions.
pub struct DocCategory {
    pub name: String,
    pub functions: Vec<String>,
}

/// Parse reference.md into categories and their functions.
pub fn parse_reference() -> Vec<DocCategory> {
    let mut categories = Vec::new();
    let mut current: Option<DocCategory> = None;

    for line in REFERENCE.lines() {
        let trimmed = line.trim();

        // Category header: ## Name
        if trimmed.starts_with("## ") && !trimmed.starts_with("###") {
            if let Some(cat) = current.take() {
                if !cat.functions.is_empty() {
                    categories.push(cat);
                }
            }
            current = Some(DocCategory {
                name: trimmed[3..].to_string(),
                functions: Vec::new(),
            });
        }
        // Function header: ### name
        else if trimmed.starts_with("### ") {
            if let Some(ref mut cat) = current {
                cat.functions.push(trimmed[4..].to_string());
            }
        }
    }

    // Push last category
    if let Some(cat) = current.take() {
        if !cat.functions.is_empty() {
            categories.push(cat);
        }
    }

    categories
}

/// Print categorized index of all built-in functions.
pub fn doc_list() {
    let categories = parse_reference();
    let total: usize = categories.iter().map(|c| c.functions.len()).sum();

    println!("Sheaf built-in functions ({})\n", total);

    for cat in &categories {
        println!("{}:", cat.name);
        for fn_name in &cat.functions {
            println!("- {}", fn_name);
        }
        println!();
    }
}

/// Print detailed documentation for a single function.
/// Returns true on success, false if the function was not found.
pub fn doc_show(name: &str) -> bool {
    let header_h4 = format!("#### {}", name);
    let header_h3 = format!("### {}", name);
    let (start, header_len) = match REFERENCE.find(&header_h4) {
        Some(pos) => (pos, header_h4.len()),
        None => match REFERENCE.find(&header_h3) {
            Some(pos) => {
                if pos > 0 && REFERENCE.as_bytes()[pos - 1] == b'#' {
                    return false;
                }
                (pos, header_h3.len())
            }
            None => return false,
        },
    };

    let content = &REFERENCE[start + header_len..];
    let end = [
        content.find("\n## "),
        content.find("\n### "),
        content.find("\n#### "),
    ]
    .into_iter()
    .flatten()
    .min()
    .unwrap_or(content.len());
    let section = content[..end].trim();

    println!("\n  {}", name);
    println!("  {}", "-".repeat(name.len()));

    let mut example_count = 0;
    let mut in_code = false;

    for line in section.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            if in_code {
                in_code = false;
                println!();
            } else {
                in_code = true;
                example_count += 1;
                if example_count > 3 {
                    continue;
                }
            }
            continue;
        }

        if example_count > 3 {
            continue;
        }

        if in_code {
            println!("    {}", line);
        } else if trimmed.starts_with("**Type:**") {
            println!("  {}", trimmed.replace("**", ""));
        } else if trimmed.starts_with("**Signature:**") {
            println!("  {}", trimmed.replace("**", "").replace('`', ""));
        } else if trimmed == "---" {
            // Skip separators
        } else if !trimmed.is_empty() {
            let clean = trimmed.replace("**", "").replace('`', "");
            println!("  {}", clean);
        }
    }
    println!();
    true
}
