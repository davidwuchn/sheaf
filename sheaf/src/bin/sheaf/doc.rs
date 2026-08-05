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
            if let Some(cat) = current.take()
                && !cat.functions.is_empty()
            {
                categories.push(cat);
            }
            current = Some(DocCategory {
                name: trimmed[3..].to_string(),
                functions: Vec::new(),
            });
        }
        // Function header: ### name[, name...]
        else if let Some(header) = trimmed.strip_prefix("### ")
            && let Some(ref mut cat) = current
        {
            cat.functions
                .extend(header.split(',').map(|name| name.trim().to_string()));
        }
    }

    // Push last category
    if let Some(cat) = current.take()
        && !cat.functions.is_empty()
    {
        categories.push(cat);
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
    let Some((start, header_len)) = find_documentation_header(name) else {
        return false;
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

fn find_documentation_header(name: &str) -> Option<(usize, usize)> {
    let mut offset = 0;

    for line in REFERENCE.split_inclusive('\n') {
        let trimmed = line.trim();
        if let Some(header) = trimmed
            .strip_prefix("#### ")
            .or_else(|| trimmed.strip_prefix("### "))
            && header.split(',').any(|entry| entry.trim() == name)
        {
            let header_len = trimmed.len();
            let leading_whitespace = line.len() - line.trim_start().len();
            return Some((offset + leading_whitespace, header_len));
        }
        offset += line.len();
    }

    None
}

#[cfg(test)]
mod tests {
    use super::{find_documentation_header, parse_reference};

    #[test]
    fn grouped_headers_are_individual_documentation_entries() {
        let categories = parse_reference();
        let tensor_creation = categories
            .iter()
            .find(|category| category.name == "Tensor Creation")
            .expect("Tensor Creation category should exist");

        assert!(tensor_creation.functions.contains(&"ones".to_string()));
        assert!(tensor_creation.functions.contains(&"zeros".to_string()));
        assert!(find_documentation_header("ones").is_some());
        assert!(find_documentation_header("zeros").is_some());
    }

    #[test]
    fn internal_autodiff_primitives_have_no_public_documentation() {
        assert!(find_documentation_header("@-grad-lhs").is_none());
        assert!(find_documentation_header("@-grad-rhs").is_none());
    }
}
