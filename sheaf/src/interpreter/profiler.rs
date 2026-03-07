// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Aggregated profiler for `--blame` mode.
//!
//! Tracks per-function cumulative time (total and self), call count,
//! and prints a sorted summary at the end of execution.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

struct CallFrame {
    name: String,
    start: Instant,
    children_ns: u64,
}

pub struct ProfileEntry {
    pub calls: u64,
    pub total_ns: u64,
    pub self_ns: u64,
}

struct TreeEdge {
    calls: u64,
    total_ns: u64,
}

pub struct Profiler {
    stats: HashMap<String, ProfileEntry>,
    call_tree: HashMap<String, HashMap<String, TreeEdge>>,
    stack: Vec<CallFrame>,
    wall_start: Instant,
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
            call_tree: HashMap::new(),
            stack: Vec::new(),
            wall_start: Instant::now(),
        }
    }

    pub fn enter(&mut self, name: &str) {
        self.stack.push(CallFrame {
            name: name.to_string(),
            start: Instant::now(),
            children_ns: 0,
        });
    }

    pub fn exit(&mut self) {
        let frame = match self.stack.pop() {
            Some(f) => f,
            None => return,
        };
        let elapsed_ns = frame.start.elapsed().as_nanos() as u64;
        let self_ns = elapsed_ns.saturating_sub(frame.children_ns);

        if let Some(parent) = self.stack.last_mut() {
            parent.children_ns += elapsed_ns;
        }

        let parent_name = self.stack.last()
            .map(|f| f.name.clone())
            .unwrap_or_else(|| "<top>".to_string());
        let edge = self.call_tree
            .entry(parent_name)
            .or_default()
            .entry(frame.name.clone())
            .or_insert(TreeEdge { calls: 0, total_ns: 0 });
        edge.calls += 1;
        edge.total_ns += elapsed_ns;

        let entry = self.stats.entry(frame.name).or_insert(ProfileEntry {
            calls: 0,
            total_ns: 0,
            self_ns: 0,
        });
        entry.calls += 1;
        entry.total_ns += elapsed_ns;
        entry.self_ns += self_ns;
    }

    pub fn report(&self) {
        let wall_ns = self.wall_start.elapsed().as_nanos() as u64;
        eprintln!("\nProfiler: {} wall\n", format_duration(wall_ns));

        let mut entries: Vec<(&String, &ProfileEntry)> = self.stats.iter().collect();
        entries.sort_by(|a, b| b.1.self_ns.cmp(&a.1.self_ns));

        eprintln!(
            "  {:<30} {:>8} {:>10} {:>10} {:>10}",
            "Function", "Calls", "Total", "Self", "Avg/call"
        );
        eprintln!("  {}", "-".repeat(72));

        let threshold = wall_ns / 1000; // 0.1% of wall time
        let mut others_calls: u64 = 0;
        let mut others_self_ns: u64 = 0;
        let mut others_count: usize = 0;

        for (name, entry) in &entries {
            if entry.calls == 0 {
                continue;
            }
            if entry.self_ns < threshold {
                others_calls += entry.calls;
                others_self_ns += entry.self_ns;
                others_count += 1;
                continue;
            }
            let avg_ns = entry.total_ns / entry.calls;
            eprintln!(
                "  {:<30} {:>8} {:>10} {:>10} {:>10}",
                truncate(name, 30),
                entry.calls,
                format_duration(entry.total_ns),
                format_duration(entry.self_ns),
                format_duration(avg_ns),
            );
        }
        if others_count > 0 {
            eprintln!(
                "  {:<30} {:>8} {:>10} {:>10}",
                format!("... {} others", others_count),
                others_calls,
                "",
                format_duration(others_self_ns),
            );
        }
        eprintln!();
        self.report_tree();
    }

    fn report_tree(&self) {
        eprintln!("  Call tree:\n");
        let mut expanded = HashSet::new();
        if let Some(top_children) = self.call_tree.get("<top>") {
            let wall_ns = self.wall_start.elapsed().as_nanos() as u64;
            let threshold = wall_ns / 100;

            let mut sorted: Vec<_> = top_children.iter().collect();
            sorted.sort_by(|a, b| b.1.total_ns.cmp(&a.1.total_ns));

            let mut significant: Vec<(&String, &TreeEdge)> = Vec::new();
            let mut others_calls: u64 = 0;
            let mut others_ns: u64 = 0;
            let mut others_count: usize = 0;

            for (name, edge) in &sorted {
                if edge.total_ns >= threshold {
                    significant.push((name, edge));
                } else {
                    others_calls += edge.calls;
                    others_ns += edge.total_ns;
                    others_count += 1;
                }
            }

            let has_others = others_count > 0;
            let len = significant.len();
            for (i, (name, edge)) in significant.iter().enumerate() {
                let is_last = i == len - 1 && !has_others;
                let connector = if is_last { "└── " } else { "├── " };
                let continuation = if is_last { "    " } else { "│   " };
                let calls_str = if edge.calls == 1 { "call" } else { "calls" };
                eprintln!(
                    "  {}{} ({}, {} {})",
                    connector, name, format_duration(edge.total_ns), edge.calls, calls_str
                );
                expanded.insert(name.to_string());
                self.print_subtree(
                    name,
                    &format!("  {}", continuation),
                    &mut vec![name.to_string()],
                    &mut expanded,
                    edge.total_ns,
                );
            }

            if has_others {
                eprintln!(
                    "  └── ... {} {} ({}, {} calls)",
                    others_count,
                    if others_count == 1 { "other" } else { "others" },
                    format_duration(others_ns), others_calls
                );
            }
        }
        eprintln!();
    }

    fn print_subtree(
        &self,
        name: &str,
        indent: &str,
        ancestors: &mut Vec<String>,
        expanded: &mut HashSet<String>,
        parent_total_ns: u64,
    ) {
        if let Some(children) = self.call_tree.get(name) {
            let threshold = parent_total_ns / 100;

            let mut sorted: Vec<_> = children.iter().collect();
            sorted.sort_by(|a, b| b.1.total_ns.cmp(&a.1.total_ns));

            let mut significant: Vec<(&String, &TreeEdge)> = Vec::new();
            let mut others_calls: u64 = 0;
            let mut others_ns: u64 = 0;
            let mut others_count: usize = 0;

            for (child_name, edge) in &sorted {
                if edge.total_ns >= threshold {
                    significant.push((child_name, edge));
                } else {
                    others_calls += edge.calls;
                    others_ns += edge.total_ns;
                    others_count += 1;
                }
            }

            let has_others = others_count > 0;
            let len = significant.len();
            for (i, (child_name, edge)) in significant.iter().enumerate() {
                let is_last = i == len - 1 && !has_others;
                let connector = if is_last { "└── " } else { "├── " };
                let continuation = if is_last { "    " } else { "│   " };
                let calls_str = if edge.calls == 1 { "call" } else { "calls" };

                if ancestors.iter().any(|a| a == *child_name) {
                    eprintln!(
                        "{}{}{} ({}, {} {}) [recursive]",
                        indent, connector, child_name,
                        format_duration(edge.total_ns), edge.calls, calls_str
                    );
                } else if expanded.contains(child_name.as_str()) {
                    eprintln!(
                        "{}{}{} ({}, {} {}) [see above]",
                        indent, connector, child_name,
                        format_duration(edge.total_ns), edge.calls, calls_str
                    );
                } else {
                    eprintln!(
                        "{}{}{} ({}, {} {})",
                        indent, connector, child_name,
                        format_duration(edge.total_ns), edge.calls, calls_str
                    );
                    expanded.insert(child_name.to_string());
                    ancestors.push(child_name.to_string());
                    self.print_subtree(
                        child_name,
                        &format!("{}{}", indent, continuation),
                        ancestors,
                        expanded,
                        edge.total_ns,
                    );
                    ancestors.pop();
                }
            }

            if has_others {
                eprintln!(
                    "{}└── ... {} {} ({}, {} calls)",
                    indent, others_count,
                    if others_count == 1 { "other" } else { "others" },
                    format_duration(others_ns), others_calls
                );
            }
        }
    }
}

fn format_duration(ns: u64) -> String {
    if ns < 1_000 {
        format!("{}ns", ns)
    } else if ns < 1_000_000 {
        format!("{:.1}μs", ns as f64 / 1_000.0)
    } else if ns < 1_000_000_000 {
        format!("{:.2}ms", ns as f64 / 1_000_000.0)
    } else {
        format!("{:.2}s", ns as f64 / 1_000_000_000.0)
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max - 3])
    }
}
