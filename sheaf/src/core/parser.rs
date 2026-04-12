// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Sheaf parser - converts source code to AST

use crate::core::ast::{SheafValue, SourceLocation};
use crate::core::error::{SheafError, SheafResult};
use std::rc::Rc;

type ParseResult<T> = SheafResult<T>;

/// Token types
#[derive(Debug, Clone, PartialEq)]
enum Token {
    // Delimiters
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,

    // Special characters
    Quote,         // '
    Quasiquote,    // `
    Unquote,       // ~
    UnquoteSplice, // ~@

    // Literals and identifiers
    Symbol(String),
    Keyword(String),
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Nil,
}

/// Token with source location
#[derive(Debug, Clone)]
struct LocatedToken {
    token: Token,
    location: SourceLocation,
}

/// Tokenizer - converts source code into tokens
struct Tokenizer {
    source: Vec<char>,
    pos: usize,
    line: usize,
    column: usize,
    filename: Rc<str>,
}

impl Tokenizer {
    fn new(source: &str, filename: impl Into<Rc<str>>) -> Self {
        Self {
            source: source.chars().collect(),
            pos: 0,
            line: 1,
            column: 1,
            filename: filename.into(),
        }
    }

    fn location(&self) -> SourceLocation {
        SourceLocation::new(self.line, self.column, Rc::clone(&self.filename))
    }

    fn peek(&self) -> Option<char> {
        self.source.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.pos += 1;
        if ch == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        Some(ch)
    }

    fn skip_whitespace_and_comments(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() || ch == ',' {
                self.advance();
            } else if ch == ';' {
                // Skip until end of line
                while let Some(ch) = self.peek() {
                    self.advance();
                    if ch == '\n' {
                        break;
                    }
                }
            } else {
                break;
            }
        }
    }

    fn read_string(&mut self) -> ParseResult<String> {
        let start_loc = self.location();
        self.advance(); // Skip opening "

        let mut result = String::new();
        loop {
            match self.peek() {
                None => {
                    return Err(SheafError::Parse {
                        message: "Unterminated string".to_string(),
                        location: start_loc,
                    });
                }
                Some('"') => {
                    self.advance();
                    break;
                }
                Some('\\') => {
                    self.advance();
                    match self.peek() {
                        Some('n') => {
                            result.push('\n');
                            self.advance();
                        }
                        Some('t') => {
                            result.push('\t');
                            self.advance();
                        }
                        Some('r') => {
                            result.push('\r');
                            self.advance();
                        }
                        Some('"') => {
                            result.push('"');
                            self.advance();
                        }
                        Some('\\') => {
                            result.push('\\');
                            self.advance();
                        }
                        Some(ch) => {
                            result.push(ch);
                            self.advance();
                        }
                        None => {
                            return Err(SheafError::Parse {
                                message: "Unterminated string escape".to_string(),
                                location: self.location(),
                            });
                        }
                    }
                }
                Some(ch) => {
                    result.push(ch);
                    self.advance();
                }
            }
        }
        Ok(result)
    }

    fn read_symbol_or_number(&mut self) -> ParseResult<Token> {
        let mut s = String::new();

        // Collect characters that are valid in symbols/numbers
        while let Some(ch) = self.peek() {
            if ch.is_whitespace()
                || ch == '('
                || ch == ')'
                || ch == '['
                || ch == ']'
                || ch == '{'
                || ch == '}'
                || ch == '"'
                || ch == '\''
                || ch == '`'
                || ch == '~'
                || ch == ';'
                || ch == ','
            {
                break;
            }
            s.push(ch);
            self.advance();
        }

        // Try to parse as boolean
        if s == "true" {
            return Ok(Token::Boolean(true));
        }
        if s == "false" {
            return Ok(Token::Boolean(false));
        }
        if s == "nil" {
            return Ok(Token::Nil);
        }

        // Try to parse as keyword (:foo)
        if s.starts_with(':') {
            return Ok(Token::Keyword(s[1..].to_string()));
        }

        // Try to parse as number
        if let Ok(n) = s.parse::<i64>() {
            return Ok(Token::Integer(n));
        }
        if let Ok(x) = s.parse::<f64>() {
            return Ok(Token::Float(x));
        }

        // Otherwise it's a symbol
        Ok(Token::Symbol(s))
    }

    fn next_token(&mut self) -> ParseResult<Option<LocatedToken>> {
        self.skip_whitespace_and_comments();

        let loc = self.location();

        match self.peek() {
            None => Ok(None),
            Some('(') => {
                self.advance();
                Ok(Some(LocatedToken {
                    token: Token::LParen,
                    location: loc,
                }))
            }
            Some(')') => {
                self.advance();
                Ok(Some(LocatedToken {
                    token: Token::RParen,
                    location: loc,
                }))
            }
            Some('[') => {
                self.advance();
                Ok(Some(LocatedToken {
                    token: Token::LBracket,
                    location: loc,
                }))
            }
            Some(']') => {
                self.advance();
                Ok(Some(LocatedToken {
                    token: Token::RBracket,
                    location: loc,
                }))
            }
            Some('{') => {
                self.advance();
                Ok(Some(LocatedToken {
                    token: Token::LBrace,
                    location: loc,
                }))
            }
            Some('}') => {
                self.advance();
                Ok(Some(LocatedToken {
                    token: Token::RBrace,
                    location: loc,
                }))
            }
            Some('\'') => {
                self.advance();
                Ok(Some(LocatedToken {
                    token: Token::Quote,
                    location: loc,
                }))
            }
            Some('`') => {
                self.advance();
                Ok(Some(LocatedToken {
                    token: Token::Quasiquote,
                    location: loc,
                }))
            }
            Some('~') => {
                self.advance();
                if self.peek() == Some('@') {
                    self.advance();
                    Ok(Some(LocatedToken {
                        token: Token::UnquoteSplice,
                        location: loc,
                    }))
                } else {
                    Ok(Some(LocatedToken {
                        token: Token::Unquote,
                        location: loc,
                    }))
                }
            }
            Some('"') => {
                let s = self.read_string()?;
                Ok(Some(LocatedToken {
                    token: Token::String(s),
                    location: loc,
                }))
            }
            Some(_) => {
                let token = self.read_symbol_or_number()?;
                Ok(Some(LocatedToken {
                    token,
                    location: loc,
                }))
            }
        }
    }

    fn tokenize(&mut self) -> ParseResult<Vec<LocatedToken>> {
        let mut tokens = Vec::new();
        while let Some(token) = self.next_token()? {
            tokens.push(token);
        }
        Ok(tokens)
    }
}

/// Parser - converts tokens into AST
struct Parser {
    tokens: Vec<LocatedToken>,
    pos: usize,
    open_delimiters: Vec<(char, SourceLocation)>,
}

impl Parser {
    fn new(tokens: Vec<LocatedToken>) -> Self {
        Self { tokens, pos: 0, open_delimiters: Vec::new() }
    }

    fn peek(&self) -> Option<&LocatedToken> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<LocatedToken> {
        let token = self.tokens.get(self.pos).cloned();
        if token.is_some() {
            self.pos += 1;
        }
        token
    }

    fn parse_expr(&mut self) -> ParseResult<SheafValue> {
        let token = self.advance().ok_or_else(|| SheafError::Parse {
            message: "Unexpected end of input".to_string(),
            location: SourceLocation::unknown(),
        })?;

        match token.token {
            Token::LParen => self.parse_list(token.location),
            Token::LBracket => {
                let vec_loc = token.location.clone();
                let vec_val = self.parse_vector(token.location)?;
                // Desugar [1 2 3] :bf16 -> (cast [1 2 3] :bf16)
                if let Some(next) = self.peek() {
                    if let Token::Keyword(ref k) = next.token {
                        if matches!(k.as_str(), "bf16" | "f16" | "f32" | "i32" | "bool") {
                            let kw_str = k.clone();
                            let kw_loc = next.location.clone();
                            self.advance();
                            return Ok(SheafValue::List(
                                vec![
                                    SheafValue::Symbol("cast".to_string(), vec_loc.clone()),
                                    vec_val,
                                    SheafValue::Keyword(kw_str, kw_loc),
                                ],
                                vec_loc,
                            ));
                        }
                    }
                }
                Ok(vec_val)
            }
            Token::LBrace => self.parse_dict(token.location),
            Token::Quote => {
                let expr = self.parse_expr()?;
                Ok(SheafValue::Quote(Box::new(expr), token.location))
            }
            Token::Quasiquote => {
                let expr = self.parse_expr()?;
                Ok(SheafValue::Quasiquote(Box::new(expr), token.location))
            }
            Token::Unquote => {
                let expr = self.parse_expr()?;
                Ok(SheafValue::Unquote(Box::new(expr), token.location))
            }
            Token::UnquoteSplice => {
                let expr = self.parse_expr()?;
                Ok(SheafValue::UnquoteSplicing(Box::new(expr), token.location))
            }
            Token::Symbol(s) => Ok(SheafValue::Symbol(s, token.location)),
            Token::Keyword(k) => Ok(SheafValue::Keyword(k, token.location)),
            Token::Integer(n) => Ok(SheafValue::Integer(n, token.location)),
            Token::Float(x) => Ok(SheafValue::Float(x, token.location)),
            Token::String(s) => Ok(SheafValue::String(s, token.location)),
            Token::Boolean(b) => Ok(SheafValue::Boolean(b, token.location)),
            Token::Nil => Ok(SheafValue::Nil(token.location)),
            Token::RParen | Token::RBracket | Token::RBrace => {
                let closing = match token.token {
                    Token::RParen => ")",
                    Token::RBracket => "]",
                    Token::RBrace => "}",
                    _ => unreachable!(),
                };
                let hint = if let Some((open_ch, open_loc)) = self.open_delimiters.last() {
                    let expected = match open_ch {
                        '(' => ")",
                        '[' => "]",
                        '{' => "}",
                        _ => "?",
                    };
                    if closing != expected {
                        format!(
                            "\n  = hint: Expected '{}' to close '{}' opened at line {}. \
                             A '{}' is probably missing before this point.",
                            expected, open_ch, open_loc.line, expected
                        )
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                };
                Err(SheafError::Parse {
                    message: format!("Unexpected closing '{closing}'{hint}"),
                    location: token.location,
                })
            }
        }
    }

    fn parse_list(&mut self, start_loc: SourceLocation) -> ParseResult<SheafValue> {
        self.open_delimiters.push(('(', start_loc.clone()));
        let mut elements = Vec::new();
        loop {
            match self.peek() {
                None => {
                    let hint = self.indentation_hint(&start_loc, &elements);
                    let head = elements.first().and_then(|e| e.as_symbol()).unwrap_or("...");
                    return Err(SheafError::Parse {
                        message: format!("Unclosed ({} ...), missing ')'{}", head, hint),
                        location: start_loc,
                    });
                }
                Some(token) if matches!(token.token, Token::RParen) => {
                    self.advance();
                    self.open_delimiters.pop();
                    break;
                }
                _ => {
                    elements.push(self.parse_expr()?);
                }
            }
        }
        Ok(SheafValue::List(elements, start_loc))
    }

    fn indentation_hint(&self, open_loc: &SourceLocation, elements: &[SheafValue]) -> String {
        let open_col = open_loc.column;
        for elem in elements {
            let loc = elem.location();
            if loc.line > open_loc.line && loc.column <= open_col {
                return format!(
                    "\n  = hint: Expression at line {} (column {}) is indented at the same level \
                     as the ( at line {}. A ) is probably missing before line {}.",
                    loc.line, loc.column, open_loc.line, loc.line
                );
            }
        }
        String::new()
    }

    fn parse_vector(&mut self, start_loc: SourceLocation) -> ParseResult<SheafValue> {
        self.open_delimiters.push(('[', start_loc.clone()));
        let mut elements = Vec::new();
        loop {
            match self.peek() {
                None => {
                    let hint = self.indentation_hint(&start_loc, &elements);
                    return Err(SheafError::Parse {
                        message: format!("Unclosed [...], missing ']'{}", hint),
                        location: start_loc,
                    });
                }
                Some(token) if matches!(token.token, Token::RBracket) => {
                    self.advance();
                    self.open_delimiters.pop();
                    break;
                }
                _ => {
                    elements.push(self.parse_expr()?);
                }
            }
        }
        Ok(SheafValue::Vector(elements, start_loc))
    }

    fn parse_dict(&mut self, start_loc: SourceLocation) -> ParseResult<SheafValue> {
        self.open_delimiters.push(('{', start_loc.clone()));
        let mut pairs = Vec::new();
        loop {
            match self.peek() {
                None => {
                    return Err(SheafError::Parse {
                        message: "Unclosed {...}, missing '}'".to_string(),
                        location: start_loc,
                    });
                }
                Some(token) if matches!(token.token, Token::RBrace) => {
                    self.advance();
                    self.open_delimiters.pop();
                    break;
                }
                _ => {
                    let key = self.parse_expr()?;
                    let value = self.parse_expr()?;
                    pairs.push((key, value));
                }
            }
        }
        Ok(SheafValue::Dict(pairs, start_loc))
    }

    fn parse_all(&mut self) -> ParseResult<Vec<SheafValue>> {
        let mut exprs = Vec::new();
        while self.peek().is_some() {
            exprs.push(self.parse_expr()?);
        }
        Ok(exprs)
    }
}

/// Parse Sheaf source code into AST
pub fn parse(source: &str, filename: impl Into<String>) -> ParseResult<Vec<SheafValue>> {
    let filename = filename.into();
    let mut tokenizer = Tokenizer::new(source, filename.clone());
    let tokens = tokenizer.tokenize()?;
    let mut parser = Parser::new(tokens);
    parser.parse_all().map_err(|e| {
        if let SheafError::Parse { ref message, ref location } = e {
            let balance = count_paren_balance(source);
            // Enrich "Unexpected closing" errors with paren balance info
            if message.contains("Unexpected closing") && !message.contains("hint:") {
                if balance.extra_close > 0 {
                    return SheafError::Parse {
                        message: format!(
                            "{}\n  = hint: Found {} extra closing delimiter{}. \
                             Check the line above for a ) that closes an expression too early.",
                            message,
                            balance.extra_close,
                            if balance.extra_close == 1 { "" } else { "s" },
                        ),
                        location: location.clone(),
                    };
                }
            }
            // Enrich "Unclosed" errors with count of missing delimiters
            if message.starts_with("Unclosed") && balance.unclosed > 0 {
                let hint = if balance.unclosed == 1 {
                    "Add 1 closing delimiter at the end.".to_string()
                } else {
                    format!("Add {} closing delimiters at the end.", balance.unclosed)
                };
                let new_message = if message.contains("\n  = hint:") {
                    // Already has a hint, append our count
                    format!("{}\n  = hint: {}", message, hint)
                } else {
                    format!("{}\n  = hint: {}", message, hint)
                };
                return SheafError::Parse {
                    message: new_message,
                    location: location.clone(),
                };
            }
        }
        e
    })
}

struct ParenBalance {
    extra_close: usize,
    unclosed: usize,
}

fn count_paren_balance(source: &str) -> ParenBalance {
    let mut depth: i64 = 0;
    let mut extra_close: usize = 0;
    let mut in_string = false;
    let mut in_comment = false;
    let mut prev_char = '\0';

    for ch in source.chars() {
        if ch == '\n' {
            in_comment = false;
            prev_char = ch;
            continue;
        }
        if in_comment {
            prev_char = ch;
            continue;
        }
        if ch == '"' && prev_char != '\\' {
            in_string = !in_string;
            prev_char = ch;
            continue;
        }
        if in_string {
            prev_char = ch;
            continue;
        }
        if ch == ';' {
            in_comment = true;
            prev_char = ch;
            continue;
        }
        match ch {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => {
                depth -= 1;
                if depth < 0 {
                    extra_close += 1;
                    depth = 0;
                }
            }
            _ => {}
        }
        prev_char = ch;
    }
    ParenBalance { extra_close, unclosed: depth.max(0) as usize }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple() {
        let result = parse("(+ 1 2)", "<test>");
        assert!(result.is_ok());
        let exprs = result.unwrap();
        assert_eq!(exprs.len(), 1);
        assert!(exprs[0].is_list());
    }

    #[test]
    fn test_parse_vector() {
        let result = parse("[1 2 3]", "<test>");
        assert!(result.is_ok());
        let exprs = result.unwrap();
        assert_eq!(exprs.len(), 1);
        assert!(exprs[0].is_vector());
    }

    #[test]
    fn test_parse_keyword() {
        let result = parse(":foo", "<test>");
        assert!(result.is_ok());
        let exprs = result.unwrap();
        assert_eq!(exprs.len(), 1);
        match &exprs[0] {
            SheafValue::Keyword(k, _) => assert_eq!(k, "foo"),
            _ => panic!("Expected keyword"),
        }
    }

    #[test]
    fn test_parse_string_escapes() {
        let result = parse(r#""hello\nworld""#, "<test>");
        assert!(result.is_ok());
        let exprs = result.unwrap();
        match &exprs[0] {
            SheafValue::String(s, _) => assert_eq!(s, "hello\nworld"),
            _ => panic!("Expected string"),
        }
    }

    #[test]
    fn test_parse_comments() {
        let result = parse("; comment\n(+ 1 2) ; another comment", "<test>");
        assert!(result.is_ok());
        let exprs = result.unwrap();
        assert_eq!(exprs.len(), 1);
    }

    #[test]
    fn test_parse_quote() {
        let result = parse("'(1 2 3)", "<test>");
        assert!(result.is_ok());
        let exprs = result.unwrap();
        assert_eq!(exprs.len(), 1);
        match &exprs[0] {
            SheafValue::Quote(inner, _) => assert!(inner.is_list()),
            _ => panic!("Expected quote"),
        }
    }
}
