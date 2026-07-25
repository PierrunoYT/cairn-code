use std::collections::HashMap;
use std::fmt;

#[allow(dead_code)]
fn write_string(f: &mut fmt::Formatter<'_>, value: &str) -> fmt::Result {
    f.write_str("\"")?;
    for character in value.chars() {
        match character {
            '"' => f.write_str("\\\"")?,
            '\\' => f.write_str("\\\\")?,
            '\u{08}' => f.write_str("\\b")?,
            '\u{0c}' => f.write_str("\\f")?,
            '\n' => f.write_str("\\n")?,
            '\r' => f.write_str("\\r")?,
            '\t' => f.write_str("\\t")?,
            character if character <= '\u{1f}' => write!(f, "\\u{:04x}", character as u32)?,
            character => write!(f, "{character}")?,
        }
    }
    f.write_str("\"")
}

#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(HashMap<String, JsonValue>),
}

impl fmt::Display for JsonValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JsonValue::Null => write!(f, "null"),
            JsonValue::Bool(b) => write!(f, "{b}"),
            JsonValue::Number(n) => {
                write!(f, "{}", serde_json::to_string(n).map_err(|_| fmt::Error)?)
            }
            JsonValue::String(s) => {
                write!(f, "{}", serde_json::to_string(s).map_err(|_| fmt::Error)?)
            }
            JsonValue::Array(arr) => {
                write!(f, "[")?;
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, "]")
            }
            JsonValue::Object(obj) => {
                write!(f, "{{")?;
                let mut first = true;
                for (k, v) in obj {
                    if !first {
                        write!(f, ",")?;
                    }
                    first = false;
                    write!(
                        f,
                        "{}:{v}",
                        serde_json::to_string(k).map_err(|_| fmt::Error)?
                    )?;
                }
                write!(f, "}}")
            }
        }
    }
}

impl JsonValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn _as_f64(&self) -> Option<f64> {
        match self {
            JsonValue::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            JsonValue::Number(n) => {
                if *n >= 0.0 && *n == n.floor() {
                    Some(*n as u64)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Vec<JsonValue>> {
        match self {
            JsonValue::Array(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&HashMap<String, JsonValue>> {
        match self {
            JsonValue::Object(m) => Some(m),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            JsonValue::Object(m) => m.get(key),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct JsonError {
    pub message: String,
    pub pos: usize,
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "JSON error at position {}: {}", self.pos, self.message)
    }
}

/// Parses `input` into a [`JsonValue`].
///
/// Backed by `serde_json`, which enforces a nesting limit — input arriving
/// from providers, MCP servers, and model tool calls is untrusted, and an
/// unbounded recursive parser would overflow the stack on deeply nested
/// payloads. Inputs over 16 MiB are rejected before parsing starts.
pub fn parse(input: &str) -> Result<JsonValue, JsonError> {
    parse_standard(input)
}

fn parse_standard(input: &str) -> Result<JsonValue, JsonError> {
    const MAX_INPUT_BYTES: usize = 16 * 1024 * 1024;

    if input.len() > MAX_INPUT_BYTES {
        return Err(JsonError {
            message: format!("input exceeds {MAX_INPUT_BYTES} byte limit"),
            pos: MAX_INPUT_BYTES,
        });
    }

    let value = serde_json::from_str(input).map_err(|error| JsonError {
        message: error.to_string(),
        pos: input
            .split_inclusive('\n')
            .take(error.line().saturating_sub(1))
            .map(str::len)
            .sum::<usize>()
            + error.column().saturating_sub(1),
    })?;
    Ok(from_serde_json(value))
}

fn from_serde_json(value: serde_json::Value) -> JsonValue {
    match value {
        serde_json::Value::Null => JsonValue::Null,
        serde_json::Value::Bool(value) => JsonValue::Bool(value),
        serde_json::Value::Number(value) => JsonValue::Number(value.as_f64().unwrap()),
        serde_json::Value::String(value) => JsonValue::String(value),
        serde_json::Value::Array(values) => {
            JsonValue::Array(values.into_iter().map(from_serde_json).collect())
        }
        serde_json::Value::Object(values) => JsonValue::Object(
            values
                .into_iter()
                .map(|(key, value)| (key, from_serde_json(value)))
                .collect(),
        ),
    }
}

pub fn serialize(val: &JsonValue) -> String {
    val.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_null() {
        let v = parse("null").unwrap();
        assert_eq!(v, JsonValue::Null);
    }

    #[test]
    fn test_parse_bool() {
        assert_eq!(parse("true").unwrap(), JsonValue::Bool(true));
        assert_eq!(parse("false").unwrap(), JsonValue::Bool(false));
    }

    #[test]
    fn test_parse_number() {
        assert_eq!(parse("42").unwrap(), JsonValue::Number(42.0));
        assert_eq!(parse("3.14").unwrap(), JsonValue::Number(3.14));
        assert_eq!(parse("-1").unwrap(), JsonValue::Number(-1.0));
    }

    #[test]
    fn test_parse_string() {
        assert_eq!(
            parse("\"hello\"").unwrap(),
            JsonValue::String("hello".into())
        );
        assert_eq!(parse("\"\"").unwrap(), JsonValue::String("".into()));
    }

    #[test]
    fn test_parse_escaped_string() {
        let v = parse("\"hello\\nworld\"").unwrap();
        assert_eq!(v, JsonValue::String("hello\nworld".into()));
    }

    #[test]
    fn test_parse_array() {
        let v = parse("[1,2,3]").unwrap();
        assert_eq!(
            v,
            JsonValue::Array(vec![
                JsonValue::Number(1.0),
                JsonValue::Number(2.0),
                JsonValue::Number(3.0),
            ])
        );
    }

    #[test]
    fn test_parse_empty_array() {
        assert_eq!(parse("[]").unwrap(), JsonValue::Array(vec![]));
    }

    #[test]
    fn test_parse_nested_array() {
        let v = parse("[[1,2],[3,4]]").unwrap();
        assert_eq!(
            v,
            JsonValue::Array(vec![
                JsonValue::Array(vec![JsonValue::Number(1.0), JsonValue::Number(2.0)]),
                JsonValue::Array(vec![JsonValue::Number(3.0), JsonValue::Number(4.0)]),
            ])
        );
    }

    #[test]
    fn test_parse_object() {
        let v = parse("{\"a\":1,\"b\":\"two\"}").unwrap();
        let obj = v.as_object().unwrap();
        assert_eq!(obj.get("a"), Some(&JsonValue::Number(1.0)));
        assert_eq!(obj.get("b"), Some(&JsonValue::String("two".into())));
    }

    #[test]
    fn test_parse_empty_object() {
        assert_eq!(parse("{}").unwrap(), JsonValue::Object(HashMap::new()));
    }

    #[test]
    fn test_parse_nested_object() {
        let v = parse("{\"outer\":{\"inner\":42}}").unwrap();
        let outer = v.get("outer").and_then(|v| v.as_object()).unwrap();
        assert_eq!(outer.get("inner"), Some(&JsonValue::Number(42.0)));
    }

    #[test]
    fn test_whitespace_tolerance() {
        let v = parse("  {  \"a\"  :  1  }  ").unwrap();
        assert_eq!(v.get("a"), Some(&JsonValue::Number(1.0)));
    }

    #[test]
    fn test_unicode_string() {
        let v = parse("\"héllo 🎉\"").unwrap();
        assert_eq!(v, JsonValue::String("héllo 🎉".into()));
    }

    #[test]
    fn test_parse_error_unclosed() {
        assert!(parse("{").is_err());
    }

    #[test]
    fn test_parse_error_trailing_comma() {
        assert!(parse("[1,]").is_err());
    }

    #[test]
    fn test_parse_rejects_trailing_data() {
        assert!(parse("null true").is_err());
    }

    #[test]
    fn test_parse_rejects_excessive_nesting() {
        let input = format!("{}null{}", "[".repeat(128), "]".repeat(128));
        assert!(parse(&input).is_err());
    }

    #[test]
    fn test_parse_rejects_oversized_input() {
        let input = format!(r#""{}""#, "a".repeat(16 * 1024 * 1024));
        assert!(parse(&input).is_err());
    }

    #[test]
    fn test_roundtrip_null() {
        let v = JsonValue::Null;
        let s = serialize(&v);
        let back = parse(&s).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn test_roundtrip_object() {
        let mut obj = HashMap::new();
        obj.insert("name".into(), JsonValue::String("test".into()));
        obj.insert("count".into(), JsonValue::Number(42.0));
        obj.insert("active".into(), JsonValue::Bool(true));
        obj.insert("data".into(), JsonValue::Null);
        let v = JsonValue::Object(obj);
        let s = serialize(&v);
        let back = parse(&s).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn test_serialize_escapes_string_characters() {
        let value = JsonValue::String(
            "quote\" slash\\ line\nreturn\rtab\tbackspace\u{0008}formfeed\u{000c}nul\u{0000}"
                .into(),
        );

        let serialized = serialize(&value);

        assert_eq!(
            serialized,
            r#""quote\" slash\\ line\nreturn\rtab\tbackspace\bformfeed\fnul\u0000""#
        );
        assert_eq!(parse(&serialized).unwrap(), value);
        assert!(serde_json::from_str::<serde_json::Value>(&serialized).is_ok());
    }

    #[test]
    fn test_serialize_escapes_object_keys() {
        let key = "quote\" slash\\ line\ncontrol\u{0001}";
        let value = JsonValue::Object(HashMap::from([(
            key.into(),
            JsonValue::String("value".into()),
        )]));

        let serialized = serialize(&value);

        assert_eq!(
            serialized,
            r#"{"quote\" slash\\ line\ncontrol\u0001":"value"}"#
        );
        assert_eq!(parse(&serialized).unwrap(), value);
        assert!(serde_json::from_str::<serde_json::Value>(&serialized).is_ok());
    }

    #[test]
    fn test_serialize_non_finite_numbers_as_null() {
        assert_eq!(serialize(&JsonValue::Number(f64::NAN)), "null");
        assert_eq!(serialize(&JsonValue::Number(f64::INFINITY)), "null");
        assert_eq!(serialize(&JsonValue::Number(f64::NEG_INFINITY)), "null");
    }

    #[test]
    fn test_roundtrip_escaped_object_keys_and_values() {
        let mut obj = HashMap::new();
        obj.insert(
            "key\"\\\n\t\u{0001}".into(),
            JsonValue::String("value\"\\\n\r\t\u{0008}\u{000c}\u{001f}".into()),
        );
        let value = JsonValue::Object(obj);

        let serialized = serialize(&value);
        let parsed = parse(&serialized).unwrap();

        assert_eq!(parsed, value);
    }

    #[test]
    fn test_as_str() {
        let v = JsonValue::String("hello".into());
        assert_eq!(v.as_str(), Some("hello"));
        assert_eq!(JsonValue::Null.as_str(), None);
    }

    #[test]
    fn test_as_u64() {
        assert_eq!(JsonValue::Number(42.0).as_u64(), Some(42));
        assert_eq!(JsonValue::Number(3.14).as_u64(), None);
    }

    #[test]
    fn test_as_array() {
        let v = JsonValue::Array(vec![JsonValue::Number(1.0)]);
        assert!(v.as_array().is_some());
        assert!(JsonValue::Null.as_array().is_none());
    }

    #[test]
    fn test_as_object() {
        let v = JsonValue::Object(HashMap::new());
        assert!(v.as_object().is_some());
        assert!(JsonValue::Null.as_object().is_none());
    }

    #[test]
    fn test_get_on_object() {
        let mut obj = HashMap::new();
        obj.insert("key".into(), JsonValue::String("val".into()));
        let v = JsonValue::Object(obj);
        assert_eq!(v.get("key").and_then(|v| v.as_str()), Some("val"));
        assert_eq!(v.get("missing"), None);
    }

    #[test]
    fn test_get_on_non_object_returns_none() {
        assert_eq!(JsonValue::Null.get("key"), None);
    }

    #[test]
    fn test_deeply_nested() {
        let input = r#"{"a":{"b":{"c":{"d":[1,2,3]}}}}"#;
        let v = parse(input).unwrap();
        let d = v
            .get("a")
            .and_then(|v| v.get("b"))
            .and_then(|v| v.get("c"))
            .and_then(|v| v.get("d"));
        assert!(d.is_some());
        let arr = d.unwrap().as_array().unwrap();
        assert_eq!(arr.len(), 3);
    }

    #[test]
    fn test_todo_tool_schema() {
        let schema = r#"{"type":"object","properties":{"todos":{"type":"array","items":{"type":"object","properties":{"content":{"type":"string"},"status":{"type":"string"},"priority":{"type":"string"}}}}},"required":["todos"]}"#;
        match parse(schema) {
            Ok(v) => {
                let obj = v.as_object().unwrap();
                assert!(obj.get("type").is_some());
                assert!(obj.get("properties").is_some());
                assert!(obj.get("required").is_some());
            }
            Err(e) => panic!("Todo schema invalid: {e}"),
        }
    }

    #[test]
    fn test_todo_tool_in_object() {
        let schema = r#"{"type":"object","properties":{"todos":{"type":"array","items":{"type":"object","properties":{"content":{"type":"string"},"status":{"type":"string"},"priority":{"type":"string"}}}}},"required":["todos"]}"#;
        let wrapped = format!(
            r#"{{"type":"function","function":{{"name":"todo_write","description":"Manage a task/todo list","parameters":{schema}}}}}"#
        );
        match parse(&wrapped) {
            Ok(v) => {
                let obj = v.as_object().unwrap();
                assert_eq!(obj.get("type").and_then(|v| v.as_str()), Some("function"));
            }
            Err(e) => panic!("Wrapped todo tool invalid: {e}"),
        }
    }
}
