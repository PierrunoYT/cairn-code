//! TUI palettes ported from zero / Grok Build (`internal/tui/theme_palettes.go`).
//! Includes the full dark registry plus light and solarized-light. Dune uses the
//! dark Claude Code colorblind (daltonized) palette from the upstream
//! `fix/dune-claude-dark-theme` PR (zero main still ships light sand Dune).

use ratatui::style::{Color, Modifier, Style};

/// One named dark theme and the ratatui styles the TUI reads.
#[derive(Clone)]
pub struct Theme {
    pub name: &'static str,
    pub label: &'static str,
    pub ink: Style,
    pub muted: Style,
    #[allow(dead_code)]
    pub faint: Style,
    pub faintest: Style,
    pub accent: Style,
    pub accent_fg: Style,
    pub red: Style,
    pub green: Style,
    #[allow(dead_code)]
    pub amber: Style,
    #[allow(dead_code)]
    pub blue: Style,
    /// Selected picker row: selection background + primary ink.
    pub selected: Style,
    /// Accent-filled chip (badge): onAccent on accent.
    #[allow(dead_code)]
    pub badge: Style,
}

struct Palette {
    panel: &'static str,
    prompt_bg: &'static str,
    line: &'static str,
    line2: &'static str,
    ink: &'static str,
    muted: &'static str,
    faint: &'static str,
    faintest: &'static str,
    accent: &'static str,
    green: &'static str,
    red: &'static str,
    amber: &'static str,
    blue: &'static str,
    sel_bg: &'static str,
    on_accent: &'static str,
}

fn rgb(hex: &str) -> Color {
    let h = hex.trim_start_matches('#');
    if h.len() < 6 {
        return Color::White;
    }
    let r = u8::from_str_radix(&h[0..2], 16).unwrap_or(255);
    let g = u8::from_str_radix(&h[2..4], 16).unwrap_or(255);
    let b = u8::from_str_radix(&h[4..6], 16).unwrap_or(255);
    Color::Rgb(r, g, b)
}

fn fg(hex: &str) -> Style {
    Style::new().fg(rgb(hex))
}

fn build(name: &'static str, label: &'static str, p: &Palette) -> Theme {
    Theme {
        name,
        label,
        ink: fg(p.ink),
        muted: fg(p.muted),
        faint: fg(p.faint),
        faintest: fg(p.faintest),
        accent: fg(p.accent).add_modifier(Modifier::BOLD),
        accent_fg: fg(p.accent),
        red: fg(p.red),
        green: fg(p.green),
        amber: fg(p.amber),
        blue: fg(p.blue),
        selected: Style::new().bg(rgb(p.sel_bg)).fg(rgb(p.ink)),
        badge: Style::new()
            .bg(rgb(p.accent))
            .fg(rgb(p.on_accent))
            .add_modifier(Modifier::BOLD),
        // Keep panel/prompt/line available for future surface fills without unused-field noise
        // by referencing them in a const-friendly way via debug assertion helpers.
    }
}

// Silence unused palette field warnings for surface tokens we keep for parity with zero
// but do not paint full-bleed (cairn does not paint terminal background).
#[allow(dead_code)]
fn _surface_tokens(p: &Palette) -> (Color, Color, Color, Color) {
    (rgb(p.panel), rgb(p.prompt_bg), rgb(p.line), rgb(p.line2))
}

// --- Dark palettes (from zero) ---

const DARK: Palette = Palette {
    panel: "#0e0e10",
    prompt_bg: "#262626",
    line: "#242429",
    line2: "#414147",
    ink: "#ececee",
    muted: "#9a9aa2",
    faint: "#8a8a92",
    faintest: "#7c7c82",
    accent: "#caff3f",
    green: "#5dd1a4",
    red: "#ff7a7a",
    amber: "#ffc25c",
    blue: "#7db4ff",
    sel_bg: "#32401b",
    on_accent: "#000000",
};

const DRACULA: Palette = Palette {
    panel: "#282a36",
    prompt_bg: "#383c4d",
    line: "#363a4b",
    line2: "#484c62",
    ink: "#f8f8f2",
    muted: "#b9bccb",
    faint: "#a2a5b8",
    faintest: "#9195ac",
    accent: "#bd93f9",
    green: "#50fa7b",
    red: "#ff5555",
    amber: "#ffb86c",
    blue: "#8be9fd",
    sel_bg: "#504482",
    on_accent: "#000000",
};

const NORD: Palette = Palette {
    panel: "#3b4252",
    prompt_bg: "#464f62",
    line: "#434c5e",
    line2: "#4c566a",
    ink: "#eceff4",
    muted: "#c8cfda",
    faint: "#b4bdcb",
    faintest: "#a5afc1",
    accent: "#88c0d0",
    green: "#a3be8c",
    red: "#bf616a",
    amber: "#d08770",
    blue: "#81a1c1",
    sel_bg: "#40688a",
    on_accent: "#000000",
};

const GRUVBOX: Palette = Palette {
    panel: "#32302f",
    prompt_bg: "#3c3836",
    line: "#504945",
    line2: "#665c54",
    ink: "#ebdbb2",
    muted: "#c9b99a",
    faint: "#b7a78d",
    faintest: "#a89984",
    accent: "#8ec07c",
    green: "#b8bb26",
    red: "#fb4934",
    amber: "#fabd2f",
    blue: "#83a598",
    sel_bg: "#3d4e30",
    on_accent: "#000000",
};

const TOKYO_NIGHT: Palette = Palette {
    panel: "#1e2030",
    prompt_bg: "#2c3149",
    line: "#262a3d",
    line2: "#3b4261",
    ink: "#c8d3f5",
    muted: "#a9b1d0",
    faint: "#9099b2",
    faintest: "#838ba8",
    accent: "#82aaff",
    green: "#c3e88d",
    red: "#ff757f",
    amber: "#ffc777",
    blue: "#86e1fc",
    sel_bg: "#2a385b",
    on_accent: "#000000",
};

// Catppuccin: all four official flavors (https://github.com/catppuccin/catppuccin).
// Mapped to cairn tokens: base->panel, mantle->prompt_bg, surface0/1->line/line2,
// text->ink, subtext0->muted, overlay2/1->faint/faintest, mauve->accent.

/// Catppuccin Mocha (darkest).
const CATPPUCCIN_MOCHA: Palette = Palette {
    panel: "#1e1e2e",
    prompt_bg: "#181825",
    line: "#313244",
    line2: "#45475a",
    ink: "#cdd6f4",
    muted: "#a6adc8",
    faint: "#9399b2",
    faintest: "#7f849c",
    accent: "#cba6f7",
    green: "#a6e3a1",
    red: "#f38ba8",
    amber: "#f9e2af",
    blue: "#89b4fa",
    sel_bg: "#45475a",
    on_accent: "#11111b",
};

/// Catppuccin Macchiato.
const CATPPUCCIN_MACCHIATO: Palette = Palette {
    panel: "#24273a",
    prompt_bg: "#1e2030",
    line: "#363a4f",
    line2: "#494d64",
    ink: "#cad3f5",
    muted: "#a5adcb",
    faint: "#939ab7",
    faintest: "#8087a2",
    accent: "#c6a0f6",
    green: "#a6da95",
    red: "#ed8796",
    amber: "#eed49f",
    blue: "#8aadf4",
    sel_bg: "#494d64",
    on_accent: "#181926",
};

/// Catppuccin Frappé.
const CATPPUCCIN_FRAPPE: Palette = Palette {
    panel: "#303446",
    prompt_bg: "#292c3c",
    line: "#414559",
    line2: "#51576d",
    ink: "#c6d0f5",
    muted: "#a5adce",
    faint: "#949cbb",
    faintest: "#838ba7",
    accent: "#ca9ee6",
    green: "#a6d189",
    red: "#e78284",
    amber: "#e5c890",
    blue: "#8caaee",
    sel_bg: "#51576d",
    on_accent: "#232634",
};

/// Catppuccin Latte (light).
const CATPPUCCIN_LATTE: Palette = Palette {
    panel: "#eff1f5",
    prompt_bg: "#e6e9ef",
    line: "#ccd0da",
    line2: "#bcc0cc",
    ink: "#4c4f69",
    muted: "#6c6f85",
    faint: "#7c7f93",
    faintest: "#8c8fa1",
    accent: "#8839ef",
    green: "#40a02b",
    red: "#d20f39",
    amber: "#df8e1d",
    blue: "#1e66f5",
    sel_bg: "#ccd0da",
    on_accent: "#eff1f5",
};

const ONE_DARK: Palette = Palette {
    panel: "#2e323b",
    prompt_bg: "#3a3f4a",
    line: "#393f4a",
    line2: "#4b525f",
    ink: "#abb2bf",
    muted: "#a2a9b6",
    faint: "#9aa1af",
    faintest: "#969cab",
    accent: "#61afef",
    green: "#98c379",
    red: "#e06c75",
    amber: "#e5c07b",
    blue: "#56b6c2",
    sel_bg: "#354256",
    on_accent: "#000000",
};

const SOLARIZED_DARK: Palette = Palette {
    panel: "#073642",
    prompt_bg: "#0b3b46",
    line: "#123f48",
    line2: "#4b636c",
    ink: "#cdd6d6",
    muted: "#a9b3b3",
    faint: "#9ba5a5",
    faintest: "#929c9c",
    accent: "#3bb3a6",
    green: "#859900",
    red: "#dc322f",
    amber: "#b58900",
    blue: "#268bd2",
    sel_bg: "#17505a",
    on_accent: "#000000",
};

const ROSE_PINE: Palette = Palette {
    panel: "#1f1d2e",
    prompt_bg: "#2f2b47",
    line: "#2b2840",
    line2: "#403d52",
    ink: "#e0def4",
    muted: "#a8a3c0",
    faint: "#928ea9",
    faintest: "#8985a0",
    accent: "#ebbcba",
    green: "#31748f",
    red: "#eb6f92",
    amber: "#f6c177",
    blue: "#9ccfd8",
    sel_bg: "#44415a",
    on_accent: "#000000",
};

const EVERFOREST: Palette = Palette {
    panel: "#333c43",
    prompt_bg: "#3d484d",
    line: "#414b52",
    line2: "#55636b",
    ink: "#d3c6aa",
    muted: "#b0bab0",
    faint: "#a4aea3",
    faintest: "#9ca99b",
    accent: "#a7c080",
    green: "#83c092",
    red: "#e67e80",
    amber: "#dbbc7f",
    blue: "#7fbbb3",
    sel_bg: "#3b482e",
    on_accent: "#000000",
};

const NEON: Palette = Palette {
    panel: "#050b06",
    prompt_bg: "#0c180d",
    line: "#1c3820",
    line2: "#2c5230",
    ink: "#c9ffd2",
    muted: "#80db8f",
    faint: "#6eca7d",
    faintest: "#74c468",
    accent: "#00e5c8",
    green: "#39ff6a",
    red: "#ff4d6d",
    amber: "#f4ff3a",
    blue: "#22e0ff",
    sel_bg: "#123a1e",
    on_accent: "#001410",
};

/// Dune: dark Claude Code colorblind (daltonized) palette from zero branch
/// `fix/dune-claude-dark-theme` (not the old light sand palette on zero main).
const DUNE: Palette = Palette {
    panel: "#0e0e10",
    prompt_bg: "#262626",
    line: "#242429",
    line2: "#414147",
    ink: "#ececee",
    muted: "#ccccd2",
    faint: "#b8b8c0",
    faintest: "#a0a0a8",
    accent: "#ff9628",
    green: "#3399ff", // success as blue under colorblind mode
    red: "#ff6666",
    amber: "#ffcc00",
    blue: "#99ccff",
    sel_bg: "#191c1f",
    on_accent: "#000000",
};

/// light: warm cream surface with olive-lime accent (zero lightPalette).
const LIGHT: Palette = Palette {
    panel: "#efebd4",
    prompt_bg: "#e3ddc2",
    line: "#d8d2bd",
    line2: "#b7b199",
    ink: "#22201a",
    muted: "#4b5149",
    faint: "#575e55",
    faintest: "#636a61",
    accent: "#54700a",
    green: "#1e725c",
    red: "#c02434",
    amber: "#8a5f00",
    blue: "#1f66c0",
    sel_bg: "#d4e08f",
    on_accent: "#ffffff",
};

/// Solarized Light: base3/base2 cream with dark-on-light accent wheel.
const SOLARIZED_LIGHT: Palette = Palette {
    panel: "#eee8d5",
    prompt_bg: "#e1d9be",
    line: "#d8d1bc",
    line2: "#c0b89e",
    ink: "#304049",
    muted: "#495b61",
    faint: "#506469",
    faintest: "#576b72",
    accent: "#0c665c",
    green: "#859900",
    red: "#dc322f",
    amber: "#7a5c00",
    blue: "#268bd2",
    sel_bg: "#a6d6c4",
    on_accent: "#ffffff",
};

/// All selectable themes. Dark group first (matching zero registry order), then
/// light themes. Catppuccin ships all four official flavors (Mocha is the zero
/// `catppuccin` entry; Latte is light).
pub fn all_themes() -> Vec<Theme> {
    // Touch surface tokens so palette fields stay intentional parity with zero.
    let _ = _surface_tokens(&DARK);
    vec![
        build("dark", "dark", &DARK),
        build("dracula", "Dracula", &DRACULA),
        build("nord", "Nord", &NORD),
        build("gruvbox", "Gruvbox", &GRUVBOX),
        build("tokyo-night", "Tokyo Night", &TOKYO_NIGHT),
        build("catppuccin-mocha", "Catppuccin Mocha", &CATPPUCCIN_MOCHA),
        build(
            "catppuccin-macchiato",
            "Catppuccin Macchiato",
            &CATPPUCCIN_MACCHIATO,
        ),
        build("catppuccin-frappe", "Catppuccin Frappé", &CATPPUCCIN_FRAPPE),
        build("one-dark", "One Dark", &ONE_DARK),
        build("solarized-dark", "Solarized Dark", &SOLARIZED_DARK),
        build("rose-pine", "Rosé Pine", &ROSE_PINE),
        build("everforest", "Everforest", &EVERFOREST),
        build("neon", "Neon", &NEON),
        build("dune", "Dune", &DUNE),
        // Light themes (zero registry order after the dark group).
        build("light", "light", &LIGHT),
        build("solarized-light", "Solarized Light", &SOLARIZED_LIGHT),
        build("catppuccin-latte", "Catppuccin Latte", &CATPPUCCIN_LATTE),
    ]
}

pub fn default_theme() -> Theme {
    build("dark", "dark", &DARK)
}

/// Resolve a theme name (case/space-insensitive), without falling back.
/// `catppuccin` alone aliases Mocha for backward compatibility.
pub fn lookup_exact(name: &str) -> Option<Theme> {
    let mut key = name.trim().to_ascii_lowercase().replace(' ', "-");
    // ASCII fold: treat "frappé" typed without accent as frappe
    if key == "catppuccin-frappé" || key == "catppuccin-frappè" {
        key = "catppuccin-frappe".into();
    }
    if key == "catppuccin" {
        key = "catppuccin-mocha".into();
    }
    all_themes().into_iter().find(|t| t.name == key)
}

/// Resolve a theme name (case/space-insensitive). Falls back to dark.
/// `catppuccin` alone aliases Mocha for backward compatibility.
pub fn lookup(name: &str) -> Theme {
    lookup_exact(name).unwrap_or_else(default_theme)
}

pub fn theme_names() -> Vec<&'static str> {
    all_themes().into_iter().map(|t| t.name).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_expected_names() {
        let names = theme_names();
        // Every Grok Build / zero themeRegistry name is present (catppuccin via mocha alias).
        for required in [
            "dark",
            "dracula",
            "nord",
            "gruvbox",
            "tokyo-night",
            "catppuccin-mocha",
            "one-dark",
            "solarized-dark",
            "rose-pine",
            "everforest",
            "neon",
            "light",
            "solarized-light",
            "dune",
        ] {
            assert!(names.contains(&required), "missing theme {required}");
        }
        // Extra Catppuccin flavors beyond zero's single mocha entry.
        assert!(names.contains(&"catppuccin-macchiato"));
        assert!(names.contains(&"catppuccin-frappe"));
        assert!(names.contains(&"catppuccin-latte"));
        // Light group after dark group: light before solarized-light before latte.
        let light_i = names.iter().position(|n| *n == "light").unwrap();
        let solar_i = names.iter().position(|n| *n == "solarized-light").unwrap();
        let latte_i = names.iter().position(|n| *n == "catppuccin-latte").unwrap();
        assert!(light_i < solar_i && solar_i < latte_i);
    }

    #[test]
    fn lookup_is_case_insensitive() {
        assert_eq!(lookup("Tokyo Night").name, "tokyo-night");
        assert_eq!(lookup("DUNE").name, "dune");
        assert_eq!(lookup("nope").name, "dark");
    }

    #[test]
    fn catppuccin_alias_and_all_four_flavors() {
        assert_eq!(lookup("catppuccin").name, "catppuccin-mocha");
        assert_eq!(lookup("Catppuccin Mocha").name, "catppuccin-mocha");
        assert_eq!(lookup("catppuccin-macchiato").name, "catppuccin-macchiato");
        assert_eq!(lookup("catppuccin-frappe").name, "catppuccin-frappe");
        assert_eq!(lookup("catppuccin-latte").name, "catppuccin-latte");
        // Latte is light: ink should be dark purple-gray, not near-white
        assert!(matches!(
            lookup("catppuccin-latte").ink.fg,
            Some(Color::Rgb(0x4c, 0x4f, 0x69))
        ));
        // Mocha mauve accent
        assert!(matches!(
            lookup("catppuccin-mocha").accent_fg.fg,
            Some(Color::Rgb(0xcb, 0xa6, 0xf7))
        ));
    }

    #[test]
    fn dune_uses_dark_canvas_not_sand() {
        let t = lookup("dune");
        // Accent is brand orange from the dark daltonized palette
        assert!(matches!(t.accent_fg.fg, Some(Color::Rgb(0xff, 0x96, 0x28))));
    }

    #[test]
    fn light_and_solarized_light_are_dark_on_light() {
        let light = lookup("light");
        assert!(matches!(light.ink.fg, Some(Color::Rgb(0x22, 0x20, 0x1a))));
        assert!(matches!(
            light.accent_fg.fg,
            Some(Color::Rgb(0x54, 0x70, 0x0a))
        ));
        let solar = lookup("Solarized Light");
        assert_eq!(solar.name, "solarized-light");
        assert!(matches!(solar.ink.fg, Some(Color::Rgb(0x30, 0x40, 0x49))));
        assert!(matches!(
            solar.accent_fg.fg,
            Some(Color::Rgb(0x0c, 0x66, 0x5c))
        ));
    }

    #[test]
    fn rgb_parses_hex() {
        assert_eq!(rgb("#caff3f"), Color::Rgb(0xca, 0xff, 0x3f));
    }
}
