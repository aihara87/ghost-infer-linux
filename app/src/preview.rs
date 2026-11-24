//! Embedded preview pane stub for Linux (pages open in the default browser instead).

use iced::Rectangle;

pub fn exists() -> bool {
    false
}

pub fn create(_: &dyn iced::window::Window, _: &str) -> Result<(), String> {
    Err("embedded preview is unavailable; use Open in browser".into())
}

pub fn navigate(_: &str) {}
pub fn set_visible(_: bool) {}
pub fn focus_app() {}
pub fn place(_: Rectangle) {}
