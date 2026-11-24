//! Desktop launcher for HTML previews, plugin panels, links, and config files on Linux.

/// Opens a web URL or local config file using the desktop's default handler (`xdg-open`).
pub fn open(target: &str) {
    let _ = std::process::Command::new("xdg-open")
        .arg(target)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
}
