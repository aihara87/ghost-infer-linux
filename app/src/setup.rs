//! What an installed app does on its own: start its server (`ghost-infer` beside it) when nothing answers
//! yet, and offer models to download when the models folder is empty (the user picks; none is pushed).

use std::path::{Path, PathBuf};
use std::process::Child;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;

static STARTED: AtomicBool = AtomicBool::new(false);
static CHILD: Mutex<Option<Child>> = Mutex::new(None);

/// Whether this GUI process started the local `ghost-infer` server itself.
pub fn we_started_server() -> bool {
    STARTED.load(Ordering::Relaxed)
}

/// If the server spawned by this app exited before connecting, returns the reason from `server.log`.
pub fn server_exit_error() -> Option<String> {
    let mut guard = CHILD.lock().ok()?;
    let child = guard.as_mut()?;
    let status = child.try_wait().ok()??;
    *guard = None;
    let tail = data_dir()
        .and_then(|d| std::fs::read_to_string(d.join("server.log")).ok())
        .and_then(|s| s.lines().rev().find(|l| !l.trim().is_empty()).map(|l| l.trim().to_string()));
    Some(tail.unwrap_or_else(|| format!("ghost-infer exited ({status})")))
}

/// A model the first run offers, from Hugging Face.
pub struct Offer {
    pub name: &'static str,
    pub note: &'static str,
    pub file: &'static str,
    pub url: &'static str,
    pub size: u64,
}

pub const OFFERS: [Offer; 3] = [
    Offer {
        name: "Qwen3.6 35B",
        note: "Thinks before it answers. Best for coding and agent work",
        file: "Qwen3.6-35B-A3B.i1-Q3_K_M.gguf",
        url: "https://huggingface.co/mradermacher/Qwen3.6-35B-A3B-i1-GGUF/resolve/main/Qwen3.6-35B-A3B.i1-Q3_K_M.gguf",
        size: 16_764_764_896,
    },
    Offer {
        name: "Qwen3 Coder 30B",
        note: "Answers at once. Fast for code",
        file: "Qwen3-Coder-30B-A3B-Instruct.i1-Q3_K_M.gguf",
        url: "https://huggingface.co/mradermacher/Qwen3-Coder-30B-A3B-Instruct-i1-GGUF/resolve/main/Qwen3-Coder-30B-A3B-Instruct.i1-Q3_K_M.gguf",
        size: 14_711_850_016,
    },
    Offer {
        name: "Qwen3 4B",
        note: "Small and quick, for less memory",
        file: "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        url: "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        size: 2_497_281_120,
    },
];

/// `$XDG_DATA_HOME/ghost-infer` or `$HOME/.local/share/ghost-infer`: the server's and the app's data.
pub fn data_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("XDG_DATA_HOME").filter(|d| !d.is_empty()) {
        return Some(PathBuf::from(d).join("ghost-infer"));
    }
    std::env::var_os("HOME").filter(|d| !d.is_empty()).map(|d| PathBuf::from(d).join(".local").join("share").join("ghost-infer"))
}

/// Where downloaded models go.
pub fn models_dir() -> Option<PathBuf> {
    data_dir().map(|d| d.join("models"))
}

/// The server beside this program, if the app was installed with it.
pub fn own_server() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let server = exe.with_file_name("ghost-infer");
    server.exists().then_some(server)
}

/// Whether something answers at `addr`.
pub fn listening(addr: &str) -> bool {
    use std::net::ToSocketAddrs;
    addr.to_socket_addrs().ok().and_then(|mut a| a.next()).is_some_and(|a| std::net::TcpStream::connect_timeout(&a, Duration::from_millis(300)).is_ok())
}

/// Starts the server beside this program, without a window, if nothing answers at `addr` yet: `model` (if
/// passed on the CLI) or `last` picked model, the models folder and `extra` folders.
pub fn start_server(addr: &str, extra: &[String], last: Option<&str>, model: Option<&Path>, extra_args: &[String]) {
    let (Some(server), Some(data), Some(models)) = (own_server(), data_dir(), models_dir()) else { return };
    if listening(addr) {
        return;
    }
    let _ = std::fs::create_dir_all(&models);
    let port = addr.rsplit(':').next().unwrap_or("8080").to_string();
    let mut dirs = vec![models];
    for d in extra.iter().map(PathBuf::from).chain(model.and_then(|p| p.parent().filter(|p| !p.as_os_str().is_empty()).map(Path::to_path_buf))) {
        if !dirs.contains(&d) {
            dirs.push(d);
        }
    }
    let mut args: Vec<std::ffi::OsString> = vec!["serve".into()];
    if let Some(path) = model.map(Path::to_path_buf).or_else(|| last.and_then(|id| dirs.iter().map(|d| d.join(format!("{id}.gguf"))).find(|p| p.exists()))) {
        args.push(path.into());
    }
    let has = |flag: &str| extra_args.iter().any(|a| a == flag);
    args.push("--gpu".into());
    if !has("--ctx") {
        args.push("--ctx".into());
        args.push("65536".into());
    }
    if !has("--kv") {
        args.push("--kv".into());
        args.push("int8".into());
    }
    args.push("--port".into());
    args.push(port.into());
    for a in extra_args {
        args.push(a.into());
    }
    for d in &dirs {
        args.push("--models".into());
        args.push(d.into());
    }
    // voice calls: a voice folder (maya1's, with snac.gguf, or CSM's three files) and a Whisper file found in the
    // models folders
    let voice_folder = |p: &PathBuf| p.join("snac.gguf").is_file() || ["backbone.gguf", "decoder.gguf", "mimi.gguf"].iter().all(|f| p.join(f).is_file());
    if let Some(voice) = dirs.iter().flat_map(|d| std::fs::read_dir(d).into_iter().flatten().flatten()).map(|e| e.path()).find(voice_folder) {
        args.push("--tts".into());
        args.push(voice.into());
    }
    let whisper = |d: &PathBuf| std::fs::read_dir(d).into_iter().flatten().flatten().map(|e| e.path()).find(|p| p.extension().is_some_and(|x| x == "gguf") && p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.to_lowercase().starts_with("whisper")));
    if let Some(stt) = dirs.iter().find_map(|d| whisper(&d.join("whisper")).or_else(|| whisper(d))) {
        args.push("--stt".into());
        args.push(stt.into());
    }
    let log = |name: &str| std::fs::File::create(data.join(name)).map(std::process::Stdio::from).unwrap_or_else(|_| std::process::Stdio::null());
    if let Ok(child) = std::process::Command::new(server).args(args).stdin(std::process::Stdio::null()).stdout(log("server-out.log")).stderr(log("server.log")).spawn() {
        STARTED.store(true, Ordering::Relaxed);
        if let Ok(mut g) = CHILD.lock() {
            *g = Some(child);
        }
    }
}

/// "6.2 of 16.8 GB".
pub fn progress_text(bytes: f64, total: f64) -> String {
    crate::tf!("{:.1} of {:.1} GB", "{:.1} / {:.1} GB", bytes / 1e9, total / 1e9)
}
