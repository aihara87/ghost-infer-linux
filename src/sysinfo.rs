//! A snapshot of this computer for the chat page's get_system_info tool (GET /api/system): CPU, memory,
//! GPU, operating system and drives. The GPU is read through NVML (`libnvidia-ml.so.1`, installed with the
//! NVIDIA driver), which works from any thread, unlike CUDA's memory queries.

use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::OnceLock;

use crate::json::{Json, num, obj, str};

const GB: f64 = 1073741824.0;

fn gb(bytes: u64) -> Json {
    num((bytes as f64 / GB * 10.0).round() / 10.0)
}

/// Where ghost-infer keeps its state and caches (`$XDG_DATA_HOME/ghost-infer` or
/// `$HOME/.local/share/ghost-infer`).
pub fn data_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("XDG_DATA_HOME").filter(|d| !d.is_empty()) {
        return Some(PathBuf::from(d).join("ghost-infer"));
    }
    std::env::var_os("HOME").filter(|d| !d.is_empty()).map(|d| PathBuf::from(d).join(".local").join("share").join("ghost-infer"))
}

pub fn data_dir_or_err() -> Result<PathBuf, String> {
    data_dir().ok_or_else(|| "HOME / XDG_DATA_HOME is not set".into())
}

/// Everything as JSON, after `first` (the server's own part); parts that cannot be read are left out
/// rather than failing the whole report.
pub fn report(first: Vec<(&'static str, Json)>) -> Json {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);
    let mut fields = first;
    fields.push(("cpu_threads", num(threads as f64)));
    linux::fill(&mut fields);
    obj(fields)
}

/// Physical memory available to programs now, in bytes (free pages and reusable file cache), if known.
pub fn available_ram() -> Option<u64> {
    linux::available_ram()
}

#[repr(C)]
struct NvmlMemory {
    total: u64,
    free: u64,
    used: u64,
}
#[repr(C)]
struct NvmlUtilization {
    gpu: u32,
    memory: u32,
}

#[allow(clippy::type_complexity)]
struct Nvml {
    count: unsafe extern "system" fn(*mut u32) -> i32,
    handle: unsafe extern "system" fn(u32, *mut *mut c_void) -> i32,
    name: unsafe extern "system" fn(*mut c_void, *mut u8, u32) -> i32,
    memory: unsafe extern "system" fn(*mut c_void, *mut NvmlMemory) -> i32,
    temperature: unsafe extern "system" fn(*mut c_void, u32, *mut u32) -> i32,
    utilization: unsafe extern "system" fn(*mut c_void, *mut NvmlUtilization) -> i32,
    driver: unsafe extern "system" fn(*mut u8, u32) -> i32,
}

fn c_string(buf: &[u8]) -> String {
    String::from_utf8_lossy(&buf[..buf.iter().position(|&b| b == 0).unwrap_or(buf.len())]).into_owned()
}

fn nvml_with(open: fn() -> Option<*mut c_void>, sym: fn(*mut c_void, &str) -> Option<*mut c_void>) -> Option<Vec<Json>> {
    static LIB: OnceLock<Option<Nvml>> = OnceLock::new();
    let n = LIB
        .get_or_init(|| unsafe {
            let lib = open()?;
            let get = |name: &str| sym(lib, name);
            let init: unsafe extern "system" fn() -> i32 = std::mem::transmute(get("nvmlInit_v2")?);
            if init() != 0 {
                return None;
            }
            Some(Nvml {
                count: std::mem::transmute(get("nvmlDeviceGetCount_v2")?),
                handle: std::mem::transmute(get("nvmlDeviceGetHandleByIndex_v2")?),
                name: std::mem::transmute(get("nvmlDeviceGetName")?),
                memory: std::mem::transmute(get("nvmlDeviceGetMemoryInfo")?),
                temperature: std::mem::transmute(get("nvmlDeviceGetTemperature")?),
                utilization: std::mem::transmute(get("nvmlDeviceGetUtilizationRates")?),
                driver: std::mem::transmute(get("nvmlSystemGetDriverVersion")?),
            })
        })
        .as_ref()?;
    let mut driver = [0u8; 96];
    let driver = if unsafe { (n.driver)(driver.as_mut_ptr(), driver.len() as u32) } == 0 { c_string(&driver) } else { String::new() };
    let mut count = 0u32;
    if unsafe { (n.count)(&mut count) } != 0 {
        return None;
    }
    Some(
        (0..count)
            .filter_map(|i| {
                let mut dev = std::ptr::null_mut();
                if unsafe { (n.handle)(i, &mut dev) } != 0 {
                    return None;
                }
                let mut name = [0u8; 96];
                unsafe { (n.name)(dev, name.as_mut_ptr(), name.len() as u32) };
                let mut g = vec![("name", str(&c_string(&name))), ("driver", str(&driver))];
                let mut mem = NvmlMemory { total: 0, free: 0, used: 0 };
                if unsafe { (n.memory)(dev, &mut mem) } == 0 {
                    g.push(("vram_total_gb", gb(mem.total)));
                    g.push(("vram_used_gb", gb(mem.used)));
                    g.push(("vram_free_gb", gb(mem.free)));
                }
                let mut temp = 0u32;
                if unsafe { (n.temperature)(dev, 0, &mut temp) } == 0 {
                    g.push(("temperature_c", num(temp as f64)));
                }
                let mut util = NvmlUtilization { gpu: 0, memory: 0 };
                if unsafe { (n.utilization)(dev, &mut util) } == 0 {
                    g.push(("load_percent", num(util.gpu as f64)));
                }
                Some(obj(g))
            })
            .collect(),
    )
}

mod linux {
    use super::{Json, c_void, gb, num, nvml_with, obj, str};
    use std::ffi::CString;

    #[link(name = "dl")]
    unsafe extern "C" {
        fn dlopen(name: *const i8, flags: i32) -> *mut c_void;
        fn dlsym(lib: *mut c_void, name: *const i8) -> *mut c_void;
    }
    unsafe extern "C" {
        fn statvfs(path: *const i8, buf: *mut u64) -> i32;
        fn sysconf(name: i32) -> isize;
    }

    fn meminfo() -> (Option<u64>, Option<u64>) {
        let Ok(text) = std::fs::read_to_string("/proc/meminfo") else { return (None, None) };
        let kb = |key: &str| {
            text.lines()
                .find_map(|l| l.strip_prefix(key))
                .and_then(|r| r.split_whitespace().next()?.parse::<u64>().ok())
                .map(|v| v * 1024)
        };
        (kb("MemTotal:"), kb("MemAvailable:"))
    }

    pub fn available_ram() -> Option<u64> {
        meminfo().1
    }

    fn nvml_open() -> Option<*mut c_void> {
        let c = CString::new("libnvidia-ml.so.1").ok()?;
        let p = unsafe { dlopen(c.as_ptr(), 2 /* RTLD_NOW */) };
        (!p.is_null()).then_some(p)
    }

    fn nvml_sym(lib: *mut c_void, name: &str) -> Option<*mut c_void> {
        let c = CString::new(name).ok()?;
        let p = unsafe { dlsym(lib, c.as_ptr()) };
        (!p.is_null()).then_some(p)
    }

    pub fn fill(f: &mut Vec<(&'static str, Json)>) {
        if let Ok(cpu) = std::fs::read_to_string("/proc/cpuinfo") {
            let mut model = None;
            let mut mhz = None;
            let mut cores: Vec<(String, String)> = Vec::new();
            let (mut pid, mut cid) = (String::new(), String::new());
            for line in cpu.lines() {
                if line.trim().is_empty() {
                    if !cid.is_empty() && !cores.iter().any(|p| p.0 == pid && p.1 == cid) {
                        cores.push((pid.clone(), cid.clone()));
                    }
                    pid.clear();
                    cid.clear();
                    continue;
                }
                let Some((k, v)) = line.split_once(':') else { continue };
                let (k, v) = (k.trim(), v.trim());
                match k {
                    "model name" if model.is_none() => model = Some(v.to_string()),
                    "cpu MHz" if mhz.is_none() => mhz = v.parse::<f64>().ok(),
                    "physical id" => pid = v.to_string(),
                    "core id" => cid = v.to_string(),
                    _ => {}
                }
            }
            if !cid.is_empty() && !cores.iter().any(|p| p.0 == pid && p.1 == cid) {
                cores.push((pid, cid));
            }
            if let Some(m) = model {
                f.push(("cpu", str(&m)));
            }
            if !cores.is_empty() {
                f.push(("cpu_cores", num(cores.len() as f64)));
            }
            if let Some(m) = mhz {
                f.push(("cpu_mhz", num(m.round())));
            }
        }

        let (total, avail) = meminfo();
        if let Some(t) = total {
            f.push(("ram_total_gb", gb(t)));
        }
        if let Some(a) = avail {
            f.push(("ram_available_gb", gb(a)));
            if let Some(t) = total.filter(|&t| t > 0) {
                f.push(("ram_in_use_percent", num(((t.saturating_sub(a)) as f64 / t as f64 * 100.0).round())));
            }
        }
        if let Ok(statm) = std::fs::read_to_string("/proc/self/statm")
            && let Some(rss_pages) = statm.split_whitespace().nth(1).and_then(|s| s.parse::<u64>().ok())
        {
            let page = unsafe { sysconf(30 /* _SC_PAGESIZE */) }.max(4096) as u64;
            f.push(("ghost_infer_ram_gb", gb(rss_pages * page)));
        }

        let os_name = std::fs::read_to_string("/etc/os-release")
            .ok()
            .and_then(|t| t.lines().find_map(|l| l.strip_prefix("PRETTY_NAME=")).map(|v| v.trim_matches('"').to_string()))
            .unwrap_or_else(|| std::env::consts::OS.to_string());
        f.push(("os", str(&os_name)));

        if let Some(gpu) = nvml_with(nvml_open, nvml_sym) {
            f.push(("gpus", Json::Arr(gpu)));
        }

        if let Ok(mounts) = std::fs::read_to_string("/proc/mounts") {
            let mut seen = Vec::new();
            let mut drives = Vec::new();
            for line in mounts.lines() {
                let mut cols = line.split_whitespace();
                let (Some(dev), Some(mp)) = (cols.next(), cols.next()) else { continue };
                if !dev.starts_with("/dev/") || dev.starts_with("/dev/loop") || seen.iter().any(|s| s == dev) {
                    continue;
                }
                let mp_unescaped = mp.replace("\\040", " ");
                let Ok(cpath) = CString::new(mp_unescaped.as_str()) else { continue };
                let mut vfs = [0u64; 14]; // sizeof(struct statvfs) == 112
                if unsafe { statvfs(cpath.as_ptr(), vfs.as_mut_ptr()) } == 0 && vfs[2] > 0 {
                    seen.push(dev.to_string());
                    let bsize = if vfs[1] > 0 { vfs[1] } else { vfs[0] };
                    drives.push(obj(vec![
                        ("drive", str(&mp_unescaped)),
                        ("total_gb", gb(vfs[2].saturating_mul(bsize))),
                        ("free_gb", gb(vfs[4].saturating_mul(bsize))),
                    ]));
                }
            }
            f.push(("drives", Json::Arr(drives)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linux_sysinfo_populates_fields() {
        assert!(data_dir().is_some());
        assert!(available_ram().is_some_and(|r| r > 0));
        let j = report(vec![]);
        assert!(j.get("cpu").and_then(Json::as_str).is_some());
        assert!(j.get("cpu_cores").and_then(Json::as_f64).is_some_and(|c| c >= 1.0));
        assert!(j.get("ram_total_gb").and_then(Json::as_f64).is_some_and(|r| r > 0.0));
        assert!(j.get("os").and_then(Json::as_str).is_some());
        assert!(j.get("gpus").and_then(Json::as_array).is_some_and(|g| !g.is_empty()));
    }
}
