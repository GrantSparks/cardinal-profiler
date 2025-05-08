use sysinfo::System;

/// Returns the available memory in GB
pub fn available_memory_gb() -> usize {
    let mut sys = System::new_all();
    sys.refresh_memory();
    (sys.available_memory() / 1_000_000_000) as usize // Convert bytes to GB
}

/// Returns the total memory in GB
pub fn total_memory_gb() -> usize {
    let mut sys = System::new_all();
    sys.refresh_memory();
    (sys.total_memory() / 1_000_000_000) as usize // Convert bytes to GB
}

/// Determines an optimal value for end-to-end concurrency based on system specs
pub fn heuristic_parallel() -> usize {
    let cores = num_cpus::get_physical().max(1);
    let memory_gb = available_memory_gb().max(1);

    // Estimate based on cores and memory:
    // - Use physical cores to avoid hyperthreading overhead
    // - Consider memory limits (assume ~0.5GB per task is reasonable)
    // - Never go below 1 or above a reasonable cap
    let by_cores = cores * 2;
    let by_memory = memory_gb * 2; // ~0.5 GB per task

    std::cmp::min(by_cores, by_memory).clamp(1, 16)
}

/// Determines an optimal value for API call concurrency based on system specs and parallel value
pub fn heuristic_api_parallel(parallel: usize) -> usize {
    // API parallelism shouldn't exceed overall parallelism
    // and should be capped at a reasonable value to avoid rate limits
    std::cmp::min(parallel, 8).max(1)
}

/// Parse a string value into a usize, handling "auto" or numeric values
pub fn parse_auto_or_number(value: &str, auto_fn: impl Fn() -> usize) -> Result<usize, String> {
    if value.trim().eq_ignore_ascii_case("auto") {
        Ok(auto_fn())
    } else {
        value
            .parse::<usize>()
            .map_err(|_| format!("Failed to parse '{}' as a positive number", value))
    }
}
