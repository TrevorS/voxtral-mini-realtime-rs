//! Chrome tracing integration for profiling.
//!
//! Feature-gated behind `profiling`. When enabled, `init()` sets up a
//! `tracing-chrome` subscriber that writes `trace.json` to the current
//! directory. When disabled, `init()` returns `None` and all tracing
//! spans are zero-cost no-ops.

/// Guard that flushes the trace file on drop.
///
/// When the `profiling` feature is disabled, this is a zero-size type.
pub struct ProfilingGuard {
    #[cfg(feature = "profiling")]
    _guard: tracing_chrome::FlushGuard,
}

/// Initialize chrome tracing profiling.
///
/// Returns `Some(ProfilingGuard)` when the `profiling` feature is enabled,
/// `None` otherwise. Keep the guard alive for the duration of profiling —
/// the trace file is flushed when it drops.
pub fn init() -> Option<ProfilingGuard> {
    #[cfg(feature = "profiling")]
    {
        let (chrome_layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
            .file("trace.json".to_string())
            .include_args(true)
            .build();

        use tracing_subscriber::prelude::*;
        tracing_subscriber::registry().with(chrome_layer).init();

        Some(ProfilingGuard { _guard: guard })
    }

    #[cfg(not(feature = "profiling"))]
    {
        None
    }
}
