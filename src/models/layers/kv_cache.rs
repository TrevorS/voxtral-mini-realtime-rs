//! KV Cache for efficient autoregressive generation.
//!
//! Supports both concatenation-based and pre-allocated caching strategies.
//! Use `KVCache::new()` for dynamic (cat-based) caching and
//! `KVCache::preallocated()` for pre-allocated buffers that avoid
//! per-step GPU allocations.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// KV Cache supporting dynamic concatenation or pre-allocated buffers.
///
/// **Dynamic mode** (`KVCache::new()`): Concatenates new keys/values onto
/// the existing cache each step. Simple but allocates growing GPU buffers.
///
/// **Pre-allocated mode** (`KVCache::preallocated()`): Writes into a
/// fixed-size buffer via `slice_assign`, avoiding per-step allocations.
/// Returns narrow slices for the filled region.
#[derive(Debug, Clone)]
pub struct KVCache<B: Backend> {
    /// Cached key tensor [batch, heads, seq_or_capacity, head_dim]
    pub k: Option<Tensor<B, 4>>,
    /// Cached value tensor [batch, heads, seq_or_capacity, head_dim]
    pub v: Option<Tensor<B, 4>>,
    /// Current filled length (only used in pre-allocated mode).
    len: usize,
    /// Pre-allocated capacity. 0 = dynamic (cat) mode.
    capacity: usize,
}

impl<B: Backend> Default for KVCache<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> KVCache<B> {
    /// Create an empty dynamic cache (concatenation-based).
    pub fn new() -> Self {
        Self {
            k: None,
            v: None,
            len: 0,
            capacity: 0,
        }
    }

    /// Create a pre-allocated cache with zero-filled buffers.
    ///
    /// Avoids GPU memory allocation on each step by writing into
    /// a fixed buffer via `slice_assign`.
    pub fn preallocated(
        batch: usize,
        heads: usize,
        max_seq: usize,
        head_dim: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            k: Some(Tensor::zeros([batch, heads, max_seq, head_dim], device)),
            v: Some(Tensor::zeros([batch, heads, max_seq, head_dim], device)),
            len: 0,
            capacity: max_seq,
        }
    }

    /// Update the cache with new key tensor.
    pub fn update_k(&mut self, k: Tensor<B, 4>) -> Tensor<B, 4> {
        if self.capacity > 0 {
            let new_seq = k.dims()[2];
            let pos = self.len;
            let buf = self.k.take().unwrap();
            let [b, h, _, hd] = buf.dims();
            let buf = buf.slice_assign([0..b, 0..h, pos..pos + new_seq, 0..hd], k);
            // len is advanced here; if using update(), it gets advanced twice
            // (once for k, once for v) — but update() handles len itself.
            let new_len = pos + new_seq;
            let view = buf.clone().slice([0..b, 0..h, 0..new_len, 0..hd]);
            self.k = Some(buf);
            view
        } else {
            match &self.k {
                None => {
                    self.k = Some(k.clone());
                    k
                }
                Some(cache) => {
                    let full = Tensor::cat(vec![cache.clone(), k], 2);
                    self.k = Some(full.clone());
                    full
                }
            }
        }
    }

    /// Update the cache with new value tensor.
    pub fn update_v(&mut self, v: Tensor<B, 4>) -> Tensor<B, 4> {
        if self.capacity > 0 {
            let new_seq = v.dims()[2];
            let pos = self.len;
            let buf = self.v.take().unwrap();
            let [b, h, _, hd] = buf.dims();
            let buf = buf.slice_assign([0..b, 0..h, pos..pos + new_seq, 0..hd], v);
            let new_len = pos + new_seq;
            let view = buf.clone().slice([0..b, 0..h, 0..new_len, 0..hd]);
            self.v = Some(buf);
            view
        } else {
            match &self.v {
                None => {
                    self.v = Some(v.clone());
                    v
                }
                Some(cache) => {
                    let full = Tensor::cat(vec![cache.clone(), v], 2);
                    self.v = Some(full.clone());
                    full
                }
            }
        }
    }

    /// Update both K and V caches.
    pub fn update(&mut self, k: Tensor<B, 4>, v: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        if self.capacity > 0 {
            let new_seq = k.dims()[2];
            let pos = self.len;
            let k_buf = self.k.take().unwrap();
            let v_buf = self.v.take().unwrap();
            let [b, h, _, hd] = k_buf.dims();

            let k_buf = k_buf.slice_assign([0..b, 0..h, pos..pos + new_seq, 0..hd], k);
            let v_buf = v_buf.slice_assign([0..b, 0..h, pos..pos + new_seq, 0..hd], v);

            self.len = pos + new_seq;
            let new_len = self.len;

            let k_view = k_buf.clone().slice([0..b, 0..h, 0..new_len, 0..hd]);
            let v_view = v_buf.clone().slice([0..b, 0..h, 0..new_len, 0..hd]);

            self.k = Some(k_buf);
            self.v = Some(v_buf);

            (k_view, v_view)
        } else {
            let k_full = self.update_k(k);
            let v_full = self.update_v(v);
            (k_full, v_full)
        }
    }

    /// Get the current sequence length in the cache.
    pub fn seq_len(&self) -> usize {
        if self.capacity > 0 {
            self.len
        } else {
            self.k.as_ref().map(|k| k.dims()[2]).unwrap_or(0)
        }
    }

    /// Reset the cache.
    pub fn reset(&mut self) {
        if self.capacity > 0 {
            // Zero out buffers and reset position.
            if let Some(k) = &self.k {
                let dims = k.dims();
                let device = k.device();
                self.k = Some(Tensor::zeros(dims, &device));
                self.v = Some(Tensor::zeros(dims, &device));
            }
            self.len = 0;
        } else {
            self.k = None;
            self.v = None;
            self.len = 0;
        }
    }

    /// Apply sliding window eviction.
    ///
    /// If cache exceeds window size, evict oldest entries.
    /// Only supported in dynamic mode; pre-allocated caches use
    /// attention masking for sliding window instead.
    pub fn apply_sliding_window(&mut self, window_size: usize) {
        assert_eq!(
            self.capacity, 0,
            "apply_sliding_window not supported in pre-allocated mode"
        );
        if let Some(k) = &self.k {
            let seq_len = k.dims()[2];
            if seq_len > window_size {
                let start = seq_len - window_size;
                let [batch, heads, _, head_dim] = k.dims();
                self.k = Some(
                    k.clone()
                        .slice([0..batch, 0..heads, start..seq_len, 0..head_dim]),
                );
            }
        }
        if let Some(v) = &self.v {
            let seq_len = v.dims()[2];
            if seq_len > window_size {
                let start = seq_len - window_size;
                let [batch, heads, _, head_dim] = v.dims();
                self.v = Some(
                    v.clone()
                        .slice([0..batch, 0..heads, start..seq_len, 0..head_dim]),
                );
            }
        }
    }
}

/// Collection of KV caches for all layers.
#[derive(Debug)]
pub struct LayerCaches<B: Backend> {
    caches: Vec<KVCache<B>>,
}

impl<B: Backend> LayerCaches<B> {
    /// Create dynamic (cat-based) caches for n layers.
    pub fn new(n_layers: usize) -> Self {
        Self {
            caches: (0..n_layers).map(|_| KVCache::new()).collect(),
        }
    }

    /// Create pre-allocated caches for n layers.
    pub fn new_preallocated(
        n_layers: usize,
        batch: usize,
        n_kv_heads: usize,
        max_seq: usize,
        head_dim: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            caches: (0..n_layers)
                .map(|_| KVCache::preallocated(batch, n_kv_heads, max_seq, head_dim, device))
                .collect(),
        }
    }

    /// Get mutable reference to a layer's cache.
    pub fn get_mut(&mut self, layer: usize) -> Option<&mut KVCache<B>> {
        self.caches.get_mut(layer)
    }

    /// Get the current sequence length (same for all layers).
    pub fn seq_len(&self) -> usize {
        self.caches.first().map(|c| c.seq_len()).unwrap_or(0)
    }

    /// Reset all caches.
    pub fn reset(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
    }

    /// Apply sliding window eviction to all caches.
    pub fn apply_sliding_window(&mut self, window_size: usize) {
        for cache in &mut self.caches {
            cache.apply_sliding_window(window_size);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;

    type TestBackend = Wgpu;

    #[test]
    fn test_kv_cache_empty() {
        let cache: KVCache<TestBackend> = KVCache::new();
        assert!(cache.k.is_none());
        assert!(cache.v.is_none());
        assert_eq!(cache.seq_len(), 0);
    }

    #[test]
    fn test_kv_cache_update() {
        let device = Default::default();
        let mut cache: KVCache<TestBackend> = KVCache::new();

        // First update
        let k1 = Tensor::<TestBackend, 4>::zeros([1, 4, 5, 16], &device);
        let k_out = cache.update_k(k1);
        assert_eq!(k_out.dims(), [1, 4, 5, 16]);
        assert_eq!(cache.seq_len(), 5);

        // Second update
        let k2 = Tensor::<TestBackend, 4>::zeros([1, 4, 3, 16], &device);
        let k_out = cache.update_k(k2);
        assert_eq!(k_out.dims(), [1, 4, 8, 16]);
        assert_eq!(cache.seq_len(), 8);
    }

    #[test]
    fn test_kv_cache_sliding_window() {
        let device = Default::default();
        let mut cache: KVCache<TestBackend> = KVCache::new();

        // Add 10 tokens
        let k = Tensor::<TestBackend, 4>::zeros([1, 4, 10, 16], &device);
        cache.update_k(k);
        assert_eq!(cache.seq_len(), 10);

        // Apply sliding window of 5
        cache.apply_sliding_window(5);
        assert_eq!(cache.seq_len(), 5);
    }

    #[test]
    fn test_layer_caches() {
        let device = Default::default();
        let mut caches: LayerCaches<TestBackend> = LayerCaches::new(4);

        // Update first layer
        if let Some(cache) = caches.get_mut(0) {
            let k = Tensor::<TestBackend, 4>::zeros([1, 4, 5, 16], &device);
            let v = Tensor::<TestBackend, 4>::zeros([1, 4, 5, 16], &device);
            cache.update(k, v);
        }

        // First layer should have entries
        assert_eq!(caches.caches[0].seq_len(), 5);
        // Other layers should be empty
        assert_eq!(caches.caches[1].seq_len(), 0);
    }
}
