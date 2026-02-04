//! KV Cache for efficient autoregressive generation.
//!
//! Supports both concatenation-based and pre-allocated caching strategies.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// KV Cache that concatenates new keys/values to existing cache.
///
/// Simple strategy that works well for short sequences but may be
/// inefficient for very long sequences due to repeated concatenation.
#[derive(Debug, Clone)]
pub struct KVCache<B: Backend> {
    /// Cached key tensor [batch, heads, seq, head_dim]
    pub k: Option<Tensor<B, 4>>,
    /// Cached value tensor [batch, heads, seq, head_dim]
    pub v: Option<Tensor<B, 4>>,
}

impl<B: Backend> Default for KVCache<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> KVCache<B> {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    /// Update the cache with new key tensor.
    ///
    /// # Arguments
    /// * `k` - New key tensor [batch, heads, seq, head_dim]
    ///
    /// # Returns
    /// Full key tensor including cache [batch, heads, total_seq, head_dim]
    pub fn update_k(&mut self, k: Tensor<B, 4>) -> Tensor<B, 4> {
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

    /// Update the cache with new value tensor.
    ///
    /// # Arguments
    /// * `v` - New value tensor [batch, heads, seq, head_dim]
    ///
    /// # Returns
    /// Full value tensor including cache [batch, heads, total_seq, head_dim]
    pub fn update_v(&mut self, v: Tensor<B, 4>) -> Tensor<B, 4> {
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

    /// Update both K and V caches.
    pub fn update(&mut self, k: Tensor<B, 4>, v: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let k_full = self.update_k(k);
        let v_full = self.update_v(v);
        (k_full, v_full)
    }

    /// Get the current sequence length in the cache.
    pub fn seq_len(&self) -> usize {
        match &self.k {
            Some(k) => k.dims()[2],
            None => 0,
        }
    }

    /// Reset the cache.
    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }

    /// Apply sliding window eviction.
    ///
    /// If cache exceeds window size, evict oldest entries.
    pub fn apply_sliding_window(&mut self, window_size: usize) {
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
    /// Create caches for n layers.
    pub fn new(n_layers: usize) -> Self {
        Self {
            caches: (0..n_layers).map(|_| KVCache::new()).collect(),
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
