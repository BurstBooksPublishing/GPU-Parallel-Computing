#include <cuda_runtime.h>
#include <vector>
#include <mutex>
#include <thread>
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <atomic>
#include <stdexcept>

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t e = (call);                                                          \
        if (e != cudaSuccess) {                                                          \
            throw std::runtime_error(std::string("CUDA error: ") +                       \
                                     cudaGetErrorString(e) + " @ " + __FILE__ +          \
                                     ":" + std::to_string(__LINE__));                    \
        }                                                                                \
    } while (0)

class CudaSlabPool {
public:
    explicit CudaSlabPool(size_t slab_size = 1ULL << 26)
        : slab_size_(align_up(slab_size, alignment_)) {}

    ~CudaSlabPool() noexcept { release_all_slabs(); }

    void* alloc(size_t bytes) {
        bytes = align_up(bytes, alignment_);
        if (bytes > slab_size_) throw std::bad_alloc();

        auto& cache = thread_cache();
        if (!cache.empty()) {
            void* p = cache.back();
            cache.pop_back();
            return p;
        }

        std::lock_guard<std::mutex> g(mutex_);
        if (free_list_.empty()) allocate_slab();
        void* p = free_list_.back();
        free_list_.pop_back();
        return p;
    }

    void free(void* ptr) noexcept {
        if (!ptr) return;
        auto& cache = thread_cache();
        cache.push_back(ptr);
        if (cache.size() > cache_limit_) {
            std::lock_guard<std::mutex> g(mutex_);
            while (cache.size() > cache_trim_to_) {
                free_list_.push_back(cache.back());
                cache.pop_back();
            }
        }
    }

    size_t capacity() const noexcept {
        std::lock_guard<std::mutex> g(mutex_);
        return slabs_.size() * slab_size_;
    }

private:
    static size_t align_up(size_t v, size_t a) { return ((v + a - 1) / a) * a; }

    std::vector<void*>& thread_cache() {
        thread_local std::vector<void*> cache;
        return cache;
    }

    void allocate_slab() {
        void* slab = nullptr;
        CUDA_CHECK(cudaMalloc(&slab, slab_size_));
        slabs_.push_back(slab);
        uint8_t* base = static_cast<uint8_t*>(slab);
        const size_t blocks = slab_size_ / alignment_;
        free_list_.reserve(free_list_.size() + blocks);
        for (size_t i = 0; i < blocks; ++i) {
            free_list_.push_back(base + i * alignment_);
        }
    }

    void release_all_slabs() noexcept {
        std::lock_guard<std::mutex> g(mutex_);
        for (void* s : slabs_) cudaFree(s);
        slabs_.clear();
        free_list_.clear();
    }

    const size_t slab_size_;
    static constexpr size_t alignment_ = 256;
    static constexpr size_t cache_limit_ = 64;
    static constexpr size_t cache_trim_to_ = 32;

    mutable std::mutex mutex_;
    std::vector<void*> slabs_;
    std::vector<void*> free_list_;
};