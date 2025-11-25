// myAtomic.hpp - gpu api wrapper 
#pragma once
#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__)
  #define MYA_HD __host__ __device__
  #define MYA_D  __device__
  #define MYA_H  __host__
  #define MYA_INLINE __forceinline__
  #define MYA_DEV (__CUDA_ARCH__)
#else
  #include <atomic>
  #include <thread>
  #define MYA_HD
  #define MYA_D
  #define MYA_H
  #define MYA_INLINE inline
  #define MYA_DEV 0
#endif

namespace myatomic {

// ---------- fences ----------
MYA_INLINE MYA_HD void fence_acq();      // GPU: __threadfence(); CPU: atomic_thread_fence(acquire)
MYA_INLINE MYA_HD void fence_rel();      // GPU: __threadfence(); CPU: atomic_thread_fence(release)
MYA_INLINE MYA_HD void fence_acq_rel();  // GPU: __threadfence(); CPU: atomic_thread_fence(acq_rel)

// ---------- polite spin/backoff ----------
MYA_INLINE MYA_HD void backoff(int spins = 64); // GPU: __nanosleep(); CPU: yield()

// ---------- primary template ----------
template<class T, class Enable = void>
class myAtomic;

// uint32_t
template<>
class myAtomic<std::uint32_t, void> {
public:
  MYA_INLINE MYA_HD myAtomic() : myAtomic(0u) {}
  MYA_INLINE MYA_HD explicit myAtomic(std::uint32_t v);

  MYA_INLINE MYA_HD std::uint32_t load_acquire() const;
  MYA_INLINE MYA_HD void          store_release(std::uint32_t v);
  MYA_INLINE MYA_D  std::uint32_t fetch_add(std::uint32_t x);
  MYA_INLINE MYA_D  std::uint32_t fetch_sub(std::uint32_t x);
  MYA_INLINE MYA_D  bool          compare_exchange(std::uint32_t& expected, std::uint32_t desired);

private:
#if MYA_DEV
  std::uint32_t v_;
#else
  std::atomic<std::uint32_t> a_{0u};
#endif
};

// uint64_t
template<>
class myAtomic<std::uint64_t, void> {
public:
  MYA_INLINE MYA_HD myAtomic() : myAtomic(0ull) {}
  MYA_INLINE MYA_HD explicit myAtomic(std::uint64_t v);

  MYA_INLINE MYA_HD std::uint64_t load_acquire() const;
  MYA_INLINE MYA_HD void          store_release(std::uint64_t v);
  MYA_INLINE MYA_D  std::uint64_t fetch_add(std::uint64_t x);
  MYA_INLINE MYA_D  bool          compare_exchange(std::uint64_t& expected, std::uint64_t desired);

private:
#if MYA_DEV
  std::uint64_t v_;
#else
  std::atomic<std::uint64_t> a_{0ull};
#endif
};

// size_t → alias to uint64_t on your mac/most 64-bit linux
template<>
class myAtomic<size_t, void> : public myAtomic<std::uint64_t, void> {
public:
  using Base = myAtomic<std::uint64_t, void>;
  MYA_INLINE MYA_HD myAtomic() : Base() {}
  MYA_INLINE MYA_HD explicit myAtomic(size_t v) : Base(static_cast<std::uint64_t>(v)) {}
};

// pointers (CAS only; load/store)
template<class P>
class myAtomic<P*, std::enable_if_t<std::is_pointer<P*>::value, void>> {
public:
  MYA_INLINE MYA_HD myAtomic() : myAtomic(nullptr) {}
  MYA_INLINE MYA_HD explicit myAtomic(P* p);

  MYA_INLINE MYA_HD P*   load_acquire() const;
  MYA_INLINE MYA_HD void store_release(P* p);
  MYA_INLINE MYA_D  bool compare_exchange(P*& expected, P* desired);

private:
#if MYA_DEV
  P* p_;
#else
  std::atomic<P*> a_{nullptr};
#endif
};


// ---------- default stubs (CPU impls; GPU TODOs) ----------
#if !MYA_DEV
// fences (CPU)
MYA_INLINE void fence_acq()      { std::atomic_thread_fence(std::memory_order_acquire); }
MYA_INLINE void fence_rel()      { std::atomic_thread_fence(std::memory_order_release); }
MYA_INLINE void fence_acq_rel()  { std::atomic_thread_fence(std::memory_order_acq_rel); }
MYA_INLINE void backoff(int s)   { for (int i=0;i<s;i++) asm volatile("" ::: "memory"); std::this_thread::yield(); }

// uint32
MYA_INLINE myAtomic<std::uint32_t,void>::myAtomic(std::uint32_t v) : a_(v) {}
MYA_INLINE std::uint32_t myAtomic<std::uint32_t,void>::load_acquire() const { return a_.load(std::memory_order_acquire); }
MYA_INLINE void myAtomic<std::uint32_t,void>::store_release(std::uint32_t v){ a_.store(v, std::memory_order_release); }
MYA_INLINE std::uint32_t myAtomic<std::uint32_t,void>::fetch_add(std::uint32_t x){ return a_.fetch_add(x, std::memory_order_acq_rel); }
MYA_INLINE std::uint32_t myAtomic<std::uint32_t,void>::fetch_sub(std::uint32_t x){ return a_.fetch_sub(x, std::memory_order_acq_rel); }
MYA_INLINE bool myAtomic<std::uint32_t,void>::compare_exchange(std::uint32_t& e, std::uint32_t d){
  return a_.compare_exchange_weak(e,d,std::memory_order_acq_rel,std::memory_order_acquire);
}

// uint64
MYA_INLINE myAtomic<std::uint64_t,void>::myAtomic(std::uint64_t v) : a_(v) {}
MYA_INLINE std::uint64_t myAtomic<std::uint64_t,void>::load_acquire() const { return a_.load(std::memory_order_acquire); }
MYA_INLINE void myAtomic<std::uint64_t,void>::store_release(std::uint64_t v){ a_.store(v, std::memory_order_release); }
MYA_INLINE std::uint64_t myAtomic<std::uint64_t,void>::fetch_add(std::uint64_t x){ return a_.fetch_add(x, std::memory_order_acq_rel); }
MYA_INLINE bool myAtomic<std::uint64_t,void>::compare_exchange(std::uint64_t& e, std::uint64_t d){
  return a_.compare_exchange_weak(e,d,std::memory_order_acq_rel,std::memory_order_acquire);
}

// pointers
template<class P>
MYA_INLINE myAtomic<P*,void>::myAtomic(P* p) : a_(p) {}
template<class P>lets
MYA_INLINE P* myAtomic<P*,void>::load_acquire() const { return a_.load(std::memory_order_acquire); }
template<class P>
MYA_INLINE void myAtomic<P*,void>::store_release(P* p){ a_.store(p, std::memory_order_release); }
template<class P>
MYA_INLINE bool myAtomic<P*,void>::compare_exchange(P*& e, P* d){
  return a_.compare_exchange_weak(e,d,std::memory_order_acq_rel,std::memory_order_acquire);
}

#else
// ===== GPU side TODOs =====
// fences (GPU)
MYA_INLINE MYA_HD void fence_acq()      { __threadfence(); }
MYA_INLINE MYA_HD void fence_rel()      { __threadfence(); }
MYA_INLINE MYA_HD void fence_acq_rel()  { __threadfence(); }
MYA_INLINE MYA_HD void backoff(int s)   { for (int i=0;i<s;i++) __nanosleep(32); }

// uint32
MYA_INLINE MYA_HD myAtomic<std::uint32_t,void>::myAtomic(std::uint32_t v) : v_(v) {}
MYA_INLINE MYA_HD std::uint32_t myAtomic<std::uint32_t,void>::load_acquire() const { return v_; }
MYA_INLINE MYA_HD void myAtomic<std::uint32_t,void>::store_release(std::uint32_t v){ v_ = v; }
MYA_INLINE MYA_D  std::uint32_t myAtomic<std::uint32_t,void>::fetch_add(std::uint32_t x){ return atomicAdd(&v_, x); }
MYA_INLINE MYA_D  std::uint32_t myAtomic<std::uint32_t,void>::fetch_sub(std::uint32_t x){ return atomicSub(&v_, x); }
MYA_INLINE MYA_D  bool myAtomic<std::uint32_t,void>::compare_exchange(std::uint32_t& e, std::uint32_t d){
  auto old = atomicCAS(&v_, e, d); bool ok = (old==e); if(!ok) e=old; return ok;
}

// uint64
MYA_INLINE MYA_HD myAtomic<std::uint64_t,void>::myAtomic(std::uint64_t v) : v_(v) {}
MYA_INLINE MYA_HD std::uint64_t myAtomic<std::uint64_t,void>::load_acquire() const { return v_; }
MYA_INLINE MYA_HD void myAtomic<std::uint64_t,void>::store_release(std::uint64_t v){ v_ = v; }
MYA_INLINE MYA_D  std::uint64_t myAtomic<std::uint64_t,void>::fetch_add(std::uint64_t x){ return atomicAdd(&v_, x); }
MYA_INLINE MYA_D  bool myAtomic<std::uint64_t,void>::compare_exchange(std::uint64_t& e, std::uint64_t d){
  auto old = atomicCAS(&v_, e, d); bool ok = (old==e); if(!ok) e=old; return ok;
}

// pointers
template<class P>
MYA_INLINE MYA_HD myAtomic<P*,void>::myAtomic(P* p) : p_(p) {}
template<class P>
MYA_INLINE MYA_HD P* myAtomic<P*,void>::load_acquire() const { return p_; }
template<class P>
MYA_INLINE MYA_HD void myAtomic<P*,void>::store_release(P* p){ p_ = p; }
template<class P>
MYA_INLINE MYA_D  bool myAtomic<P*,void>::compare_exchange(P*& e, P* d){
  auto old = (P*)atomicCAS(reinterpret_cast<unsigned long long*>(&p_),
                           (unsigned long long)e, (unsigned long long)d);
  bool ok = (old==e); if(!ok) e = old; return ok;
}
#endif

} // namespace myatomic
