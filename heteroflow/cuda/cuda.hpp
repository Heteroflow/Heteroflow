#pragma once

#include "../facility/error.hpp"

namespace hf{ namespace cuda {

// ----------------------------------------------------------------------------
// Device-related 
// ----------------------------------------------------------------------------

/**
@brief queries the number of available devices
*/
inline unsigned num_devices() {
	int N = 0;
  HF_CHECK_CUDA(cudaGetDeviceCount(&N), "failed to get device count");
	return N;
}

/**
@brief gets the current device associated with the caller thread
*/
inline int get_device() {
  int id;
  HF_CHECK_CUDA(cudaGetDevice(&id), "failed to get current device id");
	return id;
}

/**
@brief switches to a given device context
*/
inline void set_device(int id) {
  HF_CHECK_CUDA(cudaSetDevice(id), "failed to switch to device ", id);
}

/** @class ScopedDevice

@brief RAII-style device context switch

*/
class ScopedDevice {

  public:
    
    ScopedDevice(int);
    ~ScopedDevice();

  private:

    int _p;
};

// Constructor
inline ScopedDevice::ScopedDevice(int dev) { 
  HF_CHECK_CUDA(cudaGetDevice(&_p), "failed to get current device scope");
  if(_p == dev) {
    _p = -1;
  }
  else {
    HF_CHECK_CUDA(cudaSetDevice(dev), "failed to scope on device ", dev);
  }
}

// Destructor
inline ScopedDevice::~ScopedDevice() { 
  if(_p != -1) {
    cudaSetDevice(_p);
    //HF_CHECK_CUDA(cudaSetDevice(_p), "failed to scope back to device ", _p);
  }
}

// ----------------------------------------------------------------------------
// Allocator
// ----------------------------------------------------------------------------
class Allocator {

  public:

    void* allocate(size_t N) {
	    void* p = nullptr;
      HF_CHECK_CUDA(cudaMalloc(&p, N), "failed to allocate global memory");
	    return p;
    }

    void deallocate(void* ptr) {
      HF_CHECK_CUDA(cudaFree(ptr), "failed to free ", ptr);
    }
};


}}  // end of namespace hf::cuda ----------------------------------------------

#define HF_WITH_CUDA_CTX(id) \
  for (bool flag = true; flag; ) \
    for (hf::cuda::ScopedDevice __hf__spd__(id); flag; flag = false)

#define HF_WITH_CUDA_CTX_RANGE(r)  \
  for(decltype(r) id=0; id<r; ++id)  \
    for(bool flag = true; flag; )  \
      for(hf::cuda::ScopedDevice __hf__spd__(id); flag; flag = false)


