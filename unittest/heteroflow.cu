#include <heteroflow/heteroflow.hpp>
#include <vector>
#include <cassert>

// Compilation: nvcc -O2 -g ./unittest/heteroflow.cu -std=c++14 -I .

__global__ void assign_value(int n, float a, float *x) {
  // Get the corresponding idx
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if (i < n) {
    x[i] = a;
  }
}

__global__ void add(int n, float *x, float *y) {
  // Get the corresponding idx
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if (i < n) {
    y[i] += x[i];
  }
}

void gpu_only() {
  const int N = 1 << 12;

  int num_devices {-1};
  cudaGetDeviceCount(&num_devices);
  hf::Executor executor(1, num_devices);
  hf::Heteroflow hf("unittest");

  const int num_graphs = 1000;

  std::vector<std::vector<float>> xs (num_graphs);
  std::vector<std::vector<float>> ys (num_graphs);
  std::vector<std::vector<float>> zs (num_graphs);

  for(int i=0; i<num_graphs; i++) {
    auto &x = xs[i];
    auto &y = ys[i];
    auto &z = zs[i];

    x.resize(N, 1.0f);
    y.resize(N, 2.0f);
    z.resize(N, 3.0f);

    auto pull_x = hf.pull(x);
    auto pull_y = hf.pull(y);
    auto pull_z = hf.pull(z);

    auto kx = hf.kernel(assign_value, N, 1.0f, pull_x)
                .grid_x((N+255)/256)
                .block_x(256);
    auto ky = hf.kernel(assign_value, N, 2.0f, pull_y)
                .grid_x((N+255)/256)
                .block_x(256);

    pull_x.precede(kx);
    pull_y.precede(ky);

    auto kxy = hf.kernel(add, N, pull_x, pull_y)
                 .grid_x((N+255)/256)
                 .block_x(256).name("KXY")
                 .succeed(kx, ky);

    auto kzy = hf.kernel(add, N, pull_y, pull_z)
                 .grid_x((N+255)/256)
                 .block_x(256).name("KZY")
                 .succeed(pull_z, kxy);
    
    auto push_x = hf.push(pull_x, x).succeed(kzy);
    auto push_y = hf.push(pull_y, y).succeed(kzy);
    auto push_z = hf.push(pull_z, z).succeed(kzy);
  }

  //std::cout << hf.dump() << "\n"; exit(0);

  executor.run(hf).wait();
  cudaDeviceSynchronize();

  for(int j=0; j<num_graphs; j++) {
    for (int i = 0; i < N; i++) {
      assert(zs[j][i]-6.0f == 0.0f);
    }
  }

  std::cout << "Done\n";
}



void mix() {
  const int N = 1 << 12;

  int num_devices {-1};
  cudaGetDeviceCount(&num_devices);
  hf::Executor executor(40, num_devices);
  hf::Heteroflow hf("unittest");

  const int num_graphs = 1000;

  std::vector<std::vector<float>> xs (num_graphs);
  std::vector<std::vector<float>> ys (num_graphs);
  std::vector<std::vector<float>> zs (num_graphs);

  std::vector<hf::PullTask> pull_tasks;
  std::vector<hf::PushTask> push_tasks;

  for(int i=0; i<num_graphs; i++) {
    auto &x = xs[i];
    auto &y = ys[i];
    auto &z = zs[i];

    x.resize(N, 1.0f);
    y.resize(N, 2.0f);
    z.resize(N, 3.0f);

    auto pull_x = hf.pull(x);
    auto pull_y = hf.pull(y);
    auto pull_z = hf.pull(z);

    pull_tasks.emplace_back(pull_x);
    pull_tasks.emplace_back(pull_y);
    pull_tasks.emplace_back(pull_z);

    auto kx = hf.kernel(assign_value, N, 1.0f, pull_x)
                .grid_x((N+255)/256)
                .block_x(256);
    auto ky = hf.kernel(assign_value, N, 2.0f, pull_y)
                .grid_x((N+255)/256)
                .block_x(256);

    pull_x.precede(kx);
    pull_y.precede(ky);

    auto kxy = hf.kernel(add, N, pull_x, pull_y)
                 .grid_x((N+255)/256)
                 .block_x(256).name("KXY")
                 .succeed(kx, ky);

    auto kzy = hf.kernel(add, N, pull_y, pull_z)
                 .grid_x((N+255)/256)
                 .block_x(256).name("KZY")
                 .succeed(pull_z, kxy);
    
    auto push_x = hf.push(pull_x, x).succeed(kzy);
    auto push_y = hf.push(pull_y, y).succeed(kzy);
    auto push_z = hf.push(pull_z, z).succeed(kzy);

    push_tasks.emplace_back(push_x);
    push_tasks.emplace_back(push_y);
    push_tasks.emplace_back(push_z);
  }

  //std::cout << hf.dump() << "\n"; exit(0);
  
  // Create CPU tasks
  const int num_cpu_tasks = pull_tasks.size() << 2;
  std::atomic<int> cpu_counter {0};
  std::vector<hf::HostTask> host_tasks;
  for(int i=0 ; i<num_cpu_tasks; i++) {
    auto host_task = hf.host([&](){ cpu_counter++; });
    host_task.precede(pull_tasks[i%pull_tasks.size()]);
  }

  executor.run(hf).wait();
  cudaDeviceSynchronize();

  for(int j=0; j<num_graphs; j++) {
    for (int i = 0; i < N; i++) {
      assert(zs[j][i]-6.0f == 0.0f);
    }
  }

  assert(cpu_counter.load() == num_cpu_tasks);

  std::cout << "Done\n";
}

int main(int argc, char* argv[]) {
  mix();
}

