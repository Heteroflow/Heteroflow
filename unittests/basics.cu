#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <heteroflow/heteroflow.hpp>

// ----------------------------------------------------------------------------
// Parameters
// ----------------------------------------------------------------------------
const size_t C = std::min(16u, std::thread::hardware_concurrency());
const size_t G = std::min(4u, hf::cuda::num_devices());

// ----------------------------------------------------------------------------
// Kernel
// ----------------------------------------------------------------------------
template <typename T>
__global__ void k_set(T* ptr, size_t N, T value) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) {
    ptr[i] = value;
  }
}

template <typename T>
__global__ void k_add(T* ptr, size_t N, T value) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < N) {
    ptr[i] += value;
  }
}

// --------------------------------------------------------
// Testcase: host-tasks
// --------------------------------------------------------
TEST_CASE("host-tasks" * doctest::timeout(300)) {
  
  const size_t num_tasks = 100;

  SUBCASE("Empty") {
    for(size_t W=1; W<=C; ++W) {
      hf::Executor executor(W);
      hf::Heteroflow heteroflow;
      REQUIRE(heteroflow.num_nodes() == 0);
      REQUIRE(heteroflow.empty() == true);
      executor.run(heteroflow).wait();
    }
  }
    
  SUBCASE("Placeholder") {
    for(size_t W=1; W<=C; ++W) {
      hf::Executor executor(W);
      hf::Heteroflow heteroflow;
      std::atomic<int> counter {0};
      std::vector<hf::HostTask> hosts;

      for(size_t i=0; i<num_tasks; ++i) {
        hosts.emplace_back(
          heteroflow.placeholder<hf::HostTask>().name(std::to_string(i))
        );
      }

      for(size_t i=0; i<num_tasks; ++i) {
        REQUIRE(hosts[i].name() == std::to_string(i));
        REQUIRE(hosts[i].num_dependents() == 0);
        REQUIRE(hosts[i].num_successors() == 0);
      }

      for(auto& host : hosts) {
        host.work([&counter](){ counter++; });
      }

      executor.run(heteroflow).get();
      REQUIRE(counter == num_tasks);
    }
  }

  SUBCASE("EmbarrassinglyParallel"){
    for(size_t W=1; W<=C; ++W) {
      hf::Executor executor(W);
      hf::Heteroflow heteroflow;
      std::atomic<int> counter {0};
      std::vector<hf::HostTask> tasks;

      for(size_t i=0;i<num_tasks;i++) {
        tasks.emplace_back(heteroflow.host([&counter]() {counter += 1;}));
      }

      REQUIRE(heteroflow.num_nodes() == num_tasks);
      executor.run(heteroflow).get();
      REQUIRE(counter == num_tasks);
      REQUIRE(heteroflow.num_nodes() == 100);

      counter = 0;
      
      for(size_t i=0;i<num_tasks;i++){
        tasks.emplace_back(heteroflow.host([&counter]() {counter += 1;}));
      }

      REQUIRE(heteroflow.num_nodes() == num_tasks * 2);
      executor.run(heteroflow).get();
      REQUIRE(counter == num_tasks * 2);
      REQUIRE(heteroflow.num_nodes() == 200);
    }
  }

  SUBCASE("ParallelFor") {
    for(size_t W=1; W<=C; ++W) {
      hf::Executor executor(W);

      // Range for
      for(size_t i=0; i<num_tasks; i++) {
        hf::Heteroflow heteroflow;
        std::atomic<int> counter{0};
        auto N = ::rand() % 4098 + 1;
        std::vector<int> vec(N, 20);
        heteroflow.parallel_for(vec.begin(), vec.end(), [&](int i){
          counter += i;
        });
        executor.run(heteroflow).wait();
        auto res = std::accumulate(vec.begin(), vec.end(), 0, std::plus<int>());
        REQUIRE(counter == res);
      }

      // Index for
      for(size_t i=0; i<num_tasks; i++) {
        std::atomic<int> counter{0};
        hf::Heteroflow heteroflow;
        auto N = ::rand() % 4098 + 1;
        auto S = std::min(::rand()%10, N) + 1;
        heteroflow.parallel_for(0, N, S, [&](int){ ++counter; });
        executor.run(heteroflow).wait();
        auto res = 0;
        for(auto i=0; i<N; i+=S) {
          ++res;
        }
        REQUIRE(counter == res);
      }
    }
  }

  SUBCASE("BinarySequence"){
    for(size_t W=1; W<=C; ++W) {
      hf::Executor executor(W);
      hf::Heteroflow heteroflow;
      std::atomic<int> counter {0};
      std::vector<hf::HostTask> tasks;
      for(size_t i=0;i<num_tasks;i++){
        if(i%2 == 0){
          tasks.emplace_back(heteroflow.host(
            [&counter]() { REQUIRE(counter == 0); counter += 1;}
          ));
        }
        else{
          tasks.emplace_back(heteroflow.host(
            [&counter]() { REQUIRE(counter == 1); counter -= 1;}
          ));
        }
        if(i>0){
          tasks[i-1].precede(tasks[i]);
        }

        if(i==0) {
          REQUIRE(tasks[i].num_dependents() == 0);
        }
        else {
          REQUIRE(tasks[i].num_dependents() == 1);
        }
      }
      executor.run(heteroflow).get();
    }
  }

  SUBCASE("LinearCounter"){
    for(size_t W=1; W<=C; ++W) {
      hf::Executor executor(W);
      hf::Heteroflow heteroflow;
      std::atomic<int> counter {0};
      std::vector<hf::HostTask> tasks;
      for(size_t i=0;i<num_tasks;i++){
        tasks.emplace_back(
          heteroflow.host([&counter, i]() { 
            REQUIRE(counter == i); counter += 1;}
          )
        );
        if(i>0){
          tasks[i-1].precede(tasks[i]);
        }
      }
      executor.run(heteroflow).get();
      REQUIRE(counter == num_tasks);
      REQUIRE(heteroflow.num_nodes() == num_tasks);
    }
  }
 
  SUBCASE("Broadcast"){
    for(size_t W=1; W<=C; ++W) {
      hf::Executor executor(W);
      hf::Heteroflow heteroflow;
      std::atomic<int> counter {0};
      std::vector<hf::HostTask> tasks;
      auto src = heteroflow.host([&counter]() {counter -= 1;});
      for(size_t i=1; i<num_tasks; i++){
        auto tgt = heteroflow.host([&counter]() {REQUIRE(counter == -1);});
        src.precede(tgt);
      }
      executor.run(heteroflow).get();
      REQUIRE(counter == - 1);
      REQUIRE(heteroflow.num_nodes() == num_tasks);
    }
  }

  SUBCASE("Succeed"){
    for(size_t W=1; W<=C; ++W) {
      hf::Executor executor(W);
      hf::Heteroflow heteroflow;
      std::atomic<int> counter {0};
      std::vector<hf::HostTask> tasks;
      auto dst = heteroflow.host([&]() { REQUIRE(counter == num_tasks - 1);});
      for(size_t i=1;i<num_tasks;i++){
        auto src = heteroflow.host([&counter]() {counter += 1;});
        dst.succeed(src);
      }
      executor.run(heteroflow).get();
      REQUIRE(counter == num_tasks - 1);
      REQUIRE(heteroflow.num_nodes() == num_tasks);
    }
  }
}

// --------------------------------------------------------
// Testcase: gpu-memset
// --------------------------------------------------------
TEST_CASE("gpu-memset" * doctest::timeout(300)) {

  const size_t num_tasks = 100;

  SUBCASE("kernel") {
    for(size_t c=1; c<=C; ++c) {
      for(size_t g=1; g<=G; ++g) {
        hf::Executor executor(c, g);
        hf::Heteroflow heteroflow;
        for(size_t i=0; i<num_tasks; ++i) {
          auto ndata= ::rand()%4096 + 1;
          auto ptr  = new char[ndata];
          auto pull = heteroflow.pull(nullptr, ndata);
          auto mset = heteroflow.kernel(
            (ndata+255)/256, 256, 0, k_set<char>, pull, ndata, 'z'
          );
          auto push = heteroflow.push(ptr, pull, ndata);
          auto host = heteroflow.host([=](){
            for(auto j=0; j<ndata; j++) {
              REQUIRE(ptr[j] == 'z');
            }
            delete [] ptr;
          });
          pull.precede(mset);
          mset.precede(push);
          push.precede(host);
        }
        executor.run(heteroflow).wait();
      }
    }
  }
  
  SUBCASE("pull-kernel") {
    for(size_t c=1; c<=C; ++c) {
      for(size_t g=1; g<=G; ++g) {
        hf::Executor executor(c, g);
        hf::Heteroflow heteroflow;
        for(size_t i=0; i<num_tasks; ++i) {
          auto ndata= ::rand()%4096 + 1;
          auto ptr  = new char[ndata];
          auto pull = heteroflow.pull(nullptr, ndata, 'a');
          auto mset = heteroflow.kernel(
            (ndata+255)/256, 256, 0, k_add<char>, pull, ndata, 1
          );
          auto push = heteroflow.push(ptr, pull, ndata);
          auto host = heteroflow.host([=](){
            for(auto j=0; j<ndata; j++) {
              REQUIRE(ptr[j] == 'b');
            }
            delete [] ptr;
          });
          pull.precede(mset);
          mset.precede(push);
          push.precede(host);
        }
        executor.run(heteroflow).wait();
      }
    }
  }

  SUBCASE("pull-kernel-push") {
    for(size_t c=1; c<=C; ++c) {
      for(size_t g=1; g<=G; ++g) {
        hf::Executor executor(c, g);
        hf::Heteroflow heteroflow;
        for(size_t i=0; i<num_tasks; ++i) {
          auto ndata= ::rand()%4096 + 1;
          auto ofset= ::rand()%ndata;
          auto ptr  = new char[ndata];
          std::fill_n(ptr, ndata, 'z');
          auto pull = heteroflow.pull(nullptr, ndata, 'a');
          auto mset = heteroflow.kernel(
            (ndata+255)/256, 256, 0, k_add<char>, pull, ndata, 1
          );
          auto push = heteroflow.push(ptr, pull, ofset, ndata-ofset);
          auto host = heteroflow.host([=](){
            for(auto j=0; j<ndata-ofset; j++) {
              REQUIRE(ptr[j] == 'b');
            }
            for(auto j=ndata-ofset; j<ndata; j++) {
              REQUIRE(ptr[j] == 'z');
            }
            delete [] ptr;
          });
          pull.precede(mset);
          mset.precede(push);
          push.precede(host);
        }
        executor.run(heteroflow).wait();
      }
    }
  }
  
  SUBCASE("from-host") {
    for(size_t c=1; c<=C; ++c) {
      for(size_t g=1; g<=G; ++g) {
        hf::Executor executor(c, g);
        hf::Heteroflow heteroflow;
        for(size_t i=0; i<num_tasks; ++i) {
          auto ndata= ::rand()%4096 + 1;
          auto ptr  = new char[ndata];
          std::fill_n(ptr, ndata, 'a');
          auto pull = heteroflow.pull(ptr, ndata);
          auto madd = heteroflow.kernel(
            (ndata+255)/256, 256, 0, k_add<char>, pull, ndata, 1
          );
          auto push = heteroflow.push(ptr, pull, ndata);
          auto host = heteroflow.host([=](){
            for(auto j=0; j<ndata; j++) {
              REQUIRE(ptr[j] == 'b');
            }
            delete [] ptr;
          });
          pull.precede(madd);
          madd.precede(push);
          push.precede(host);
        }
        executor.run(heteroflow).wait();
      }
    }
  }
}

// --------------------------------------------------------
// Testcase: gpu-transfer
// --------------------------------------------------------
TEST_CASE("gpu-transfer" * doctest::timeout(300)) {
  
  SUBCASE("without-offset") {
    for(size_t c=1; c<=C; ++c) {
      for(size_t g=1; g<=G; ++g) {
        hf::Executor executor(c, g);
        hf::Heteroflow heteroflow;
        for(size_t i=0; i<100; ++i) {
          auto ndata = ::rand()%4096 + 1;
          auto data  = new char[ndata];
          auto pull1 = heteroflow.pull(nullptr, ndata, 'a');
          auto pull2 = heteroflow.pull(nullptr, ndata, 'b');
          auto kadd1 = heteroflow.kernel(
            (ndata + 255)/256, 256, 0, k_add<char>, pull1, ndata, 1
          );
          auto kadd2 = heteroflow.kernel(
            (ndata + 255)/256, 256, 0, k_add<char>, pull2, ndata, 1
          );
          auto trans = heteroflow.transfer(
            pull1, 0, pull2, 0, ndata
          );
          auto push1 = heteroflow.push(data, pull1, ndata);
          auto test1 = heteroflow.host([data, ndata](){
            for(int i=0; i<ndata; ++i) {
              REQUIRE(data[i] == 'c');
            }
            delete [] data;
          });
          
          pull1.precede(kadd1);
          pull2.precede(kadd2);
          trans.succeed(kadd1, kadd2)
               .precede(push1);
          push1.precede(test1);
        }
        executor.run(heteroflow).wait();
      }
    }
  }
  
  SUBCASE("with-offset") {
    for(size_t c=1; c<=C; ++c) {
      for(size_t g=1; g<=G; ++g) {
        hf::Executor executor(c, g);
        hf::Heteroflow heteroflow;
        for(size_t i=0; i<1024; ++i) {
          auto ndata = ::rand()%4096 + 1;
          auto offs1 = ::rand()%ndata;
          auto offs2 = ::rand()%ndata;
          auto togo  = std::min(ndata-offs1, ndata-offs2);
          auto data  = new char[ndata];
          auto pull1 = heteroflow.pull(nullptr, ndata, 'a');
          auto pull2 = heteroflow.pull(nullptr, ndata, 'b');
          auto kadd1 = heteroflow.kernel(
            (ndata + 255)/256, 256, 0, k_add<char>, pull1, ndata, 1
          );
          auto kadd2 = heteroflow.kernel(
            (ndata + 255)/256, 256, 0, k_add<char>, pull2, ndata, 1
          );
          auto trans = heteroflow.transfer(
            pull1, offs1, pull2, offs2, togo
          );
          auto push1 = heteroflow.push(data, pull1, ndata);
          auto test1 = heteroflow.host([=](){
            for(int i=0; i<offs1; ++i) {
              REQUIRE(data[i] == 'b');
            }
            for(int i=offs1; i<offs1+togo; ++i) {
              REQUIRE(data[i] == 'c');
            }
            for(int i=offs1+togo; i<ndata; ++i) {
              REQUIRE(data[i] == 'b');
            }
            delete [] data;
          });
          
          pull1.precede(kadd1);
          pull2.precede(kadd2);
          trans.succeed(kadd1, kadd2)
               .precede(push1);
          push1.precede(test1);
        }
        executor.run(heteroflow).wait();
      }
    }
  }
}

// --------------------------------------------------------
// Testcase: state-transition
// --------------------------------------------------------
TEST_CASE("statefulness" * doctest::timeout(300)) {

  SUBCASE("linear-chain") {
    for(size_t c=1; c<=C; ++c) {
      for(size_t g=1; g<=G; ++g) {
        hf::Executor executor(c, g);
        hf::Heteroflow heteroflow;

        std::vector<char> vec;
        size_t size = 0;
        char* data = nullptr;
        dim3 grid, block;

        auto host = heteroflow.host([&](){
          size = 1234567;
          vec.resize(size, 'a');
          data = vec.data();
          grid = (size+255)/256;
          block = 256;
        });
        auto pull = heteroflow.pull(std::ref(data), std::ref(size));
        auto kadd = heteroflow.kernel(
          std::ref(grid), std::ref(block), 0, k_add<char>, pull, std::ref(size), 1
        );
        auto push = heteroflow.push(std::ref(data), pull, std::ref(size));
        auto test = heteroflow.host([&](){
          REQUIRE(size == vec.size());
          REQUIRE(data == vec.data());
          REQUIRE(grid.x == (size+255)/256);
          REQUIRE(block.x == 256);
          for(auto i : vec) {
            REQUIRE(i == 'b');
          }
        });

        host.precede(pull);
        pull.precede(kadd);
        kadd.precede(push);
        push.precede(test);

        executor.run(heteroflow).wait();
      }
    }
  }
}

// --------------------------------------------------------
// Testcase: run_n
// --------------------------------------------------------
TEST_CASE("run_n" * doctest::timeout(300)) {
  
  SUBCASE("linear-chain") {
    for(size_t c=1; c<=C; ++c) {
      for(size_t g=1; g<=G; ++g) {
        hf::Executor executor(c, g);
        hf::Heteroflow heteroflow;
        const size_t ndata = 5000;
        std::vector<char> vec(ndata, 'a');

        auto pull = heteroflow.pull(vec.data(), ndata);
        auto kadd = heteroflow.kernel(
          (ndata + 255)/256, 256, 0, k_add<char>, pull, ndata, 1
        );
        auto push = heteroflow.push(vec.data(), pull, ndata);

        pull.precede(kadd);
        kadd.precede(push);
        
        auto res = 'a';
        for(size_t s=0; s<25; ++s){
          auto r = ::rand() % 5;
          res += r;
          executor.run_n(heteroflow, r).wait();
          for(auto c : vec) {
            REQUIRE(c == res);
          }
        }
      }
    }
  }
}


