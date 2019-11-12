#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <heteroflow/heteroflow.hpp>

// --------------------------------------------------------
// Testcase: cpu-tasks
// --------------------------------------------------------
TEST_CASE("cpu-tasks" * doctest::timeout(300)) {
  
  const size_t min_W = 1;
  const size_t max_W = 32;
  const size_t num_tasks = 100;

  SUBCASE("Empty") {
    for(size_t W=min_W; W<=max_W; ++W) {
      hf::Executor executor(W);
      hf::Heteroflow heteroflow;
      REQUIRE(heteroflow.num_nodes() == 0);
      REQUIRE(heteroflow.empty() == true);
      // TODO: bug here!!!
      //executor.run(heteroflow).wait();
    }
  }
    
  SUBCASE("Placeholder") {
    for(size_t W=min_W; W<=max_W; ++W) {
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

  //SUBCASE("EmbarrassinglyParallel"){

  //  for(size_t i=0;i<num_tasks;i++) {
  //    tasks.emplace_back(taskflow.emplace([&counter]() {counter += 1;}));
  //  }

  //  REQUIRE(taskflow.num_nodes() == num_tasks);
  //  executor.run(taskflow).get();
  //  REQUIRE(counter == num_tasks);
  //  REQUIRE(taskflow.num_nodes() == 100);

  //  counter = 0;
  //  
  //  for(size_t i=0;i<num_tasks;i++){
  //    silent_tasks.emplace_back(taskflow.emplace([&counter]() {counter += 1;}));
  //  }

  //  REQUIRE(taskflow.num_nodes() == num_tasks * 2);
  //  executor.run(taskflow).get();
  //  REQUIRE(counter == num_tasks * 2);
  //  REQUIRE(taskflow.num_nodes() == 200);
  //}
  //
  //SUBCASE("BinarySequence"){
  //  for(size_t i=0;i<num_tasks;i++){
  //    if(i%2 == 0){
  //      tasks.emplace_back(
  //        taskflow.emplace([&counter]() { REQUIRE(counter == 0); counter += 1;})
  //      );
  //    }
  //    else{
  //      tasks.emplace_back(
  //        taskflow.emplace([&counter]() { REQUIRE(counter == 1); counter -= 1;})
  //      );
  //    }
  //    if(i>0){
  //      //tasks[i-1].first.precede(tasks[i].first);
  //      tasks[i-1].precede(tasks[i]);
  //    }

  //    if(i==0) {
  //      //REQUIRE(tasks[i].first.num_dependents() == 0);
  //      REQUIRE(tasks[i].num_dependents() == 0);
  //    }
  //    else {
  //      //REQUIRE(tasks[i].first.num_dependents() == 1);
  //      REQUIRE(tasks[i].num_dependents() == 1);
  //    }
  //  }
  //  executor.run(taskflow).get();
  //}

  //SUBCASE("LinearCounter"){
  //  for(size_t i=0;i<num_tasks;i++){
  //    tasks.emplace_back(
  //      taskflow.emplace([&counter, i]() { 
  //        REQUIRE(counter == i); counter += 1;}
  //      )
  //    );
  //    if(i>0){
  //      taskflow.precede(tasks[i-1], tasks[i]);
  //    }
  //  }
  //  executor.run(taskflow).get();
  //  REQUIRE(counter == num_tasks);
  //  REQUIRE(taskflow.num_nodes() == num_tasks);
  //}
 
  //SUBCASE("Broadcast"){
  //  auto src = taskflow.emplace([&counter]() {counter -= 1;});
  //  for(size_t i=1; i<num_tasks; i++){
  //    silent_tasks.emplace_back(
  //      taskflow.emplace([&counter]() {REQUIRE(counter == -1);})
  //    );
  //  }
  //  taskflow.broadcast(src, silent_tasks);
  //  executor.run(taskflow).get();
  //  REQUIRE(counter == - 1);
  //  REQUIRE(taskflow.num_nodes() == num_tasks);
  //}

  //SUBCASE("Succeed"){
  //  auto dst = taskflow.emplace([&]() { REQUIRE(counter == num_tasks - 1);});
  //  for(size_t i=1;i<num_tasks;i++){
  //    silent_tasks.emplace_back(
  //      taskflow.emplace([&counter]() {counter += 1;})
  //    );
  //  }
  //  dst.succeed(silent_tasks);
  //  executor.run(taskflow).get();
  //  REQUIRE(counter == num_tasks - 1);
  //  REQUIRE(taskflow.num_nodes() == num_tasks);
  //}

  //SUBCASE("MapReduce"){
  //  auto src = taskflow.emplace([&counter]() {counter = 0;});
  //  for(size_t i=0;i<num_tasks;i++){
  //    silent_tasks.emplace_back(
  //      taskflow.emplace([&counter]() {counter += 1;})
  //    );
  //  }
  //  taskflow.broadcast(src, silent_tasks);
  //  auto dst = taskflow.emplace(
  //    [&counter, num_tasks]() { REQUIRE(counter == num_tasks);}
  //  );
  //  taskflow.gather(silent_tasks, dst);
  //  executor.run(taskflow).get();
  //  REQUIRE(taskflow.num_nodes() == num_tasks + 2);
  //}

  //SUBCASE("Linearize"){
  //  for(size_t i=0;i<num_tasks;i++){
  //    silent_tasks.emplace_back(
  //      taskflow.emplace([&counter, i]() { 
  //        REQUIRE(counter == i); counter += 1;}
  //      )
  //    );
  //  }
  //  taskflow.linearize(silent_tasks);
  //  executor.run(taskflow).get();
  //  REQUIRE(counter == num_tasks);
  //  REQUIRE(taskflow.num_nodes() == num_tasks);
  //}

  //SUBCASE("Kite"){
  //  auto src = taskflow.emplace([&counter]() {counter = 0;});
  //  for(size_t i=0;i<num_tasks;i++){
  //    silent_tasks.emplace_back(
  //      taskflow.emplace([&counter, i]() { 
  //        REQUIRE(counter == i); counter += 1; }
  //      )
  //    );
  //  }
  //  taskflow.broadcast(src, silent_tasks);
  //  taskflow.linearize(silent_tasks);
  //  auto dst = taskflow.emplace(
  //    [&counter, num_tasks]() { REQUIRE(counter == num_tasks);}
  //  );
  //  taskflow.gather(silent_tasks, dst);
  //  executor.run(taskflow).get();
  //  REQUIRE(taskflow.num_nodes() == num_tasks + 2);
  //}
}
