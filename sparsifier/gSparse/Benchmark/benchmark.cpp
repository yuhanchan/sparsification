#include <benchmark/benchmark.h>

#include <string>
#include <gSparse/gSparse.hpp>

static void BM_ApproxER_C100(benchmark::State& state) {
  auto graph = gSparse::Builder::buildUnitCompleteGraph(100);
    gSparse::PrecisionRowMatrix er;
    gSparse::ER::ApproximateER approxER;
  for (auto _ : state)
    approxER.CalculateER(er, graph );
}
BENCHMARK(BM_ApproxER_C100);

static void BM_ApproxER_C1000(benchmark::State& state) {
  auto graph = gSparse::Builder::buildUnitCompleteGraph(1000);
    gSparse::PrecisionRowMatrix er;
    gSparse::ER::ApproximateER approxER;
  for (auto _ : state)
    approxER.CalculateER(er, graph );
}
BENCHMARK(BM_ApproxER_C1000);

static void BM_ApproxER_C2000(benchmark::State& state) {
  auto graph = gSparse::Builder::buildUnitCompleteGraph(2000);
    gSparse::PrecisionRowMatrix er;
    gSparse::ER::ApproximateER approxER;
  for (auto _ : state)
    approxER.CalculateER(er, graph );
}
BENCHMARK(BM_ApproxER_C2000);

static void BM_ApproxER_C3000(benchmark::State& state) {
  auto graph = gSparse::Builder::buildUnitCompleteGraph(3000);
    gSparse::PrecisionRowMatrix er;
    gSparse::ER::ApproximateER approxER;
  for (auto _ : state)
    approxER.CalculateER(er, graph );
}
BENCHMARK(BM_ApproxER_C3000);

static void BM_ExactER_C100(benchmark::State& state) {
  auto graph = gSparse::Builder::buildUnitCompleteGraph(100);
    gSparse::ER::ExactER exactER;
    gSparse::PrecisionRowMatrix er;
  for (auto _ : state)
    exactER.CalculateER(er, graph );
}
BENCHMARK(BM_ExactER_C100);

static void BM_StringCreation(benchmark::State& state) {
  for (auto _ : state)
    std::string empty_string;
}
// Register the function as a benchmark
BENCHMARK(BM_StringCreation);

// Define another benchmark
static void BM_StringCopy(benchmark::State& state) {
  std::string x = "hello";
  for (auto _ : state)
    std::string copy(x);
}
BENCHMARK(BM_StringCopy);

BENCHMARK_MAIN();