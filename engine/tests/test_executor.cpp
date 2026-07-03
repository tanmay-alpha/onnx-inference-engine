// GoogleTest cases for crucible::topological_sort and
// crucible::run_inference (Issue #9: graph executor).
//
// AC test required by the plan:
//   * RunsThreeNodeGraph — MatMul(A, B) → Relu → Softmax, with
//                          A as a 1x2 input and B as a 2x4 weight
//                          initializer of all 0.5. The expected
//                          output has 4 elements that sum to 1.0
//                          (softmax invariant). The exact values
//                          don't matter — what matters is that the
//                          graph executed all three ops in order
//                          and produced the right shape.
//
// We also cover:
//   * The topological_sort helper on its own (DAG, cycle, self-loop).
//   * Missing-input error path.
//   * Unknown-op error path.

#include "crucible/executor.hpp"
#include "crucible/onnx_parser.hpp"
#include "crucible/tensor.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

using crucible::Model;
using crucible::Tensor;
using crucible::GraphNode;
using crucible::Attribute;
using crucible::topological_sort;
using crucible::run_inference;

namespace {

// Helper: construct a float initializer and store it in a Model under
// the given name. We avoid a constructor that takes an `Attribute`-
// bearing name by going through the public `weights` map.
void add_weight(Model& m, const std::string& name,
                const std::vector<int64_t>& shape,
                const std::vector<float>& data) {
    m.weights[name] = Tensor(shape, data);
}

}  // namespace

// -----------------------------------------------------------------------
// AC: end-to-end 3-node graph
// -----------------------------------------------------------------------

TEST(Executor, RunsThreeNodeGraph) {
    // Build a graph: A (input) ──MatMul── M ──Relu── R ──Softmax── Y
    // B is a 2x4 weight of all 0.5.
    //
    //   A = [[1, 2]]      (1x2 row-major)
    //   B = 2x4 of 0.5
    //   M = A @ B        = [[1.5, 1.5, 1.5, 1.5]]    (1x4)
    //   R = ReLU(M)      = same                     (1x4, all positive)
    //   Y = softmax(R)   = [0.25, 0.25, 0.25, 0.25]  (1x4, sums to 1.0)
    Model model;
    model.input_names  = {"A"};
    model.output_names = {"Y"};

    add_weight(model, "B", {2, 4}, std::vector<float>(8, 0.5f));

    GraphNode matmul;
    matmul.op_type = "MatMul";
    matmul.name    = "mm0";
    matmul.inputs  = {"A", "B"};
    matmul.outputs = {"M"};

    GraphNode relu;
    relu.op_type = "Relu";
    relu.name    = "relu0";
    relu.inputs  = {"M"};
    relu.outputs = {"R"};

    GraphNode softmax;
    softmax.op_type = "Softmax";
    softmax.name    = "sm0";
    softmax.inputs  = {"R"};
    softmax.outputs = {"Y"};
    // axis=1 → softmax over the 4-element axis. Default is -1 in ONNX
    // and the operator's `axis` attribute default is 1 (we set it
    // explicitly here to make the test independent of the operator's
    // default choice).
    Attribute axis_attr;
    axis_attr.type = Attribute::Type::Int;
    axis_attr.i    = 1;
    softmax.attributes["axis"] = axis_attr;

    model.graph.node = {matmul, relu, softmax};

    Tensor A({1, 2}, {1.0f, 2.0f});
    std::unordered_map<std::string, Tensor> inputs{{"A", A}};

    Tensor Y = run_inference(model, inputs);

    // Output is 1x4 — 4 elements.
    ASSERT_EQ(Y.size(), 4);
    // Softmax invariant: elements sum to 1.0.
    float sum = 0.0f;
    for (int64_t i = 0; i < Y.size(); ++i) sum += Y.data()[i];
    EXPECT_NEAR(sum, 1.0f, 1e-6f);
    // And each element is positive (softmax of finite inputs).
    for (int64_t i = 0; i < Y.size(); ++i) {
        EXPECT_GT(Y.data()[i], 0.0f) << "i=" << i;
    }
}

// -----------------------------------------------------------------------
// Topological sort — independent of the full executor
// -----------------------------------------------------------------------

TEST(TopologicalSort, AlreadyOrdered) {
    // Three independent nodes that happen to already be in
    // topological order. The algorithm should preserve it.
    GraphNode a, b, c;
    a.op_type = "Relu"; a.name = "a"; a.outputs = {"x"};
    b.op_type = "Relu"; b.name = "b"; b.inputs = {"x"}; b.outputs = {"y"};
    c.op_type = "Relu"; c.name = "c"; c.inputs = {"y"}; c.outputs = {"z"};
    auto sorted = topological_sort({a, b, c});
    ASSERT_EQ(sorted.size(), 3u);
    EXPECT_EQ(sorted[0].name, "a");
    EXPECT_EQ(sorted[1].name, "b");
    EXPECT_EQ(sorted[2].name, "c");
}

TEST(TopologicalSort, ReverseOrder) {
    // Same graph as above, but the input list is reversed. The
    // algorithm should still produce a → b → c.
    GraphNode a, b, c;
    a.op_type = "Relu"; a.name = "a"; a.outputs = {"x"};
    b.op_type = "Relu"; b.name = "b"; b.inputs = {"x"}; b.outputs = {"y"};
    c.op_type = "Relu"; c.name = "c"; c.inputs = {"y"}; c.outputs = {"z"};
    auto sorted = topological_sort({c, b, a});
    ASSERT_EQ(sorted.size(), 3u);
    // Find the index of 'a' (the producer) and verify it precedes
    // both consumers.
    std::size_t ia = 0, ib = 0, ic = 0;
    for (std::size_t i = 0; i < sorted.size(); ++i) {
        if (sorted[i].name == "a") ia = i;
        if (sorted[i].name == "b") ib = i;
        if (sorted[i].name == "c") ic = i;
    }
    EXPECT_LT(ia, ib);
    EXPECT_LT(ib, ic);
}

TEST(TopologicalSort, IndependentNodesAllFirst) {
    // Three independent nodes. Any order is valid; the algorithm
    // should produce all three.
    GraphNode a, b, c;
    a.op_type = "Relu"; a.name = "a";
    b.op_type = "Relu"; b.name = "b";
    c.op_type = "Relu"; c.name = "c";
    auto sorted = topological_sort({a, b, c});
    EXPECT_EQ(sorted.size(), 3u);
}

TEST(TopologicalSort, CycleThrows) {
    // A → B → A. The algorithm must detect the cycle.
    GraphNode a, b;
    a.op_type = "Relu"; a.name = "a"; a.inputs = {"y"}; a.outputs = {"x"};
    b.op_type = "Relu"; b.name = "b"; b.inputs = {"x"}; b.outputs = {"y"};
    EXPECT_THROW(topological_sort({a, b}), std::runtime_error);
}

TEST(TopologicalSort, SelfLoopThrows) {
    // Single node whose input equals its own output. Should throw
    // with a clear self-loop message.
    GraphNode a;
    a.op_type = "Relu"; a.name = "a"; a.inputs = {"x"}; a.outputs = {"x"};
    EXPECT_THROW(topological_sort({a}), std::runtime_error);
}

// -----------------------------------------------------------------------
// run_inference error paths
// -----------------------------------------------------------------------

TEST(Executor, MissingInputThrows) {
    Model model;
    model.input_names  = {"A"};
    model.output_names = {"Y"};
    // No nodes, no initializers. Just an empty graph.
    model.graph.node = {GraphNode{}};  // one placeholder so we can
                                       // even reach the dispatch loop
    // ... actually with no nodes, run_inference returns the value
    // bound to "Y" if it exists, else throws. To exercise the
    // missing-input error we need a graph that names A as an input.
    GraphNode id;
    id.op_type = "Identity";
    id.name    = "id0";
    id.inputs  = {"A"};
    id.outputs = {"Y"};
    model.graph.node = {id};

    std::unordered_map<std::string, Tensor> empty;
    EXPECT_THROW(run_inference(model, empty), std::invalid_argument);
}

TEST(Executor, UnknownOpThrows) {
    // A graph with a single node whose op_type isn't in the dispatch
    // table. The executor should refuse with a clear message.
    Model model;
    model.input_names  = {"A"};
    model.output_names = {"Y"};

    GraphNode mystery;
    mystery.op_type = "MysteryOp";
    mystery.name    = "m0";
    mystery.inputs  = {"A"};
    mystery.outputs = {"Y"};
    model.graph.node = {mystery};

    Tensor A({1}, {1.0f});
    std::unordered_map<std::string, Tensor> inputs{{"A", A}};
    EXPECT_THROW(run_inference(model, inputs), std::runtime_error);
}

TEST(Executor, DanglingInputThrows) {
    // A graph where a node consumes a tensor that's neither an
    // initializer nor a prior output. The executor's `require_tensor`
    // helper should fail with a clear message.
    Model model;
    model.input_names  = {"A"};
    model.output_names = {"Y"};

    GraphNode id;
    id.op_type = "Identity";
    id.name    = "id0";
    id.inputs  = {"ghost"};   // not in tensor_map, not an initializer
    id.outputs = {"Y"};
    model.graph.node = {id};

    Tensor A({1}, {1.0f});
    std::unordered_map<std::string, Tensor> inputs{{"A", A}};
    EXPECT_THROW(run_inference(model, inputs), std::runtime_error);
}

// -----------------------------------------------------------------------
// Issue #10: end-to-end MobileNetV2.
//
// Loads the real mobilenet-v2-7.onnx from the model zoo, feeds a zero
// tensor of shape (1, 3, 224, 224), and asserts that:
//   * run_inference returns a tensor of shape (1, 1000) (the
//     ImageNet-1k classifier head)
//   * it does so without crashing
//
// We don't assert the top-1 class index here — that needs a real
// ImageNet image and is covered by the benchmarks/ scripts in
// Issue #14. The AC for Issue #10 is "shape (1, 1000), no crash".
//
// The test path is `../../models/mobilenet_v2.onnx` relative to the
// test working directory, which is `engine/build/{preset}/`. Both
// `cmake --preset debug` and `cmake --preset release` set up the
// ctest cwd this way.
// -----------------------------------------------------------------------

TEST(Executor, RunsMobileNetV2Shape) {
    // Skip gracefully if the model file isn't present — local
    // developers may run the test suite before the download script.
    // CI downloads the model as a separate job before this test
    // runs, so the skip-path is never taken in CI.
    const std::string model_path = "../../models/mobilenet_v2.onnx";
    std::ifstream probe(model_path);
    if (!probe.good()) {
        GTEST_SKIP() << "mobilenet_v2.onnx not found at " << model_path
                     << "; run `python models/download_models.py` first";
    }
    probe.close();

    Model model = crucible::load_model(model_path);

    // Zero input of the canonical ImageNet shape. We use zeros
    // because we only check the output *shape* here — actual
    // numerical correctness (top-1 vs ONNX Runtime) is a benchmark
    // concern, not a unit-test one.
    Tensor input({1, 3, 224, 224}, 0.0f);

    // The graph's only input is named "data" in the model zoo
    // mobilenetv2 export; pass it through with that key.
    std::unordered_map<std::string, Tensor> inputs{{"data", input}};

    Tensor output = run_inference(model, inputs);

    EXPECT_EQ(output.shape().size(), 2u);
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 1000);
}
