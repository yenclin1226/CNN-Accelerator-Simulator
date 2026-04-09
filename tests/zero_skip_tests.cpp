#include "Accelerator.h"
#include "ConvLayer.h"
#include "ReferenceConv.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

ConvLayer makeSinglePixelLayer(int input_channels, bool use_bias = false) {
    ConvLayerConfig config;
    config.input_height = 1;
    config.input_width = 1;
    config.kernel_size = 1;
    config.stride = 1;
    config.padding = 0;
    config.input_channels = input_channels;
    config.output_channels = 1;
    config.use_bias = use_bias;
    return ConvLayer(config);
}

AcceleratorConfig makeSerialConfig() {
    AcceleratorConfig config;
    config.num_groups = 1;
    config.lanes_per_group = 1;
    config.weight_precision_bits = 8;
    config.execution_mode = ExecutionMode::Int8BitSerial;
    config.grouping_policy = GroupingPolicy::OutputChannelModulo;
    config.broadcast_mode = BroadcastMode::DemandDriven;
    config.pipeline_mode = PipelineMode::BaselineConvRelu;
    config.enable_activation_reuse = false;
    config.enable_weight_reuse = false;
    config.enable_importance_ordering = false;
    config.enable_early_termination = false;
    config.max_cycles = 1'000'000;
    config.memory.dram_latency_cycles = 1;
    config.memory.dram_bandwidth_bytes_per_cycle = 64;
    config.memory.global_buffer_latency_cycles = 1;
    config.memory.global_buffer_bandwidth_bytes_per_cycle = 64;
    config.memory.enable_local_buffer = false;
    return config;
}

SimulationStats runAndExpectExact(ConvLayer layer, const AcceleratorConfig& config) {
    const Tensor3D<std::int32_t> expected = applyRelu(runReferenceConvolution(layer));
    Accelerator accelerator(config);
    SimulationStats stats = accelerator.run(layer);
    const OutputComparison comparison = compareOutputs(expected, layer.outputTensor(), 4);
    expect(comparison.pass, "accelerator output mismatch");
    return stats;
}

}  // namespace

int main() {
    {
        ConvLayer layer = makeSinglePixelLayer(1);
        layer.loadQuantizedInput({0});
        layer.loadQuantizedWeights({5});

        AcceleratorConfig base = makeSerialConfig();
        AcceleratorConfig zero_skip = base;
        zero_skip.enable_reactive_zero_skip = true;

        const SimulationStats base_stats = runAndExpectExact(layer, base);
        const SimulationStats zero_stats = runAndExpectExact(layer, zero_skip);

        expect(base_stats.task_reports.size() == 1U, "expected one task report");
        expect(zero_stats.task_reports.size() == 1U, "expected one task report");
        const TaskReport& report = zero_stats.task_reports.front();
        expect(report.processed_macs == 0U, "reactive zero skip should avoid starting the MAC");
        expect(report.skipped_macs_zero_only == 1U, "reactive zero skip should count one zero MAC");
        expect(report.zero_run_events == 0U, "isolated zero MAC should not count as a run event");
        expect(zero_stats.total_cycles < base_stats.total_cycles,
               "reactive zero skip should reduce total cycles");
    }

    {
        ConvLayer layer = makeSinglePixelLayer(3);
        layer.loadQuantizedInput({0, 0, 5});
        layer.loadQuantizedWeights({2, 3, 4});

        AcceleratorConfig config = makeSerialConfig();
        config.enable_reactive_zero_skip = true;
        config.enable_proactive_zero_run_skip = true;
        config.zero_run_order_mode = ZeroRunOrderMode::ExecutionOrder;

        const SimulationStats stats = runAndExpectExact(layer, config);
        const TaskReport& report = stats.task_reports.front();
        expect(report.processed_macs == 1U, "only the non-zero MAC should execute");
        expect(report.skipped_macs_zero_only == 2U, "two zero MACs should be skipped");
        expect(report.zero_run_events == 1U, "contiguous zero MACs should collapse into one run");
    }

    {
        ConvLayer layer = makeSinglePixelLayer(3);
        layer.loadQuantizedInput({0, 0, 5});
        layer.loadQuantizedWeights({2, 3, 4});

        AcceleratorConfig config = makeSerialConfig();
        config.enable_reactive_zero_skip = true;
        config.enable_proactive_zero_run_skip = true;
        config.zero_run_order_mode = ZeroRunOrderMode::KernelOrder;

        const SimulationStats stats = runAndExpectExact(layer, config);
        const TaskReport& report = stats.task_reports.front();
        expect(report.processed_macs == 1U, "kernel-order run skipping should preserve useful MAC");
        expect(report.skipped_macs_zero_only == 2U, "kernel-order run skipping should skip both zeros");
        expect(report.zero_run_events == 1U, "kernel-order run should emit one event");
    }

    {
        ConvLayer layer = makeSinglePixelLayer(1);
        layer.loadQuantizedInput({3});
        layer.loadQuantizedWeights({8});

        AcceleratorConfig base = makeSerialConfig();
        AcceleratorConfig bit_column = base;
        bit_column.enable_bit_column_skip = true;

        const SimulationStats base_stats = runAndExpectExact(layer, base);
        const SimulationStats bit_stats = runAndExpectExact(layer, bit_column);

        const TaskReport& report = bit_stats.task_reports.front();
        expect(report.processed_macs == 1U, "bit-column skip should still execute the MAC");
        expect(report.processed_bit_steps == 1U, "only the one useful bit should execute");
        expect(report.skipped_bit_steps_bit_column_only == 7U,
               "bit-column skip should jump over seven empty bit positions");
        expect(bit_stats.total_cycles < base_stats.total_cycles,
               "bit-column skip should reduce total cycles");
    }

    {
        ConvLayer layer = makeSinglePixelLayer(4, true);
        layer.loadQuantizedInput({0, 0, 1, 1});
        layer.loadQuantizedWeights({5, 5, -10, 1});
        layer.loadQuantizedBias({0});

        AcceleratorConfig config = makeSerialConfig();
        config.enable_reactive_zero_skip = true;
        config.enable_proactive_zero_run_skip = true;
        config.enable_early_termination = true;
        config.pipeline_mode = PipelineMode::FusedConvEarlyTerminationRelu;

        const SimulationStats stats = runAndExpectExact(layer, config);
        const TaskReport& report = stats.task_reports.front();
        expect(report.skipped_macs_proactive_only == 2U,
               "proactive skip should own the leading zero run");
        expect(report.skipped_macs_reactive_only == 0U,
               "reactive skip should not double-count proactive run MACs");
        expect(report.skipped_macs_et_only == 1U,
               "ET should still skip the final rescue MAC after the negative contribution");
        expect(report.skipped_macs_total == 3U,
               "total skipped MACs should include proactive and ET work");
        expect(report.zero_run_events == 1U,
               "combined zero-skip and ET case should record one proactive run event");
        expect(stats.tasks_terminated_early == 1U, "expected ET to terminate the task");
        expect(stats.macs_skipped_et_only == 1U, "ET-only stats should stay separate");
        expect(stats.macs_skipped_zero_only == 2U, "zero-only stats should stay separate");
    }

    return 0;
}
