#include "Lane.h"

#include "ConvLayer.h"
#include "Memory.h"

#include <algorithm>
#include <stdexcept>

namespace {

MacOrderingPolicy effectiveMacOrderingPolicy(const LaneExecutionConfig& config) {
    if (config.broadcast_mode != BroadcastMode::DemandDriven ||
        config.pipeline_mode != PipelineMode::FusedConvEarlyTerminationRelu ||
        !config.enable_early_termination) {
        return MacOrderingPolicy::Importance;
    }
    return config.mac_ordering_policy;
}

}  // namespace

Lane::Lane(int id) : id_(id) {
    if (id_ < 0) {
        throw std::invalid_argument("Lane id must be non-negative.");
    }
}

void Lane::assignTask(Task task,
                      std::uint64_t current_cycle,
                      int group_id,
                      const LaneExecutionConfig& config,
                      MemoryHierarchy& memory,
                      const ConvLayer& layer) {
    if (state_ != LaneState::IDLE) {
        throw std::runtime_error("Attempted to assign task to a non-idle lane.");
    }

    task.initializeContext(
        layer,
        config.execution_mode,
        config.enable_importance_ordering,
        effectiveMacOrderingPolicy(config),
        config.broadcast_mode);
    task.setStatus(TaskStatus::Issued);
    task_reg_ = std::move(task);

    resetEphemeralRegisters();

    activation_ready_cycle_ = current_cycle;
    weight_ready_cycle_ = current_cycle;
    writeback_ready_cycle_ = 0;

    current_op_ = task_reg_->currentOp();
    if (!current_op_) {
        writeback_ready_cycle_ = memory.requestOutputStore(group_id, current_cycle);
        state_ = LaneState::WRITEBACK_OUTPUT;
    } else {
        state_ = LaneState::CHECK_REUSE;
    }
    ++task_issue_events_;
}

std::optional<Task> Lane::step(std::uint64_t current_cycle,
                               int group_id,
                               const WeightBroadcaster& broadcaster,
                               MemoryHierarchy& memory,
                               ConvLayer& layer,
                               const LaneExecutionConfig& config) {
    if (!task_reg_ && state_ != LaneState::IDLE) {
        state_ = LaneState::IDLE;
    }

    switch (state_) {
        case LaneState::IDLE:
            ++idle_cycles_;
            if (completed_tasks_ > 0) {
                ++idle_finished_cycles_;
            }
            return std::nullopt;

        case LaneState::CHECK_REUSE: {
            if (!task_reg_) {
                state_ = LaneState::IDLE;
                ++idle_cycles_;
                return std::nullopt;
            }

            current_op_ = task_reg_->currentOp();
            if (!current_op_) {
                state_ = LaneState::APPLY_RELU;
                ++active_cycles_;
                return std::nullopt;
            }

            if (skipCurrentZeroWorkIfPossible(config)) {
                return std::nullopt;
            }

            activation_ready_cycle_ =
                memory.requestActivation(group_id, current_op_->activation_key, current_cycle);
            weight_ready_cycle_ = memory.requestWeight(group_id, current_op_->weight_key, current_cycle);
            task_reg_->setStatus(TaskStatus::WaitingForData);
            state_ = LaneState::WAIT_FOR_DATA;
            ++active_cycles_;
            return std::nullopt;
        }

        case LaneState::WAIT_FOR_DATA: {
            if (config.broadcast_mode == BroadcastMode::SnapeaFixedSchedule) {
                return stepScheduleDriven(current_cycle, group_id, broadcaster, memory, config);
            }

            if (!task_reg_ || !current_op_) {
                throw std::runtime_error("Lane WAIT_FOR_DATA without task/op context.");
            }

            if (current_cycle < activation_ready_cycle_ || current_cycle < weight_ready_cycle_) {
                ++idle_cycles_;
                ++stall_memory_cycles_;
                return std::nullopt;
            }

            const int demanded_bit = demandedBitIndex(config);
            if (config.execution_mode == ExecutionMode::Int8BitSerial && demanded_bit < 0) {
                task_reg_->setStatus(TaskStatus::Executing);
                state_ = LaneState::FETCH_NEXT_SIGNIFICANT_OP;
                ++active_cycles_;
                return std::nullopt;
            }

            BroadcastDemand demand;
            demand.weight_key = current_op_->weight_key;
            demand.required_bit =
                (config.execution_mode == ExecutionMode::Int8BitParallel) ? -1 : demanded_bit;
            if (!broadcaster.isDemandInPayload(demand)) {
                ++idle_cycles_;
                ++stall_broadcast_cycles_;
                if (broadcaster.currentPayload().valid) {
                    ++idle_broadcast_mismatch_cycles_;
                }
                return std::nullopt;
            }

            task_reg_->setStatus(TaskStatus::Executing);
            state_ = LaneState::FETCH_NEXT_SIGNIFICANT_OP;
            ++active_cycles_;
            return std::nullopt;
        }

        case LaneState::FETCH_NEXT_SIGNIFICANT_OP: {
            if (!task_reg_ || !current_op_) {
                throw std::runtime_error("Lane FETCH_NEXT_SIGNIFICANT_OP without task/op context.");
            }

            if (!current_mac_counted_) {
                task_reg_->markMacStarted();
                current_mac_counted_ = true;
            }

            if (finishCurrentMacIfBitColumnExhausted(config)) {
                state_ = LaneState::CHECK_TERMINATION;
                ++active_cycles_;
                return std::nullopt;
            }

            if (config.execution_mode == ExecutionMode::Int8BitParallel) {
                task_reg_->markParallelMacProcessed(current_op_->full_product);
                ++macs_executed_;
                bit_steps_executed_ += 8;
                ++multiplier_active_cycles_;
                ++accumulator_active_cycles_;

                current_mac_counted_ = false;
                serial_step_in_op_ = 0;
                task_reg_->advanceToNextOp();
                current_op_ = task_reg_->currentOp();

                state_ = LaneState::CHECK_TERMINATION;
                ++active_cycles_;
                return std::nullopt;
            }

            state_ = LaneState::MULTIPLY_OR_SERIAL_STEP;
            ++active_cycles_;
            return std::nullopt;
        }

        case LaneState::MULTIPLY_OR_SERIAL_STEP: {
            if (config.broadcast_mode == BroadcastMode::SnapeaFixedSchedule) {
                return stepScheduleDriven(current_cycle, group_id, broadcaster, memory, config);
            }

            if (!task_reg_ || !current_op_) {
                throw std::runtime_error("Lane MULTIPLY_OR_SERIAL_STEP without task/op context.");
            }

            if (serial_step_in_op_ >= 8) {
                ++macs_executed_;
                task_reg_->advanceToNextOp();
                current_op_ = task_reg_->currentOp();
                current_mac_counted_ = false;
                serial_step_in_op_ = 0;
                processed_bits_in_current_op_ = 0;
                state_ = LaneState::CHECK_TERMINATION;
                ++active_cycles_;
                return std::nullopt;
            }

            if (finishCurrentMacIfBitColumnExhausted(config)) {
                state_ = LaneState::CHECK_TERMINATION;
                ++active_cycles_;
                return std::nullopt;
            }

            const int demanded_bit = demandedBitIndex(config);
            if (demanded_bit < 0) {
                state_ = LaneState::CHECK_TERMINATION;
                ++active_cycles_;
                return std::nullopt;
            }

            const int target_cursor =
                config.enable_msb_first ? (7 - demanded_bit) : demanded_bit;
            if (target_cursor > serial_step_in_op_) {
                const std::size_t skipped_steps =
                    static_cast<std::size_t>(target_cursor - serial_step_in_op_);
                task_reg_->skipBitStepsInCurrentMac(skipped_steps);
                processed_bits_in_current_op_ += skipped_steps;
                serial_step_in_op_ = target_cursor;
            }

            BroadcastDemand demand;
            demand.weight_key = current_op_->weight_key;
            demand.required_bit = demanded_bit;
            if (!broadcaster.isDemandInPayload(demand)) {
                ++idle_cycles_;
                ++stall_broadcast_cycles_;
                if (broadcaster.currentPayload().valid) {
                    ++idle_broadcast_mismatch_cycles_;
                }
                return std::nullopt;
            }

            const int bit_index = demanded_bit;
            const std::int32_t bit_contribution =
                current_op_->bit_contribution[static_cast<std::size_t>(bit_index)];
            task_reg_->markBitProcessed(bit_contribution);
            ++multiplier_active_cycles_;
            ++accumulator_active_cycles_;
            ++bit_steps_executed_;
            ++processed_bits_in_current_op_;
            ++serial_step_in_op_;
            ++active_cycles_;

            if (serial_step_in_op_ >= 8) {
                ++macs_executed_;
                task_reg_->advanceToNextOp();
                current_op_ = task_reg_->currentOp();
                current_mac_counted_ = false;
                serial_step_in_op_ = 0;
                processed_bits_in_current_op_ = 0;
            }

            state_ = LaneState::CHECK_TERMINATION;
            return std::nullopt;
        }

        case LaneState::CHECK_TERMINATION: {
            if (!task_reg_) {
                throw std::runtime_error("Lane CHECK_TERMINATION without task.");
            }

            const bool done = !task_reg_->hasMoreOps();
            const bool terminate_now = shouldEarlyTerminate(*task_reg_, config);
            ++active_cycles_;

            if (terminate_now) {
                task_reg_->setEarlyTerminated(true);
                task_reg_->finalizeSkippedWorkOnEarlyTermination(
                    config.execution_mode, current_mac_counted_, processed_bits_in_current_op_);
                state_ = LaneState::APPLY_RELU;
                return std::nullopt;
            }

            if (done) {
                state_ = LaneState::APPLY_RELU;
                return std::nullopt;
            }

            current_op_ = task_reg_->currentOp();
            if (config.execution_mode == ExecutionMode::Int8BitSerial && serial_step_in_op_ > 0) {
                state_ = LaneState::MULTIPLY_OR_SERIAL_STEP;
            } else {
                state_ = LaneState::CHECK_REUSE;
            }
            return std::nullopt;
        }

        case LaneState::APPLY_RELU: {
            if (!task_reg_) {
                throw std::runtime_error("Lane APPLY_RELU without task.");
            }

            const std::int32_t pre_relu = task_reg_->accumulator();
            const std::int32_t post_relu = std::max<std::int32_t>(0, pre_relu);
            task_reg_->setPreReluOutput(pre_relu);
            task_reg_->setFinalOutput(post_relu);
            task_reg_->setStatus(TaskStatus::Writeback);

            writeback_ready_cycle_ = memory.requestOutputStore(group_id, current_cycle);
            state_ = LaneState::WRITEBACK_OUTPUT;
            ++active_cycles_;
            return std::nullopt;
        }

        case LaneState::WRITEBACK_OUTPUT: {
            if (!task_reg_) {
                throw std::runtime_error("Lane WRITEBACK_OUTPUT without task.");
            }

            if (current_cycle < writeback_ready_cycle_) {
                ++idle_cycles_;
                ++stall_memory_cycles_;
                return std::nullopt;
            }

            layer.writeOutput(task_reg_->outputChannel(),
                              task_reg_->outputY(),
                              task_reg_->outputX(),
                              task_reg_->finalOutput());
            task_reg_->setStatus(TaskStatus::Completed);

            ++active_cycles_;
            ++writeback_cycles_;
            state_ = LaneState::COMPLETE;
            return std::nullopt;
        }

        case LaneState::COMPLETE:
            ++active_cycles_;
            return finalizeCompletedTask();
    }

    throw std::runtime_error("Unknown lane state.");
}

std::optional<BroadcastDemand> Lane::broadcastDemand(std::uint64_t current_cycle,
                                                     const LaneExecutionConfig& config) const {
    if (config.broadcast_mode != BroadcastMode::DemandDriven) {
        return std::nullopt;
    }

    if (!task_reg_ || !current_op_) {
        return std::nullopt;
    }

    if (state_ == LaneState::WAIT_FOR_DATA) {
        if (current_cycle < activation_ready_cycle_ || current_cycle < weight_ready_cycle_) {
            return std::nullopt;
        }

        const int demanded_bit = demandedBitIndex(config);
        if (config.execution_mode == ExecutionMode::Int8BitSerial && demanded_bit < 0) {
            return std::nullopt;
        }

        BroadcastDemand demand;
        demand.weight_key = current_op_->weight_key;
        demand.required_bit =
            (config.execution_mode == ExecutionMode::Int8BitParallel) ? -1 : demanded_bit;
        return demand;
    }

    if (state_ == LaneState::MULTIPLY_OR_SERIAL_STEP &&
        config.execution_mode == ExecutionMode::Int8BitSerial && serial_step_in_op_ < 8) {
        const int demanded_bit = demandedBitIndex(config);
        if (demanded_bit < 0) {
            return std::nullopt;
        }
        BroadcastDemand demand;
        demand.weight_key = current_op_->weight_key;
        demand.required_bit = demanded_bit;
        return demand;
    }

    return std::nullopt;
}

bool Lane::isIdle() const {
    return state_ == LaneState::IDLE;
}

bool Lane::isBusy() const {
    return state_ != LaneState::IDLE;
}

LaneState Lane::state() const {
    return state_;
}

std::uint64_t Lane::activeCycles() const {
    return active_cycles_;
}

std::uint64_t Lane::idleCycles() const {
    return idle_cycles_;
}

std::uint64_t Lane::idleFinishedCycles() const {
    return idle_finished_cycles_;
}

std::uint64_t Lane::stallMemoryCycles() const {
    return stall_memory_cycles_;
}

std::uint64_t Lane::stallBroadcastCycles() const {
    return stall_broadcast_cycles_;
}

std::uint64_t Lane::idleBroadcastMismatchCycles() const {
    return idle_broadcast_mismatch_cycles_;
}

std::uint64_t Lane::multiplierActiveCycles() const {
    return multiplier_active_cycles_;
}

std::uint64_t Lane::accumulatorActiveCycles() const {
    return accumulator_active_cycles_;
}

std::uint64_t Lane::writebackCycles() const {
    return writeback_cycles_;
}

std::uint64_t Lane::completedTasks() const {
    return completed_tasks_;
}

std::uint64_t Lane::taskIssueEvents() const {
    return task_issue_events_;
}

std::uint64_t Lane::macsExecuted() const {
    return macs_executed_;
}

std::uint64_t Lane::earlyTerminatedTasks() const {
    return early_terminated_tasks_;
}

std::uint64_t Lane::bitStepsExecuted() const {
    return bit_steps_executed_;
}

std::uint64_t Lane::skippedMacsTotal() const {
    return skipped_macs_total_;
}

std::uint64_t Lane::skippedMacsEtOnly() const {
    return skipped_macs_et_only_;
}

std::uint64_t Lane::skippedMacsReactiveOnly() const {
    return skipped_macs_reactive_only_;
}

std::uint64_t Lane::skippedMacsProactiveOnly() const {
    return skipped_macs_proactive_only_;
}

std::uint64_t Lane::skippedMacsZeroOnly() const {
    return skipped_macs_zero_only_;
}

std::uint64_t Lane::skippedMacs() const {
    return skippedMacsTotal();
}

std::uint64_t Lane::skippedBitStepsTotal() const {
    return skipped_bit_steps_total_;
}

std::uint64_t Lane::skippedBitStepsEtOnly() const {
    return skipped_bit_steps_et_only_;
}

std::uint64_t Lane::skippedBitStepsBitColumnOnly() const {
    return skipped_bit_steps_bit_column_only_;
}

std::uint64_t Lane::skippedBitSteps() const {
    return skippedBitStepsTotal();
}

std::uint64_t Lane::zeroRunEvents() const {
    return zero_run_events_;
}

std::uint64_t Lane::estimatedCyclesSavedByEarlyTermination() const {
    return estimated_cycles_saved_et_;
}

double Lane::averageProcessedFractionPerTask() const {
    if (completed_tasks_ == 0) {
        return 1.0;
    }
    return cumulative_processed_fraction_ / static_cast<double>(completed_tasks_);
}

std::optional<Task> Lane::stepScheduleDriven(std::uint64_t current_cycle,
                                             int group_id,
                                             const WeightBroadcaster& broadcaster,
                                             MemoryHierarchy& memory,
                                             const LaneExecutionConfig& config) {
    if (!task_reg_ || !current_op_) {
        throw std::runtime_error("Lane schedule-driven step without task/op context.");
    }

    if (state_ == LaneState::WAIT_FOR_DATA) {
        if (current_cycle < activation_ready_cycle_ || current_cycle < weight_ready_cycle_) {
            ++idle_cycles_;
            ++stall_memory_cycles_;
            return std::nullopt;
        }
    }

    if (config.execution_mode == ExecutionMode::Int8BitSerial) {
        const int demanded_bit = demandedBitIndex(config);
        if (demanded_bit < 0) {
            task_reg_->setStatus(TaskStatus::Executing);
            if (!current_mac_counted_) {
                task_reg_->markMacStarted();
                current_mac_counted_ = true;
            }
            if (finishCurrentMacIfBitColumnExhausted(config)) {
                ++active_cycles_;
                return finalizeScheduleDrivenProgress(config);
            }
        }
    }

    BroadcastDemand demand;
    demand.weight_key = current_op_->weight_key;
    demand.required_bit =
        (config.execution_mode == ExecutionMode::Int8BitParallel) ? -1 : demandedBitIndex(config);
    if (!broadcaster.isDemandInPayload(demand)) {
        ++idle_cycles_;
        ++stall_broadcast_cycles_;
        if (broadcaster.currentPayload().valid) {
            ++idle_broadcast_mismatch_cycles_;
        }
        return std::nullopt;
    }

    task_reg_->setStatus(TaskStatus::Executing);
    if (!current_mac_counted_) {
        task_reg_->markMacStarted();
        current_mac_counted_ = true;
    }

    if (config.execution_mode == ExecutionMode::Int8BitParallel) {
        task_reg_->markParallelMacProcessed(current_op_->full_product);
        ++macs_executed_;
        bit_steps_executed_ += 8;
        ++multiplier_active_cycles_;
        ++accumulator_active_cycles_;

        current_mac_counted_ = false;
        serial_step_in_op_ = 0;
        processed_bits_in_current_op_ = 0;
        task_reg_->advanceToNextOp();
        current_op_ = task_reg_->currentOp();
    } else {
        if (finishCurrentMacIfBitColumnExhausted(config)) {
            ++active_cycles_;
            return finalizeScheduleDrivenProgress(config);
        }

        const int demanded_bit = demandedBitIndex(config);
        if (demanded_bit < 0) {
            ++active_cycles_;
            return finalizeScheduleDrivenProgress(config);
        }

        const int target_cursor =
            config.enable_msb_first ? (7 - demanded_bit) : demanded_bit;
        if (target_cursor > serial_step_in_op_) {
            const std::size_t skipped_steps =
                static_cast<std::size_t>(target_cursor - serial_step_in_op_);
            task_reg_->skipBitStepsInCurrentMac(skipped_steps);
            processed_bits_in_current_op_ += skipped_steps;
            serial_step_in_op_ = target_cursor;
        }

        const int bit_index = demanded_bit;
        const std::int32_t bit_contribution =
            current_op_->bit_contribution[static_cast<std::size_t>(bit_index)];
        task_reg_->markBitProcessed(bit_contribution);
        ++multiplier_active_cycles_;
        ++accumulator_active_cycles_;
        ++bit_steps_executed_;
        ++processed_bits_in_current_op_;
        ++serial_step_in_op_;

        if (serial_step_in_op_ >= 8) {
            ++macs_executed_;
            task_reg_->advanceToNextOp();
            current_op_ = task_reg_->currentOp();
            current_mac_counted_ = false;
            serial_step_in_op_ = 0;
            processed_bits_in_current_op_ = 0;
        }
    }

    ++active_cycles_;
    return finalizeScheduleDrivenProgress(config);
}

std::optional<Task> Lane::finalizeScheduleDrivenProgress(const LaneExecutionConfig& config) {
    if (!task_reg_) {
        throw std::runtime_error("Lane schedule-driven finalize without task.");
    }

    const bool done = !task_reg_->hasMoreOps();
    const bool terminate_now = shouldEarlyTerminate(*task_reg_, config);

    if (terminate_now) {
        task_reg_->setEarlyTerminated(true);
        task_reg_->finalizeSkippedWorkOnEarlyTermination(
            config.execution_mode, current_mac_counted_, processed_bits_in_current_op_);
        state_ = LaneState::APPLY_RELU;
        return std::nullopt;
    }

    if (done) {
        state_ = LaneState::APPLY_RELU;
        return std::nullopt;
    }

    current_op_ = task_reg_->currentOp();
    if (config.execution_mode == ExecutionMode::Int8BitSerial && serial_step_in_op_ > 0) {
        state_ = LaneState::MULTIPLY_OR_SERIAL_STEP;
    } else {
        state_ = LaneState::CHECK_REUSE;
    }
    return std::nullopt;
}

int Lane::currentBitIndex(const LaneExecutionConfig& config) const {
    if (config.enable_msb_first) {
        return 7 - serial_step_in_op_;
    }
    return serial_step_in_op_;
}

int Lane::demandedBitIndex(const LaneExecutionConfig& config) const {
    if (config.execution_mode != ExecutionMode::Int8BitSerial) {
        return -1;
    }
    if (!config.enable_bit_column_skip || !task_reg_ || !current_op_) {
        return currentBitIndex(config);
    }
    return task_reg_->nextUsefulBitIndex(serial_step_in_op_, config.enable_msb_first);
}

bool Lane::skipCurrentZeroWorkIfPossible(const LaneExecutionConfig& config) {
    if (config.execution_mode != ExecutionMode::Int8BitSerial || !task_reg_ || !current_op_) {
        return false;
    }
    if (!task_reg_->currentOpIsZeroProduct()) {
        return false;
    }

    task_reg_->setStatus(TaskStatus::Executing);
    if (config.enable_proactive_zero_run_skip &&
        task_reg_->currentOpCanRunProactiveZeroSkip(config.zero_run_order_mode)) {
        task_reg_->skipZeroRun(config.zero_run_order_mode);
    } else if (config.enable_reactive_zero_skip) {
        task_reg_->skipCurrentOpZeroOnly();
    } else {
        return false;
    }

    current_op_ = task_reg_->currentOp();
    state_ = LaneState::CHECK_TERMINATION;
    ++active_cycles_;
    return true;
}

bool Lane::finishCurrentMacIfBitColumnExhausted(const LaneExecutionConfig& config) {
    if (config.execution_mode != ExecutionMode::Int8BitSerial || !config.enable_bit_column_skip ||
        !task_reg_ || !current_op_ || !current_mac_counted_) {
        return false;
    }

    if (task_reg_->nextUsefulBitIndex(serial_step_in_op_, config.enable_msb_first) >= 0) {
        return false;
    }

    if (processed_bits_in_current_op_ < 8U) {
        const std::size_t skipped_steps = 8U - processed_bits_in_current_op_;
        task_reg_->skipBitStepsInCurrentMac(skipped_steps);
        processed_bits_in_current_op_ = 8U;
        serial_step_in_op_ = 8;
    }

    ++macs_executed_;
    task_reg_->advanceToNextOp();
    current_op_ = task_reg_->currentOp();
    current_mac_counted_ = false;
    serial_step_in_op_ = 0;
    processed_bits_in_current_op_ = 0;
    return true;
}

bool Lane::shouldEarlyTerminate(const Task& task, const LaneExecutionConfig& config) const {
    if (config.pipeline_mode != PipelineMode::FusedConvEarlyTerminationRelu ||
        !config.enable_early_termination) {
        return false;
    }

    const std::int64_t remaining = task.remainingPositiveContributionUpperBound();
    const std::int64_t partial = static_cast<std::int64_t>(task.accumulator());

    // Exact safety: final post-ReLU output is guaranteed zero.
    if (partial + remaining <= 0) {
        return true;
    }

    return false;
}

void Lane::resetEphemeralRegisters() {
    current_op_ = nullptr;
    activation_ready_cycle_ = 0;
    weight_ready_cycle_ = 0;
    writeback_ready_cycle_ = 0;
    serial_step_in_op_ = 0;
    current_mac_counted_ = false;
    processed_bits_in_current_op_ = 0;
}

std::optional<Task> Lane::finalizeCompletedTask() {
    if (!task_reg_) {
        throw std::runtime_error("Lane COMPLETE without task.");
    }

    Task finished = std::move(*task_reg_);

    if (finished.earlyTerminated()) {
        ++early_terminated_tasks_;
        estimated_cycles_saved_et_ +=
            static_cast<std::uint64_t>(finished.skippedBitStepsEtOnly() +
                                       finished.skippedMacsEtOnly() * 2U);
    }

    skipped_macs_total_ += static_cast<std::uint64_t>(finished.skippedMacsTotal());
    skipped_macs_et_only_ += static_cast<std::uint64_t>(finished.skippedMacsEtOnly());
    skipped_macs_reactive_only_ += static_cast<std::uint64_t>(finished.skippedMacsReactiveOnly());
    skipped_macs_proactive_only_ += static_cast<std::uint64_t>(finished.skippedMacsProactiveOnly());
    skipped_macs_zero_only_ += static_cast<std::uint64_t>(finished.skippedMacsZeroOnly());
    skipped_bit_steps_total_ += static_cast<std::uint64_t>(finished.skippedBitStepsTotal());
    skipped_bit_steps_et_only_ += static_cast<std::uint64_t>(finished.skippedBitStepsEtOnly());
    skipped_bit_steps_bit_column_only_ +=
        static_cast<std::uint64_t>(finished.skippedBitStepsBitColumnOnly());
    zero_run_events_ += static_cast<std::uint64_t>(finished.zeroRunEvents());

    if (finished.totalMacs() == 0U) {
        cumulative_processed_fraction_ += 1.0;
    } else {
        cumulative_processed_fraction_ +=
            static_cast<double>(finished.processedMacs()) /
            static_cast<double>(finished.totalMacs());
    }

    ++completed_tasks_;
    task_reg_.reset();
    state_ = LaneState::IDLE;
    resetEphemeralRegisters();

    return finished;
}
