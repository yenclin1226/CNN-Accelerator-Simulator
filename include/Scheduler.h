#pragma once

#include "Task.h"
#include "WorkQueue.h"

#include <cstdint>
#include <optional>
#include <vector>

class Scheduler {
public:
    explicit Scheduler(int num_groups);

    void clear();
    void addTask(int group_id, Task task);
    void setCostAwareStealingEnabled(bool enabled);
    void setGroupTargetCosts(std::vector<double> target_costs);
    void setBucketReferenceCosts(std::vector<double> reference_costs);
    void setGroupPhaseHints(std::vector<int> phase_hints);
    void recordTaskCompletion(int group_id, const Task& task, double observed_cost_proxy);

    std::optional<Task> fetchTask(int requester_group_id, std::optional<int> preferred_output_channel);

    bool hasPendingTasks() const;
    std::uint64_t workStealingEvents() const;

    const WorkQueue& groupQueue(int group_id) const;

private:
    struct RuntimeFeedback {
        double ema_observed_cost{0.0};
        double ema_processed_fraction{1.0};
        double ema_et_rate{0.0};
        std::uint64_t samples{0};
    };

    int findDonorGroup(int requester_group_id) const;
    std::vector<int> rankCostAwareDonorGroups(int requester_group_id,
                                              double requester_target_cost) const;
    int preferredBucketForTargetCost(double target_cost) const;
    int groupPhaseHint(int group_id) const;
    void refreshDynamicTargets();
    double queueMeanCostForStealing(const WorkQueue& queue) const;
    double requesterTargetCost(int requester_group_id) const;
    void validateGroupId(int group_id) const;

    std::vector<WorkQueue> queues_;
    std::uint64_t work_stealing_events_{0};
    bool cost_aware_stealing_enabled_{false};
    std::vector<double> group_target_costs_;
    std::vector<double> bucket_reference_costs_;
    std::vector<int> group_phase_hints_;
    std::vector<RuntimeFeedback> group_feedback_;
    std::vector<RuntimeFeedback> bucket_feedback_;
    std::uint64_t completed_feedback_samples_{0};
};
