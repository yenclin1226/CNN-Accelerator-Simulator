#pragma once

#include "Task.h"

#include <cstddef>
#include <deque>
#include <optional>
#include <vector>

class WorkQueue {
public:
    void push(Task task);
    std::optional<Task> popFront();
    std::optional<Task> popBack();
    std::optional<Task> popFrontMatchingOutputChannel(int output_channel);
    std::optional<Task> popBestLocalMatch(double target_cost,
                                          int preferred_bucket_id,
                                          int preferred_output_family,
                                          int output_family_modulus,
                                          int preferred_phase_hint,
                                          std::optional<int> preferred_output_channel,
                                          const std::vector<double>& bucket_reference_costs);
    std::optional<Task> popClosestPredictedCost(double target_cost,
                                                std::optional<int> preferred_output_channel,
                                                const std::vector<double>& bucket_reference_costs);
    std::optional<Task> popBestMatchWithin(double target_cost,
                                           double max_distance,
                                           int preferred_bucket_id,
                                           int preferred_output_family,
                                           int output_family_modulus,
                                           int preferred_phase_hint,
                                           std::optional<int> preferred_output_channel,
                                           const std::vector<double>& bucket_reference_costs,
                                           std::size_t scan_limit);
    std::optional<Task> popClosestPredictedCostWithin(double target_cost,
                                                      double max_distance,
                                                      std::optional<int> preferred_output_channel,
                                                      const std::vector<double>& bucket_reference_costs);

    double averagePredictedCost() const;
    double averageCostForStealing(const std::vector<double>& bucket_reference_costs) const;
    double predictedCostVariance() const;

    bool empty() const;
    std::size_t size() const;

private:
    std::deque<Task> queue_;
};
