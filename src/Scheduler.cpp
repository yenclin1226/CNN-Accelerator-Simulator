#include "Scheduler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

constexpr double kFeedbackAlpha = 0.10;
constexpr std::uint64_t kFeedbackRefreshInterval = 16;
constexpr double kImmediateTargetBlend = 0.06;
constexpr double kImmediateBucketBlend = 0.04;
constexpr double kRefreshOldWeight = 0.80;
constexpr double kRefreshObservedWeight = 0.20;
constexpr double kRequesterTargetWeight = 0.55;
constexpr double kRequesterObservedWeight = 0.30;
constexpr double kRequesterQueueWeight = 0.15;
constexpr double kMinimumStealWindow = 8.0;
constexpr double kStrictStealWindowRatio = 0.15;
constexpr double kRelaxedStealWindowRatio = 0.30;

}  // namespace

Scheduler::Scheduler(int num_groups)
    : queues_(static_cast<std::size_t>(num_groups)),
      group_target_costs_(static_cast<std::size_t>(num_groups), 0.0),
      group_feedback_(static_cast<std::size_t>(num_groups)) {
    if (num_groups <= 0) {
        throw std::invalid_argument("Scheduler requires at least one group.");
    }
}

void Scheduler::clear() {
    for (WorkQueue& queue : queues_) {
        queue = WorkQueue{};
    }
    std::fill(group_target_costs_.begin(), group_target_costs_.end(), 0.0);
    work_stealing_events_ = 0;
    bucket_reference_costs_.clear();
    group_phase_hints_.clear();
    group_feedback_.assign(queues_.size(), RuntimeFeedback{});
    bucket_feedback_.clear();
    completed_feedback_samples_ = 0;
}

void Scheduler::addTask(int group_id, Task task) {
    validateGroupId(group_id);
    queues_[static_cast<std::size_t>(group_id)].push(std::move(task));
}

void Scheduler::setCostAwareStealingEnabled(bool enabled) {
    cost_aware_stealing_enabled_ = enabled;
}

void Scheduler::setGroupTargetCosts(std::vector<double> target_costs) {
    if (target_costs.size() != queues_.size()) {
        throw std::invalid_argument("Scheduler target-cost vector size mismatch.");
    }
    group_target_costs_ = std::move(target_costs);
}

void Scheduler::setBucketReferenceCosts(std::vector<double> reference_costs) {
    bucket_reference_costs_ = std::move(reference_costs);
    bucket_feedback_.assign(bucket_reference_costs_.size(), RuntimeFeedback{});
}

void Scheduler::setGroupPhaseHints(std::vector<int> phase_hints) {
    if (!phase_hints.empty() && phase_hints.size() != queues_.size()) {
        throw std::invalid_argument("Scheduler group-phase vector size mismatch.");
    }
    group_phase_hints_ = std::move(phase_hints);
}

void Scheduler::recordTaskCompletion(int group_id, const Task& task, double observed_cost_proxy) {
    validateGroupId(group_id);

    const double processed_fraction =
        (task.totalMacs() == 0U)
            ? 1.0
            : static_cast<double>(task.processedMacs()) /
                  static_cast<double>(task.totalMacs());
    const double et_sample = task.earlyTerminated() ? 1.0 : 0.0;

    auto update_feedback = [&](RuntimeFeedback& feedback) {
        if (feedback.samples == 0U) {
            feedback.ema_observed_cost = observed_cost_proxy;
            feedback.ema_processed_fraction = processed_fraction;
            feedback.ema_et_rate = et_sample;
        } else {
            feedback.ema_observed_cost =
                (1.0 - kFeedbackAlpha) * feedback.ema_observed_cost +
                kFeedbackAlpha * observed_cost_proxy;
            feedback.ema_processed_fraction =
                (1.0 - kFeedbackAlpha) * feedback.ema_processed_fraction +
                kFeedbackAlpha * processed_fraction;
            feedback.ema_et_rate =
                (1.0 - kFeedbackAlpha) * feedback.ema_et_rate + kFeedbackAlpha * et_sample;
        }
        ++feedback.samples;
    };

    update_feedback(group_feedback_[static_cast<std::size_t>(group_id)]);
    if (observed_cost_proxy > 0.0) {
        double& group_target = group_target_costs_[static_cast<std::size_t>(group_id)];
        group_target =
            (group_target <= 0.0)
                ? observed_cost_proxy
                : (1.0 - kImmediateTargetBlend) * group_target +
                      kImmediateTargetBlend * observed_cost_proxy;
    }

    const int bucket_id = task.costBucketId();
    if (bucket_id >= 0 && static_cast<std::size_t>(bucket_id) < bucket_feedback_.size()) {
        update_feedback(bucket_feedback_[static_cast<std::size_t>(bucket_id)]);
        if (observed_cost_proxy > 0.0) {
            double& bucket_reference = bucket_reference_costs_[static_cast<std::size_t>(bucket_id)];
            bucket_reference =
                (bucket_reference <= 0.0)
                    ? observed_cost_proxy
                    : (1.0 - kImmediateBucketBlend) * bucket_reference +
                          kImmediateBucketBlend * observed_cost_proxy;
        }
    }

    ++completed_feedback_samples_;
    if (completed_feedback_samples_ % kFeedbackRefreshInterval == 0U) {
        refreshDynamicTargets();
    }
}

std::optional<Task> Scheduler::fetchTask(int requester_group_id,
                                         std::optional<int> preferred_output_channel) {
    validateGroupId(requester_group_id);

    WorkQueue& local_queue = queues_[static_cast<std::size_t>(requester_group_id)];
    const double target_cost = requesterTargetCost(requester_group_id);
    const int preferred_bucket = preferredBucketForTargetCost(target_cost);
    const int preferred_phase = groupPhaseHint(requester_group_id);
    const int output_family_modulus = static_cast<int>(queues_.size());

    if (cost_aware_stealing_enabled_) {
        if (std::optional<Task> task = local_queue.popBestLocalMatch(target_cost,
                                                                     preferred_bucket,
                                                                     requester_group_id,
                                                                     output_family_modulus,
                                                                     preferred_phase,
                                                                     preferred_output_channel,
                                                                     bucket_reference_costs_)) {
            return task;
        }
    } else {
        if (preferred_output_channel.has_value()) {
            if (std::optional<Task> preferred_task =
                    local_queue.popFrontMatchingOutputChannel(*preferred_output_channel)) {
                return preferred_task;
            }
        }
        if (std::optional<Task> task = local_queue.popFront()) {
            return task;
        }
    }

    if (cost_aware_stealing_enabled_) {
        const RuntimeFeedback& requester_feedback =
            group_feedback_[static_cast<std::size_t>(requester_group_id)];
        const double et_alignment_strength =
            (requester_feedback.samples == 0U)
                ? 0.0
                : std::clamp((1.0 - requester_feedback.ema_processed_fraction) +
                                 0.25 * requester_feedback.ema_et_rate,
                             0.0,
                             1.0);
        const double strict_ratio =
            std::max(0.10, kStrictStealWindowRatio - 0.05 * et_alignment_strength);
        const double relaxed_ratio =
            std::max(strict_ratio + 0.05, kRelaxedStealWindowRatio - 0.05 * et_alignment_strength);
        const double strict_window =
            std::max(kMinimumStealWindow, target_cost * strict_ratio);
        const double relaxed_window =
            std::max(kMinimumStealWindow, target_cost * relaxed_ratio);
        const std::vector<int> donor_groups =
            rankCostAwareDonorGroups(requester_group_id, target_cost);

        for (int donor_group : donor_groups) {
            WorkQueue& donor_queue = queues_[static_cast<std::size_t>(donor_group)];
            if (std::optional<Task> stolen_task = donor_queue.popBestMatchWithin(
                    target_cost,
                    strict_window,
                    preferred_bucket,
                    requester_group_id,
                    output_family_modulus,
                    preferred_phase,
                    preferred_output_channel,
                    bucket_reference_costs_,
                    0U)) {
                ++work_stealing_events_;
                return stolen_task;
            }
        }

        for (int donor_group : donor_groups) {
            WorkQueue& donor_queue = queues_[static_cast<std::size_t>(donor_group)];
            if (std::optional<Task> stolen_task = donor_queue.popBestMatchWithin(
                    target_cost,
                    relaxed_window,
                    preferred_bucket,
                    requester_group_id,
                    output_family_modulus,
                    preferred_phase,
                    preferred_output_channel,
                    bucket_reference_costs_,
                    0U)) {
                ++work_stealing_events_;
                return stolen_task;
            }
        }

        // Severe starvation fallback: requester is empty and no nearby-cost donor fit the
        // strict/relaxed windows, so pick the least disruptive donor task by the same score.
        for (int donor_group : donor_groups) {
            WorkQueue& donor_queue = queues_[static_cast<std::size_t>(donor_group)];
            if (std::optional<Task> stolen_task = donor_queue.popBestMatchWithin(
                    target_cost,
                    std::numeric_limits<double>::infinity(),
                    preferred_bucket,
                    requester_group_id,
                    output_family_modulus,
                    preferred_phase,
                    preferred_output_channel,
                    bucket_reference_costs_,
                    0U)) {
                ++work_stealing_events_;
                return stolen_task;
            }
        }
        return std::nullopt;
    }

    const int donor_group = findDonorGroup(requester_group_id);
    if (donor_group < 0) {
        return std::nullopt;
    }

    WorkQueue& donor_queue = queues_[static_cast<std::size_t>(donor_group)];
    std::optional<Task> stolen_task = donor_queue.popBack();
    if (stolen_task) {
        ++work_stealing_events_;
    }
    return stolen_task;
}

bool Scheduler::hasPendingTasks() const {
    for (const WorkQueue& queue : queues_) {
        if (!queue.empty()) {
            return true;
        }
    }
    return false;
}

std::uint64_t Scheduler::workStealingEvents() const {
    return work_stealing_events_;
}

const WorkQueue& Scheduler::groupQueue(int group_id) const {
    validateGroupId(group_id);
    return queues_[static_cast<std::size_t>(group_id)];
}

int Scheduler::findDonorGroup(int requester_group_id) const {
    std::size_t max_queue_size = 0;
    int donor_group = -1;

    for (std::size_t group = 0; group < queues_.size(); ++group) {
        if (static_cast<int>(group) == requester_group_id) {
            continue;
        }
        const std::size_t queue_size = queues_[group].size();
        if (queue_size > max_queue_size) {
            max_queue_size = queue_size;
            donor_group = static_cast<int>(group);
        }
    }

    return donor_group;
}

std::vector<int> Scheduler::rankCostAwareDonorGroups(int requester_group_id,
                                                     double requester_target_cost) const {
    struct DonorCandidate {
        int group_id{0};
        double score{0.0};
        std::size_t queue_size{0};
    };

    std::vector<DonorCandidate> candidates;
    candidates.reserve(queues_.size());

    for (std::size_t group = 0; group < queues_.size(); ++group) {
        if (static_cast<int>(group) == requester_group_id || queues_[group].empty()) {
            continue;
        }

        DonorCandidate candidate;
        candidate.group_id = static_cast<int>(group);
        candidate.score =
            std::abs(queueMeanCostForStealing(queues_[group]) - requester_target_cost);
        candidate.queue_size = queues_[group].size();
        candidates.push_back(candidate);
    }

    std::sort(candidates.begin(),
              candidates.end(),
              [](const DonorCandidate& a, const DonorCandidate& b) {
                  if (a.score != b.score) {
                      return a.score < b.score;
                  }
                  if (a.queue_size != b.queue_size) {
                      return a.queue_size > b.queue_size;
                  }
                  return a.group_id < b.group_id;
              });

    std::vector<int> ranked_groups;
    ranked_groups.reserve(candidates.size());
    for (const DonorCandidate& candidate : candidates) {
        ranked_groups.push_back(candidate.group_id);
    }
    return ranked_groups;
}

int Scheduler::preferredBucketForTargetCost(double target_cost) const {
    if (bucket_reference_costs_.empty()) {
        return -1;
    }

    int best_bucket = -1;
    double best_distance = 0.0;
    for (std::size_t bucket = 0; bucket < bucket_reference_costs_.size(); ++bucket) {
        const double reference_cost = bucket_reference_costs_[bucket];
        if (reference_cost <= 0.0) {
            continue;
        }

        const double distance = std::abs(reference_cost - target_cost);
        if (best_bucket < 0 || distance < best_distance) {
            best_bucket = static_cast<int>(bucket);
            best_distance = distance;
        }
    }
    return best_bucket;
}

int Scheduler::groupPhaseHint(int group_id) const {
    validateGroupId(group_id);

    if (group_phase_hints_.size() != queues_.size()) {
        return -1;
    }
    return group_phase_hints_[static_cast<std::size_t>(group_id)];
}

void Scheduler::refreshDynamicTargets() {
    for (std::size_t group = 0; group < group_feedback_.size(); ++group) {
        const RuntimeFeedback& feedback = group_feedback_[group];
        if (feedback.samples == 0U || feedback.ema_observed_cost <= 0.0) {
            continue;
        }

        if (group_target_costs_[group] <= 0.0) {
            group_target_costs_[group] = feedback.ema_observed_cost;
            continue;
        }

        group_target_costs_[group] =
            kRefreshOldWeight * group_target_costs_[group] +
            kRefreshObservedWeight * feedback.ema_observed_cost;
    }

    for (std::size_t bucket = 0; bucket < bucket_feedback_.size(); ++bucket) {
        const RuntimeFeedback& feedback = bucket_feedback_[bucket];
        if (feedback.samples == 0U || feedback.ema_observed_cost <= 0.0) {
            continue;
        }

        if (bucket_reference_costs_[bucket] <= 0.0) {
            bucket_reference_costs_[bucket] = feedback.ema_observed_cost;
            continue;
        }

        bucket_reference_costs_[bucket] =
            kRefreshOldWeight * bucket_reference_costs_[bucket] +
            kRefreshObservedWeight * feedback.ema_observed_cost;
    }
}

double Scheduler::queueMeanCostForStealing(const WorkQueue& queue) const {
    return queue.averageCostForStealing(bucket_reference_costs_);
}

double Scheduler::requesterTargetCost(int requester_group_id) const {
    validateGroupId(requester_group_id);

    const RuntimeFeedback& feedback =
        group_feedback_[static_cast<std::size_t>(requester_group_id)];
    const double group_target = group_target_costs_[static_cast<std::size_t>(requester_group_id)];
    const WorkQueue& requester_queue = queues_[static_cast<std::size_t>(requester_group_id)];
    if (requester_queue.empty()) {
        if (feedback.samples > 0U && group_target > 0.0) {
            return kRequesterTargetWeight * group_target +
                   (1.0 - kRequesterTargetWeight) * feedback.ema_observed_cost;
        }
        return (group_target > 0.0) ? group_target : feedback.ema_observed_cost;
    }

    const double queue_mean = queueMeanCostForStealing(requester_queue);
    if (group_target <= 0.0) {
        return queue_mean;
    }
    const double observed_cost =
        (feedback.samples > 0U && feedback.ema_observed_cost > 0.0)
            ? feedback.ema_observed_cost
            : group_target;
    return kRequesterTargetWeight * group_target +
           kRequesterObservedWeight * observed_cost +
           kRequesterQueueWeight * queue_mean;
}

void Scheduler::validateGroupId(int group_id) const {
    if (group_id < 0 || group_id >= static_cast<int>(queues_.size())) {
        throw std::out_of_range("Group id out of range in Scheduler.");
    }
}
