#include "WorkQueue.h"

#include <cmath>
#include <limits>

namespace {

double effectiveTaskCost(const Task& task, const std::vector<double>& bucket_reference_costs) {
    // Bucket reference costs let runtime-observed bucket behavior gently correct static prediction
    // without hiding per-task variation needed for local matching.
    const int bucket_id = task.costBucketId();
    if (bucket_id >= 0 &&
        static_cast<std::size_t>(bucket_id) < bucket_reference_costs.size() &&
        bucket_reference_costs[static_cast<std::size_t>(bucket_id)] > 0.0) {
        constexpr double kBucketReferenceBlend = 0.70;
        return kBucketReferenceBlend * bucket_reference_costs[static_cast<std::size_t>(bucket_id)] +
               (1.0 - kBucketReferenceBlend) * task.predictedCost();
    }
    return task.predictedCost();
}

std::deque<Task>::iterator findBestTaskByScore(
    std::deque<Task>& queue,
    double target_cost,
    double max_distance,
    int preferred_bucket_id,
    int preferred_output_family,
    int output_family_modulus,
    int preferred_phase_hint,
    std::optional<int> preferred_output_channel,
    const std::vector<double>& bucket_reference_costs,
    std::size_t scan_limit) {
    constexpr double kMinCostScale = 16.0;
    constexpr double kCostWeight = 3.0;
    constexpr double kPreferredOutputPenalty = 0.60;
    constexpr double kOutputFamilyPenalty = 0.35;
    constexpr double kBucketPenalty = 0.40;
    constexpr double kPhasePenalty = 0.20;
    constexpr double kAgePenalty = 0.10;

    if (queue.empty()) {
        return queue.end();
    }

    const std::size_t limit =
        (scan_limit == 0U) ? queue.size() : std::min(queue.size(), scan_limit);
    auto best_it = queue.end();
    double best_score = 0.0;
    double best_cost_distance = 0.0;
    double best_raw_distance = 0.0;
    std::size_t best_index = 0U;

    for (std::size_t index = 0; index < limit; ++index) {
        auto it = queue.begin() + static_cast<std::ptrdiff_t>(index);
        const double effective_cost = effectiveTaskCost(*it, bucket_reference_costs);
        const double cost_distance = std::abs(effective_cost - target_cost);
        if (cost_distance > max_distance) {
            continue;
        }

        const bool channel_match =
            preferred_output_channel.has_value() &&
            it->outputChannel() == *preferred_output_channel;
        const double normalized_cost_distance =
            cost_distance / std::max(kMinCostScale, std::max(target_cost, effective_cost));
        const double output_channel_penalty =
            (preferred_output_channel.has_value() && !channel_match) ? kPreferredOutputPenalty : 0.0;
        double output_family_penalty = 0.0;
        if (!channel_match && output_family_modulus > 0 && preferred_output_family >= 0) {
            const int task_output_family = it->outputChannel() % output_family_modulus;
            output_family_penalty =
                (task_output_family == preferred_output_family) ? 0.0 : kOutputFamilyPenalty;
        }
        const double bucket_penalty =
            (preferred_bucket_id >= 0)
                ? static_cast<double>(std::abs(it->costBucketId() - preferred_bucket_id)) *
                      kBucketPenalty
                : 0.0;
        const double phase_penalty =
            (preferred_phase_hint >= 0 && it->phaseAffinityHint() != preferred_phase_hint)
                ? kPhasePenalty
                : 0.0;
        const double age_penalty =
            (limit <= 1U)
                ? 0.0
                : kAgePenalty *
                      (static_cast<double>(index) / static_cast<double>(limit - 1U));
        const double score =
            normalized_cost_distance * kCostWeight + output_channel_penalty +
            output_family_penalty + bucket_penalty + phase_penalty + age_penalty;
        const double raw_distance = std::abs(it->predictedCost() - target_cost);

        if (best_it == queue.end() || score < best_score ||
            (score == best_score && cost_distance < best_cost_distance) ||
            (score == best_score && cost_distance == best_cost_distance &&
             raw_distance < best_raw_distance) ||
            (score == best_score && cost_distance == best_cost_distance &&
             raw_distance == best_raw_distance && index < best_index) ||
            (score == best_score && cost_distance == best_cost_distance &&
             raw_distance == best_raw_distance && index == best_index && it->id() < best_it->id())) {
            best_it = it;
            best_score = score;
            best_cost_distance = cost_distance;
            best_raw_distance = raw_distance;
            best_index = index;
        }
    }

    return best_it;
}

}  // namespace

void WorkQueue::push(Task task) {
    queue_.push_back(std::move(task));
}

std::optional<Task> WorkQueue::popFront() {
    if (queue_.empty()) {
        return std::nullopt;
    }
    Task task = std::move(queue_.front());
    queue_.pop_front();
    return task;
}

std::optional<Task> WorkQueue::popBack() {
    if (queue_.empty()) {
        return std::nullopt;
    }
    Task task = std::move(queue_.back());
    queue_.pop_back();
    return task;
}

std::optional<Task> WorkQueue::popFrontMatchingOutputChannel(int output_channel) {
    for (auto it = queue_.begin(); it != queue_.end(); ++it) {
        if (it->outputChannel() == output_channel) {
            Task task = std::move(*it);
            queue_.erase(it);
            return task;
        }
    }
    return std::nullopt;
}

std::optional<Task> WorkQueue::popBestLocalMatch(
    double target_cost,
    int preferred_bucket_id,
    int preferred_output_family,
    int output_family_modulus,
    int preferred_phase_hint,
    std::optional<int> preferred_output_channel,
    const std::vector<double>& bucket_reference_costs) {
    constexpr std::size_t kLocalQueueScanLimit = 16;

    return popBestMatchWithin(target_cost,
                              std::numeric_limits<double>::infinity(),
                              preferred_bucket_id,
                              preferred_output_family,
                              output_family_modulus,
                              preferred_phase_hint,
                              preferred_output_channel,
                              bucket_reference_costs,
                              kLocalQueueScanLimit);
}

std::optional<Task> WorkQueue::popClosestPredictedCost(double target_cost,
                                                       std::optional<int> preferred_output_channel,
                                                       const std::vector<double>& bucket_reference_costs) {
    return popBestMatchWithin(target_cost,
                              std::numeric_limits<double>::infinity(),
                              -1,
                              -1,
                              0,
                              -1,
                              preferred_output_channel,
                              bucket_reference_costs,
                              0U);
}

std::optional<Task> WorkQueue::popBestMatchWithin(
    double target_cost,
    double max_distance,
    int preferred_bucket_id,
    int preferred_output_family,
    int output_family_modulus,
    int preferred_phase_hint,
    std::optional<int> preferred_output_channel,
    const std::vector<double>& bucket_reference_costs,
    std::size_t scan_limit) {
    auto best_it = findBestTaskByScore(queue_,
                                       target_cost,
                                       max_distance,
                                       preferred_bucket_id,
                                       preferred_output_family,
                                       output_family_modulus,
                                       preferred_phase_hint,
                                       preferred_output_channel,
                                       bucket_reference_costs,
                                       scan_limit);
    if (best_it == queue_.end()) {
        return std::nullopt;
    }

    Task task = std::move(*best_it);
    queue_.erase(best_it);
    return task;
}

std::optional<Task> WorkQueue::popClosestPredictedCostWithin(
    double target_cost,
    double max_distance,
    std::optional<int> preferred_output_channel,
    const std::vector<double>& bucket_reference_costs) {
    return popBestMatchWithin(target_cost,
                              max_distance,
                              -1,
                              -1,
                              0,
                              -1,
                              preferred_output_channel,
                              bucket_reference_costs,
                              0U);
}

double WorkQueue::averagePredictedCost() const {
    if (queue_.empty()) {
        return 0.0;
    }

    double sum = 0.0;
    for (const Task& task : queue_) {
        sum += task.predictedCost();
    }
    return sum / static_cast<double>(queue_.size());
}

double WorkQueue::averageCostForStealing(const std::vector<double>& bucket_reference_costs) const {
    if (queue_.empty()) {
        return 0.0;
    }

    double sum = 0.0;
    for (const Task& task : queue_) {
        sum += effectiveTaskCost(task, bucket_reference_costs);
    }
    return sum / static_cast<double>(queue_.size());
}

double WorkQueue::predictedCostVariance() const {
    if (queue_.size() <= 1U) {
        return 0.0;
    }

    const double mean = averagePredictedCost();
    double variance = 0.0;
    for (const Task& task : queue_) {
        const double diff = task.predictedCost() - mean;
        variance += diff * diff;
    }
    return variance / static_cast<double>(queue_.size());
}

bool WorkQueue::empty() const {
    return queue_.empty();
}

std::size_t WorkQueue::size() const {
    return queue_.size();
}
