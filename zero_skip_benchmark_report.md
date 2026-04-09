# Zero-Skip and Bit-Column Benchmark Report

## Summary

This report captures the high-level changes and benchmark results after adding:

- reactive zero-MAC skipping
- proactive zero-run skipping
- configurable proactive run definition:
  - execution-order mode
  - kernel-order mode
- bit-column skipping
- dual-view reporting for ET-only skips versus total skips

The main outcome is that the simulator now distinguishes between:

- work skipped because the task ends early through ET
- work skipped because zero-valued computation is avoided before it starts
- work skipped inside a started MAC because empty bit positions are jumped over

Across the broad quick-suite benchmark runs, execution-order proactive mode was consistently better than kernel-order proactive mode on average cycle count. In the focused memory-pressure stress case, the two proactive modes were nearly tied in cycles, but they produced very different skip breakdowns.

## High-Level Changes

The implementation changed the simulator behavior at a policy level, not just at a reporting level.

### 1. Zero work is now removed before it becomes real work

Previously, the simulator could stop a task early when ET proved the final output would remain non-positive, but inside a live task it still stepped through the serial work mostly in order.

Now, the simulator can avoid useless work earlier:

- reactive zero skip removes an isolated zero MAC before it starts
- proactive zero-run skip removes a contiguous stretch of zero MACs in one decision
- bit-column skip jumps over empty bit positions inside a started MAC

This means some work now disappears before:

- activation/weight memory requests
- broadcaster demand formation
- multiplier and accumulator activity

### 2. Proactive skipping is configurable in two ways

The simulator now supports two definitions of a proactive zero run:

- `execution-order`: zero runs are defined over the final scheduled worklist
- `kernel-order`: zero runs are defined over original kernel traversal order

This matters because the simulator already reorders work for importance and ET behavior. The two modes therefore represent different assumptions about what “look ahead” means.

### 3. Reporting now separates skip causes

The benchmark and simulator reports now preserve both:

- ET-only views
- total-skip views

The reporting also separates zero skipping into:

- reactive-only MAC skips
- proactive-only MAC skips

This makes it possible to answer questions like:

- Did a speedup come from ET or from zero skipping?
- Is kernel-order causing more isolated reactive skips and fewer proactive runs?
- Is bit-column skip carrying most of the serial savings even when proactive mode changes?

## Metric Meanings

The new metric meanings are:

- `macs_skipped_total`: MACs that never started for any reason
- `macs_skipped_et_only`: MACs skipped because ET ended the rest of the task
- `macs_skipped_reactive_only`: isolated zero MACs skipped before they started
- `macs_skipped_proactive_only`: zero MACs skipped as part of a proactive run
- `macs_skipped_zero_only`: reactive-only plus proactive-only
- `bit_steps_skipped_total`: all skipped serial bit positions
- `bit_steps_skipped_et_only`: bit positions skipped because ET truncated remaining work
- `bit_steps_skipped_bit_column_only`: bit positions skipped by jumping over empty weight bit columns
- `zero_run_events`: number of proactive run decisions, not the number of MACs inside those runs

Important correctness rule:

- ET-only metrics remain zero in ET-off runs, even when zero skipping and bit-column skipping are enabled.

## Benchmark Setup

The main benchmark runs used:

- `bitserial_bench`
- quick suite, detailed mode, all grouping policies
- paired ET (`Off` and `On`)
- reactive zero skip enabled
- proactive zero-run skip enabled
- bit-column skip enabled
- compared once with execution-order proactive mode
- compared once with kernel-order proactive mode

Generated benchmark artifacts:

- [quick_exec_detailed.csv](/home/kylewang02/eecs573-comp/573-og/build/quick_exec_detailed.csv)
- [quick_kernel_detailed.csv](/home/kylewang02/eecs573-comp/573-og/build/quick_kernel_detailed.csv)
- [memory_exec_focus.csv](/home/kylewang02/eecs573-comp/573-og/build/memory_exec_focus.csv)
- [memory_kernel_focus.csv](/home/kylewang02/eecs573-comp/573-og/build/memory_kernel_focus.csv)

The full extended memory-pressure sweep across all grouping policies and both proactive modes exceeded the available time window, so the deeper memory-pressure comparison was narrowed to the representative focused scenario:

- `extended-synthetic-memory-pressure-uniform-seed-1`
- grouping policy: `ETAwareCostBalancedMemoryAware`

## Results

### Quick Suite: Execution-Order vs Kernel-Order

Across 80 aggregate rows in the quick suite:

- execution-order ET off average cycles: `37,604.2`
- kernel-order ET off average cycles: `38,701.3`
- execution-order ET on average cycles: `25,166.6`
- kernel-order ET on average cycles: `25,648.0`

Average kernel-order slowdown relative to execution-order:

- ET off: about `1.079x`
- ET on: about `1.033x`

Interpretation:

- execution-order mode generally converts more zero work into proactive runs
- kernel-order mode leaves more zero work as isolated skips or work later removed by ET
- execution-order therefore tends to reduce cycles more effectively in the broad quick suite

### Quick Suite: Mechanism Breakdown

Average per-row skip breakdown in the quick suite:

#### ET Off

- execution-order:
  - reactive-only MAC skips: `2,158.0`
  - proactive-only MAC skips: `49,940.2`
  - ET-only MAC skips: `0.0`
  - bit-column skipped bit-steps: `248,906.2`
  - zero-run events: `7,385.5`

- kernel-order:
  - reactive-only MAC skips: `30,083.0`
  - proactive-only MAC skips: `22,015.2`
  - ET-only MAC skips: `0.0`
  - bit-column skipped bit-steps: `248,906.2`
  - zero-run events: `8,327.2`

Interpretation:

- both modes skip the same total amount of zero MAC work at a high level
- execution-order packs much more of that work into proactive runs
- kernel-order produces many more isolated reactive skips
- bit-column behavior is effectively unchanged between the two modes

#### ET On

- execution-order:
  - reactive-only MAC skips: `48.2`
  - proactive-only MAC skips: `38,680.4`
  - ET-only MAC skips: `28,736.5`
  - bit-column skipped bit-steps: `145,344.2`
  - zero-run events: `3,461.4`

- kernel-order:
  - reactive-only MAC skips: `13,958.5`
  - proactive-only MAC skips: `11,140.2`
  - ET-only MAC skips: `42,366.4`
  - bit-column skipped bit-steps: `145,344.2`
  - zero-run events: `3,912.4`

Interpretation:

- once ET is enabled, the choice of proactive mode shifts where skip credit lands
- execution-order gets more benefit earlier through proactive runs
- kernel-order leaves more work to be removed later by ET
- this matches the cycle result: execution-order remains better

### ET Interaction Depth

Average ET speedup in the quick suite:

- execution-order: `1.3221x`
- kernel-order: `1.3710x`

This does not mean kernel-order is better overall. It means:

- kernel-order leaves more work available for ET to cut away later
- execution-order removes more zero work before ET has a chance to act

So the higher ET-only activity in kernel-order is mostly a redistribution of skip cause, not proof of a better total policy.

### Memory-Pressure Focus Scenario

Focused scenario:

- `extended-synthetic-memory-pressure-uniform-seed-1`
- grouping: `ETAwareCostBalancedMemoryAware`

Cycle results:

- ET off:
  - execution-order: `3,163,302`
  - kernel-order: `3,158,821`

- ET on:
  - execution-order: `2,158,770`
  - kernel-order: `2,162,070`

Interpretation:

- the two proactive modes are almost tied in this stress case
- locality and memory pressure dominate enough that proactive run definition changes the cycle count only slightly

Mechanism breakdown:

#### ET Off

- execution-order:
  - reactive-only MAC skips: `33,679`
  - proactive-only MAC skips: `558,827`
  - ET-only MAC skips: `0`
  - bit-column skipped bit-steps: `11,923,756`
  - zero-run events: `95,827`

- kernel-order:
  - reactive-only MAC skips: `366,808`
  - proactive-only MAC skips: `225,698`
  - ET-only MAC skips: `0`
  - bit-column skipped bit-steps: `11,923,756`
  - zero-run events: `116,068`

#### ET On

- execution-order:
  - reactive-only MAC skips: `0`
  - proactive-only MAC skips: `304,379`
  - ET-only MAC skips: `855,974`
  - bit-column skipped bit-steps: `8,068,602`
  - zero-run events: `8,415`

- kernel-order:
  - reactive-only MAC skips: `187,739`
  - proactive-only MAC skips: `116,640`
  - ET-only MAC skips: `855,974`
  - bit-column skipped bit-steps: `8,068,602`
  - zero-run events: `59,720`

Interpretation:

- even when cycle counts are almost the same, the proactive mode strongly changes skip composition
- execution-order still consolidates more zero work into proactive runs
- kernel-order again falls back to more reactive skips
- ET behavior is essentially identical in the ET-on memory-pressure case

## Overall Conclusions

### 1. Execution-order proactive mode is the stronger default

In the broad quick-suite benchmark set, execution-order proactive mode consistently outperformed kernel-order proactive mode on average cycle count.

The main reason is that it better aligns proactive skipping with the actual scheduled execution order, so more zero work is removed in larger proactive chunks before it turns into lane activity.

### 2. Kernel-order changes attribution more than outcome in stress cases

In the focused memory-pressure scenario, execution-order and kernel-order produced nearly identical cycle counts. The major difference was not total performance, but how skipped work was distributed across:

- reactive-only
- proactive-only
- ET-only

### 3. ET and zero skipping are complementary, not redundant

The results show a consistent pattern:

- zero skipping removes obviously useless work before it starts
- ET removes the remaining tail once the final output is provably dead

Changing proactive mode shifts the balance between those two mechanisms, but both remain useful.

### 4. Bit-column skip contributes a large and stable share of serial savings

Bit-column skipped bit-steps were large in every run and were essentially unchanged between execution-order and kernel-order proactive modes for the same workload set. That is expected because bit-column skip acts inside a started MAC and is mostly independent of proactive run definition.

## Validation

The following verification completed successfully after the reporting and benchmark-driver changes:

- `ctest --output-on-failure`

This covered:

- helper/report logic
- benchmark regression invariants
- real-world benchmark scenarios
- targeted zero-skip and ET interaction tests

## Recommended Default Policy

For future experiments, the recommended default configuration is:

- reactive zero skip: enabled
- proactive zero-run skip: enabled
- zero-run order: `execution`
- bit-column skip: enabled

Reason:

- it gave the best broad quick-suite cycle results
- it converts more zero work into proactive skips instead of leaving it to reactive handling or later ET cleanup
- it aligns best with the simulator’s actual scheduled execution behavior
