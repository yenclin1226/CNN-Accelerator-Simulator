#pragma once

enum class ExecutionMode {
    Int8BitParallel,
    Int8BitSerial
};

enum class BroadcastMode {
    DemandDriven,
    SnapeaFixedSchedule
};

enum class PipelineMode {
    BaselineConvRelu,
    FusedConvEarlyTerminationRelu
};

enum class MacOrderingPolicy {
    Importance,
    EtAwareNegativeFirst
};

enum class ZeroRunOrderMode {
    ExecutionOrder,
    KernelOrder
};
