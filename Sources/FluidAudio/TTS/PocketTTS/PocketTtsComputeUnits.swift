import CoreML
import Foundation

/// Per-stage Core ML compute-unit overrides for PocketTTS.
///
/// Backport of upstream FluidAudio #893 (issue #881) onto v0.15.2, kept
/// API-compatible with it so a later move to an upstream release needs no
/// caller changes. Every field defaults to `nil`, which keeps the routing
/// this version shipped with for that stage (documented in
/// `PocketTtsModelStore`). Set a field to move a single stage. Use
/// `avoidNeuralEngine` on hardware where the Neural Engine rejects or
/// distorts a stage: the default `.all` places `flow_decoder_fused` on the
/// ANE and nothing else in this version lets a caller move it.
public struct PocketTtsComputeUnits: Sendable, Hashable {
    /// `cond_prefill`. Default `.all`.
    public var conditioner: MLComputeUnits?
    /// `flowlm_step` / `flowlm_stepv2`. Default `.all`.
    public var flowLM: MLComputeUnits?
    /// `flow_decoder_fused`. Default `.all` (the one stage that is 100% ANE).
    public var flowDecoder: MLComputeUnits?
    /// `mimi_decoder`. Default `.cpuOnly`; keep it off the ANE, where its fp16
    /// streaming-state feedback compounds into audible beeps.
    public var mimiDecoder: MLComputeUnits?
    /// `mimi_encoder` (voice cloning only). Default `.cpuAndGPU`.
    public var mimiEncoder: MLComputeUnits?

    public init(
        conditioner: MLComputeUnits? = nil,
        flowLM: MLComputeUnits? = nil,
        flowDecoder: MLComputeUnits? = nil,
        mimiDecoder: MLComputeUnits? = nil,
        mimiEncoder: MLComputeUnits? = nil
    ) {
        self.conditioner = conditioner
        self.flowLM = flowLM
        self.flowDecoder = flowDecoder
        self.mimiDecoder = mimiDecoder
        self.mimiEncoder = mimiEncoder
    }

    /// The routing this version shipped with for every stage.
    public static let `default` = PocketTtsComputeUnits()

    /// Every stage on the same units.
    public static func uniform(_ units: MLComputeUnits) -> PocketTtsComputeUnits {
        PocketTtsComputeUnits(
            conditioner: units, flowLM: units, flowDecoder: units,
            mimiDecoder: units, mimiEncoder: units)
    }

    /// No stage may be scheduled on the Neural Engine: GPU for the
    /// transformer stages, CPU for the mimi decoder.
    public static let avoidNeuralEngine = PocketTtsComputeUnits(
        conditioner: .cpuAndGPU, flowLM: .cpuAndGPU, flowDecoder: .cpuAndGPU,
        mimiDecoder: .cpuOnly, mimiEncoder: .cpuAndGPU)

    var resolvedConditioner: MLComputeUnits { conditioner ?? .all }
    var resolvedFlowLM: MLComputeUnits { flowLM ?? .all }
    var resolvedFlowDecoder: MLComputeUnits { flowDecoder ?? .all }
    var resolvedMimiDecoder: MLComputeUnits { mimiDecoder ?? .cpuOnly }
    var resolvedMimiEncoder: MLComputeUnits { mimiEncoder ?? .cpuAndGPU }
}

extension MLComputeUnits {
    /// Stable log label (the raw value is an opaque integer).
    var pocketTtsLabel: String {
        switch self {
        case .cpuOnly: return "cpuOnly"
        case .cpuAndGPU: return "cpuAndGPU"
        case .cpuAndNeuralEngine: return "cpuAndNeuralEngine"
        case .all: return "all"
        @unknown default: return "unknown(\(rawValue))"
        }
    }
}
