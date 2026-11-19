// Copyright © 2026 Eigen Labs.
//
// MLXRunners — the runner layer's own view of a drafter that reads the
// target's intermediate state.
//
// A drafter may need the target model to keep certain hidden rows on every
// forward, which costs work a serial build must not pay. The batching
// contracts publish no such switch, and this layer may not add one to the
// library, so the seam lives here: a drafter that needs the tap conforms to
// `RunnerTargetContextArming`, and the engine build arms it once, where both
// directions are visible. A drafter that does not conform is left alone.

import Foundation

/// A drafter whose target-side capture is switched on and off per build.
public protocol RunnerTargetContextArming: AnyObject {
    /// Arm or disarm the target's intermediate-state capture for this build.
    /// - Parameter armed: true when this build will speculate, false when it
    ///   will decode serially.
    func setTargetContextArmed(_ armed: Bool) throws
}
