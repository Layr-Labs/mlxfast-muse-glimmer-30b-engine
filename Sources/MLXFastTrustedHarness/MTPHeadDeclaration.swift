import Foundation
import MLXFastCore

// Head delivery for the native-MTP track.
//
// OPERATOR-RATIFIED 2026-08-14. The MTP head is part of the competitive
// surface: a submission may bring its own. It declares one in
// `mtp-head.manifest.json`, which is an editable path, and the RUNNER resolves
// that declaration pre-sandbox the way it resolves a hidden golden -- refuse on
// oversize. A declared digest is parsed and carried when present but is NOT
// verified against the head bytes and is not a gate. Per DECIDE-2 (Q-B) a
// bring-your-own head is NOT required to declare one, and a digestless head and
// a wrong-digest head are treated identically: both are bounded by SIZE ONLY.
//
// THE SAFETY ARGUMENT, in one line: a head only PROPOSES tokens. The
// organizer-pinned target decides every emitted token and the trusted parent
// re-checks the whole stream against a hidden serial trajectory after the clock
// stops, so a substituted head moves the accept rate -- which is the game --
// and cannot move the output.
//
// THIS FILE IS TRUSTED CODE. It parses the declaration and applies the size
// gate; it never decides whether a run passes.
//
// THE DECLARABLE DECODERS ARE THE FIXTURE'S (muse-glimmer-30b-mlx-v1).
// `spec.decoder` names the decoder a submission arms. The names a declaration
// may use are the contract fixture's `allowed_modes` without `serial` --
// today `dflash` alone -- read by `declarableDecoders(contractFixture:)`.
// tools/spec-declaration.sh reads the same list, so the two readers cannot
// disagree about a submission. An absent key means `dflash`. Any other name --
// including the older `mtp` slot spelling -- is REFUSED by name. `Decoder`
// below holds what this harness knows about each decoder (its size cap); a
// decoder the fixture allows but `Decoder` lacks is refused, never served with
// a guessed cap. `max_bytes` may lower the 2 GiB cap and may not raise it.
//
// THERE IS NO `arm` SELECTION KEY. `spec.decoder` is the selection, and it
// lives beside the depth it goes with. `arm` is an unread key like any other
// unknown key.
//
// THE HEAD IS NOT STAGED OR MEASURED HERE. It is a separate pinned export (the
// quantized DFlash drafter, pinned in the contract fixture's `mtp_head`
// block), staged beside the target by `./setup-mtp-head.sh`, so
// `source: "pinned"` means "the export this track pins". A submission carries
// no head weights, so this file has no directory to digest and no artifact to
// fetch. It parses the declaration and applies the size gate.

/// One parsed head declaration (`mtp-head.manifest.json`).
public struct MTPHeadDeclaration: Equatable, Sendable {
    public enum Source: String, Equatable, CaseIterable, Sendable {
        /// The head embedded in the organizer's pinned target checkpoint. The
        /// default, and what an ABSENT manifest means.
        case pinned
        /// Fetched by the runner from `sourceURL` (`hf:<repo>@<rev>` or
        /// `r2:<key>`) into the run's private directory.
        case remote
        /// Shipped in the submission under the editable weights directory
        /// named by `path`.
        case inBranch = "in_branch"
    }

    /// Which speculative decoder the declaration arms.
    ///
    /// ONE case, because this track declares one decoder: the pinned DFlash
    /// drafter. The contract fixture's `mtp_head.decoder` is `dflash` and its
    /// `allowed_modes` is serial + dflash, so a declaration that asks for any
    /// other decoder is refused BY NAME rather than quietly served.
    public enum Decoder: String, Equatable, CaseIterable, Sendable {
        /// The DFlash drafter, this track's decoder, and every default here.
        case dflash

        /// The track cap on what this decoder's declaration may bound. A
        /// declaration may lower it and may not raise it.
        public var trackMaxBytes: Int {
            switch self {
            case .dflash: return 2_147_483_648
            }
        }

        /// How a refusal names this decoder's drafter.
        public var declarationNoun: String {
            switch self {
            case .dflash: return "DFlash"
            }
        }
    }

    public let decoder: Decoder
    public let source: Source
    public let sourceURL: String?
    public let path: String?
    public let sha256: String?
    public let bytes: Int
    public let maxBytes: Int

    public init(
        source: Source,
        decoder: Decoder = .dflash,
        sourceURL: String? = nil,
        path: String? = nil,
        sha256: String? = nil,
        bytes: Int = 0,
        maxBytes: Int? = nil
    ) {
        self.decoder = decoder
        self.source = source
        self.sourceURL = sourceURL
        self.path = path
        self.sha256 = sha256
        self.bytes = bytes
        self.maxBytes = maxBytes ?? decoder.trackMaxBytes
    }

    /// 2 GiB, the head cap. Mirrored in `mtp-head.manifest.json`
    /// (`max_bytes`) and in the contract manifest's
    /// `editableSurfaceByteBudget.exemptPathMaxBytes`; a declaration may lower
    /// it and may not raise it.
    public static let defaultMaxBytes = Decoder.dflash.trackMaxBytes

    /// The default when no manifest exists at all.
    public static let pinnedDefault = MTPHeadDeclaration(source: .pinned)

    public static let relativePath = "mtp-head.manifest.json"

    /// The contract fixture, relative to a contract root.
    public static let contractFixtureRelativePath =
        "fixtures/muse_glimmer_30b_mlx_v1_track.json"

    /// The decoder names a declaration may use: the contract fixture's
    /// `allowed_modes` without `serial`. FAILS CLOSED on a fixture that cannot
    /// be read or that allows no speculative decoder.
    public static func declarableDecoders(contractFixture url: URL) throws -> [String] {
        guard let data = FileManager.default.contents(atPath: url.path),
            let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else {
            throw MLXFastError.invalidInput(
                "the track contract fixture at \(url.path) could not be read as a "
                    + "JSON object; the declarable decoders come from its allowed_modes")
        }
        guard let modes = root["allowed_modes"] as? [String] else {
            throw MLXFastError.invalidInput(
                "the track contract fixture at \(url.path) has no allowed_modes list "
                    + "of strings")
        }
        let decoders = modes.filter { $0 != "serial" }
        guard !decoders.isEmpty else {
            throw MLXFastError.invalidInput(
                "the track contract fixture at \(url.path) allows no speculative "
                    + "decoder in allowed_modes")
        }
        return decoders
    }

    /// How a refusal names the head when no declaration has been parsed yet.
    public static let declarationNoun = Decoder.dflash.declarationNoun

    /// Parse a declaration, FAILING CLOSED on anything malformed.
    ///
    /// Only two things select the pinned head: the file being absent, and an
    /// explicit `source: "pinned"`. A manifest that is present but unreadable,
    /// unparseable, or internally inconsistent is a REFUSAL -- never a silent
    /// fall back -- because "your head declaration was broken so we quietly
    /// scored you on the pinned head" is exactly the failure mode that makes a
    /// leaderboard number unattributable.
    ///
    /// `declarableDecoders` is the fixture's list
    /// (`declarableDecoders(contractFixture:)`); there is no built-in default,
    /// so no caller can judge a declaration against a second copy of it.
    public static func parse(
        contentsOf url: URL,
        declarableDecoders: [String]
    ) throws -> MTPHeadDeclaration {
        guard let data = FileManager.default.contents(atPath: url.path) else {
            throw MLXFastError.invalidInput(
                "the \(declarationNoun) head declaration at \(url.path) "
                    + "exists but could not be read; refusing to fall back to "
                    + "the pinned head")
        }
        return try parse(data: data, origin: url.path, declarableDecoders: declarableDecoders)
    }

    public static func parse(
        data: Data,
        origin: String,
        declarableDecoders: [String]
    ) throws -> MTPHeadDeclaration {
        guard let root = (try? JSONSerialization.jsonObject(with: data))
            as? [String: Any]
        else {
            throw MLXFastError.invalidInput(
                "the \(declarationNoun) head declaration at \(origin) is not a "
                    + "JSON object")
        }
        // The decoder is read FIRST: it names the drafter every later refusal
        // talks about, and it selects the size cap the declaration is bounded
        // by. An absent key is `dflash`.
        let spec = root["spec"] as? [String: Any]
        let rawDecoder = (spec?["decoder"] as? String) ?? Decoder.dflash.rawValue
        guard declarableDecoders.contains(rawDecoder) else {
            throw MLXFastError.invalidInput(
                "the head declaration at \(origin) names an unknown decoder "
                    + "'\(rawDecoder)'; this track declares "
                    + declarableDecoders.joined(separator: ", ")
                    + " (the fixture's allowed_modes)")
        }
        guard let decoder = Decoder(rawValue: rawDecoder) else {
            throw MLXFastError.invalidInput(
                "the head declaration at \(origin) names decoder '\(rawDecoder)', "
                    + "which the fixture's allowed_modes lists but this harness has "
                    + "no size cap for; refusing rather than guessing one")
        }
        let noun = decoder.declarationNoun

        let rawSource = (root["source"] as? String) ?? Source.pinned.rawValue
        guard let source = Source(rawValue: rawSource) else {
            throw MLXFastError.invalidInput(
                "the \(noun) head declaration at \(origin) names an unknown source "
                    + "'\(rawSource)'; expected one of "
                    + Source.allCases.map(\.rawValue).joined(separator: ", "))
        }
        let trackMaxBytes = decoder.trackMaxBytes
        let maxBytes = (root["max_bytes"] as? NSNumber)?.intValue
            ?? trackMaxBytes
        guard maxBytes > 0, maxBytes <= trackMaxBytes else {
            throw MLXFastError.invalidInput(
                "the \(noun) head declaration at \(origin) sets max_bytes "
                    + "\(maxBytes); it must be positive and may not exceed the "
                    + "\(decoder.rawValue) track cap \(trackMaxBytes)")
        }
        let sourceURL = root["source_url"] as? String
        let path = root["path"] as? String
        let sha256 = (root["sha256"] as? String)?.lowercased()
        let bytes = (root["bytes"] as? NSNumber)?.intValue ?? 0

        // REQUANT-ONLY (David ruling, 2026-08-26). `pinned` is the ONLY source
        // this track accepts. The head is the organizer's own weights; a
        // participant may declare a re-quantization of them and may not
        // substitute weights of their own. `remote` and `in_branch` are the two
        // spellings of "load bytes the participant chose", so both are named
        // refusals rather than gated flows.
        //
        // WHY THE CASES STAY IN THE ENUM. Deleting them would make a manifest
        // that names one fail as "unknown source", which reads like a typo. The
        // participant did not typo; they used a mode this track retired, and the
        // refusal should say so and say what replaced it.
        switch source {
        case .pinned:
            break
        case .remote:
            throw MLXFastError.invalidInput(
                "the \(noun) head declaration at \(origin) selects source "
                    + "'remote'; this track accepts source 'pinned' only. The "
                    + "\(noun) drafter is a separate pinned export "
                    + "(fixtures/muse_glimmer_30b_mlx_v1_track.json), and custom "
                    + "drafter weights are not accepted. Declare a "
                    + "re-quantization of the pinned head instead")
        case .inBranch:
            throw MLXFastError.invalidInput(
                "the \(noun) head declaration at \(origin) selects source "
                    + "'in_branch'; this track accepts source 'pinned' only. A "
                    + "submission carries no drafter weights -- both the drafter and its source artifact "
                    + "are staged from their own pins -- so there is nothing an "
                    + "in-branch path could name. Declare a re-quantization of "
                    + "the pinned head instead")
        }

        // THE SIZE GATE SURVIVES THE NARROWING. It used to be reachable only
        // for a non-pinned source, which after the ruling would make it dead
        // code -- and deleting it would silently drop the one numeric bound a
        // declaration still carries. A `pinned` declaration may state the
        // `bytes` of the head it expects (a re-quantized head is smaller than
        // the shipped one, and stating it is how a participant records what they
        // expect), so the bound is "if you state a size, it must fit the cap"
        // rather than "non-pinned sources must state one". `bytes: 0` means
        // "not stated", which is what the checked-in declaration says.
        if bytes != 0 {
            guard bytes > 0 else {
                throw MLXFastError.invalidInput(
                    "the \(noun) head declaration at \(origin) states bytes "
                        + "\(bytes); a stated byte count must be positive")
            }
            guard bytes <= maxBytes else {
                throw MLXFastError.invalidInput(
                    "the declared \(noun) head is \(bytes) bytes, above the "
                        + "\(maxBytes)-byte cap in \(origin)")
            }
        }
        return MTPHeadDeclaration(
            source: source,
            decoder: decoder,
            sourceURL: sourceURL,
            path: path,
            sha256: sha256,
            bytes: bytes,
            maxBytes: maxBytes
        )
    }

    /// Read the declaration next to a contract root, treating ABSENCE as the
    /// pinned default and everything else as parse-or-refuse. The declarable
    /// decoders come from the contract fixture under the same root.
    public static func resolve(
        contractRoot: URL
    ) throws -> MTPHeadDeclaration {
        let url = contractRoot.appendingPathComponent(relativePath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return pinnedDefault
        }
        let decoders = try declarableDecoders(
            contractFixture: contractRoot.appendingPathComponent(contractFixtureRelativePath))
        return try parse(contentsOf: url, declarableDecoders: decoders)
    }
}
