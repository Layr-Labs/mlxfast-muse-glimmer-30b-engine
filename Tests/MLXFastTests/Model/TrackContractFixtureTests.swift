import Foundation
import MLXFastCore
import Testing

// Contract tests for `fixtures/muse_glimmer_30b_mlx_v1_track.json`.
//
// The track is STAMPED AND NOT ARMED, AND ITS TARGET IS PINNED. So:
//
// - `official_scoring_enabled` is false, the timed pool is empty, `live_goldens`
//   is empty and the hidden golden carries this track's pending sentinel;
// - every pin that names the target's bytes is CONCRETE: the checkpoint
//   revision, the drafter's two file digests and the config fixture's digest;
// - the geometry the fixture declares is the pinned snapshot's own, read back
//   out of the checked-in byte copy of its `config.json`.
//
// The suite reads ONE switch (`official_scoring_enabled`) and asserts the
// matching state as a whole, so an arming commit has to move every pin at once
// and cannot leave the file half-armed.
//
// Nothing here reaches the network, the GPU, a weight shard or the model. It
// is static content plus filesystem existence, like the manifest suite beside
// it.

// MARK: - identity

@Test("the fixture names this track and its runner")
func fixtureNamesTheTrack() throws {
    let object = try trackContractObject()
    #expect(object["schema_version"] as? Int == 1)
    #expect(object["track_id"] as? String == "muse-glimmer-30b-mlx-v1")
    #expect(object["benchmark_name"] as? String == "mlxfast-muse-glimmer-30b")
    #expect(object["runner_name"] as? String == "muse_glimmer")
    #expect(object["allowed_modes"] as? [String] == ["serial", "dflash"])
    #expect(object["scored_batch_size"] as? Int == 1)
    #expect(object["decode_speedup_floor"] as? Double == 0.95)
    #expect(object["prefill_speedup_floor"] as? Double == 0.95)
    #expect(object["score_decode_weight"] as? Double == 0.75)
    #expect(object["score_prefill_weight"] as? Double == 0.25)
}

@Test("the track serves a paged cache only")
func fixtureDeclaresPagedCache() throws {
    // The model's own engine build refuses a contiguous cache, so the declared
    // backend is the single backend this track can serve.
    let object = try trackContractObject()
    #expect(object["kv_backend"] as? String == "paged")
    let note = try #require(object["kv_backend_note"] as? String)
    #expect(note.contains("requiresPagedKV"))
}

// MARK: - target pin

@Test("the target is the pinned NVFP4 checkpoint")
func targetIsThePinnedCheckpoint() throws {
    let object = try trackContractObject()
    let target = try #require(object["target"] as? [String: Any])
    #expect(target["upstream_model_id"] as? String == "nvidia/Muse-Glimmer-30B-NVFP4")
    #expect(target["upstream_revision"] as? String == "47818374517751c48c55cde2621594926b1888b6")
    #expect(target["format"] as? String == "safetensors-modelopt")
    #expect(target["architecture"] as? String == "MuseGlimmerForConditionalGeneration")
    #expect(target["model_type"] as? String == "muse_glimmer")
    #expect(target["text_model_type"] as? String == "muse_glimmer_text")
    #expect(target["upstream_base_model"] as? String == "nvidia/Muse-Glimmer-30B-NVFP4")
    #expect(target["hadamard"] is NSNull)
    #expect(isHex40(target["upstream_revision"]), "the revision is a 40-hex commit id")
}

@Test("the compiled golden pin is the fixture's target")
func compiledGoldenPinIsThePinnedTarget() throws {
    // Every golden's `model_provenance` is validated against these two, so a
    // sentinel here refuses every capture recorded on the box while a wrong
    // checkpoint is accepted. They must BE the fixture's pin.
    let object = try trackContractObject()
    let target = try #require(object["target"] as? [String: Any])
    #expect(
        MLXFastConstants.referenceModelRepository == target["upstream_model_id"] as? String)
    #expect(
        MLXFastConstants.referenceModelRevision == target["upstream_revision"] as? String)
    #expect(!ReferenceModelPin.pinned.isPending)
}

@Test("a commit pin is 40 hex, and a 64-hex test does not accept one")
func commitPinsAre40Hex() throws {
    // The revision is a git commit, not a file digest. Both helpers exist so a
    // 40-hex pin can never be checked with the 64-hex one again -- which is
    // what the first draft of this suite did, and it failed the suite.
    let object = try trackContractObject()
    let target = try #require(object["target"] as? [String: Any])
    let revision = try #require(target["upstream_revision"] as? String)
    #expect(isHex40(revision))
    #expect(!isHex64(revision), "a 40-hex commit is not a 64-hex digest")
    let commit = try #require(object["baseline_reference_commit"] as? String)
    #expect(isHex40(commit))
    #expect(!isHex64(commit), "a 40-hex commit is not a 64-hex digest")
}

@Test("the target geometry equals the pinned snapshot's own config.json")
func targetGeometryMatchesTheConfigFixture() throws {
    let object = try trackContractObject()
    let target = try #require(object["target"] as? [String: Any])
    let text = try targetTextConfigObject()
    for key in [
        "num_hidden_layers", "hidden_size", "intermediate_size", "vocab_size",
        "max_position_embeddings", "num_attention_heads", "num_key_value_heads", "head_dim",
    ] {
        #expect(
            target[key] as? Int == text[key] as? Int,
            Comment(rawValue: "target.\(key) does not match the config fixture"))
    }
    #expect(target["num_hidden_layers"] as? Int == 52)
    #expect(target["hidden_size"] as? Int == 6_656)
    #expect(target["vocab_size"] as? Int == 202_048)
    #expect(target["num_attention_heads"] as? Int == 32)
    #expect(target["num_key_value_heads"] as? Int == 2)
    #expect(target["head_dim"] as? Int == 128)
    #expect(target["sliding_window"] as? Int == 2_048)
}

@Test("the hybrid attention schedule is 13 full layers every fourth index")
func attentionScheduleIsEveryFourthLayer() throws {
    let object = try trackContractObject()
    let target = try #require(object["target"] as? [String: Any])
    let counts = try #require(target["layer_type_counts"] as? [String: Any])
    #expect(counts["sliding_attention"] as? Int == 39)
    #expect(counts["full_attention"] as? Int == 13)
    let indices = try #require(target["full_attention_layer_indices"] as? [Int])
    #expect(indices.count == 13)
    #expect(indices.allSatisfy { $0 % 4 == 3 })
    let layers = try #require(target["num_hidden_layers"] as? Int)
    #expect((counts["sliding_attention"] as? Int ?? 0) + indices.count == layers)
    // The same schedule is in the config fixture's own `layer_types`.
    let text = try targetTextConfigObject()
    let layerTypes = try #require(text["layer_types"] as? [String])
    #expect(layerTypes.count == layers)
    #expect(layerTypes.filter { $0 == "full_attention" }.count == 13)
}

@Test("the config fixture is the pinned snapshot's own config.json")
func configFixtureIsPinned() throws {
    let object = try trackContractObject()
    let target = try #require(object["target"] as? [String: Any])
    let pin = try #require(target["config_fixture"] as? [String: Any])
    #expect(pin["path"] as? String == "fixtures/muse_glimmer_30b_nvfp4_config.json")
    #expect(pin["sha256"] as? String == targetConfigSHA256)
    #expect(pin["bytes"] as? Int == targetConfigByteCount)
    let data = try Data(contentsOf: targetConfigFixtureURL)
    #expect(data.count == targetConfigByteCount)
    #expect(sha256Hex(data) == targetConfigSHA256)
}

@Test("the target is text only, and the vision tower is not loaded")
func targetIsTextOnly() throws {
    let object = try trackContractObject()
    let target = try #require(object["target"] as? [String: Any])
    let components = try #require(target["components"] as? [String: Any])
    #expect(components["text"] as? Bool == true)
    #expect(components["mtp"] as? Bool == false)
    let vision = try #require(target["vision_tower"] as? [String: Any])
    #expect(vision["loaded"] as? Bool == false)
    #expect(vision["model_type"] as? String == "muse_glimmer_vision")
    // The quantization is the shipped modelopt mix, frozen as shipped.
    let quantization = try #require(target["quantization"] as? [String: Any])
    #expect(quantization["mode"] as? String == "nvfp4-mixed")
    #expect(quantization["bits"] as? Int == 4)
    #expect(quantization["group_size"] as? Int == 16)
    #expect(quantization["producer"] as? String == "modelopt")
    let groups = try #require(quantization["groups"] as? [String: Any])
    #expect((groups["group_0"] as? [String: Any])?["modules"] as? Int == 180)
    #expect((groups["group_1"] as? [String: Any])?["modules"] as? Int == 213)
}

// MARK: - the drafter

@Test("the drafter is the pinned dflash decoder under the byte cap")
func drafterIsPinnedUnderTheCap() throws {
    let object = try trackContractObject()
    let head = try #require(object["mtp_head"] as? [String: Any])
    #expect(head["decoder"] as? String == "dflash")
    #expect(head["source"] as? String == "pinned")
    #expect(head["upstream_model_id"] as? String == "meta-models/Muse-Glimmer-30B-assistant")
    #expect(head["upstream_revision"] as? String == "c676cce91658f9568aaf264a416211e2782735b7")
    let cap = try #require(head["byte_cap"] as? Int)
    #expect(cap == 2_147_483_648)
    let artifact = try #require(head["artifact"] as? [String: Any])
    let bytes = try #require(artifact["bytes"] as? Int)
    #expect(bytes < cap, "the drafter must fit the declaration cap")
    #expect(isConcretePin(artifact["weights_sha256"]))
    #expect(isHex64(artifact["weights_sha256"]))
    #expect(isHex64(artifact["configuration_sha256"]))
    #expect(artifact["files"] as? [String] == ["config.json", "model.safetensors"])
    #expect(head["block_size"] as? Int == 16)
    #expect(head["target_layer_ids"] as? [Int] == [1, 13, 25, 37, 49])
    #expect(head["permitted_draft_depths"] as? [Int] == [1, 2, 3, 4, 5, 6, 7])
}

@Test("the dflash envelope is single-stream and exact")
func dflashEnvelopeIsExact() throws {
    let object = try trackContractObject()
    let protocolBlock = try #require(object["protocol"] as? [String: Any])
    let envelope = try #require(protocolBlock["dflash_envelope_constants"] as? [String: Any])
    #expect(envelope["max_draft_tokens"] as? Int == 7)
    // The runner declares single-stream regimes only; one is what it serves.
    #expect(envelope["max_speculative_batch"] as? Int == 1)
    // The drafter only proposes: the pinned target decides every emitted token.
    #expect(envelope["verification_mode"] as? String == "exact")
}

// MARK: - not armed

@Test("official scoring is NOT armed and the hidden golden is pending")
func officialScoringIsNotArmed() throws {
    let object = try trackContractObject()
    #expect(trackContractIsArmed(object) == false)
    #expect(object["official_scoring_enabled"] as? Bool == false)
    #expect((object["live_goldens"] as? [Any])?.isEmpty == true)
    #expect((object["timed_prompt_pool"] as? [Any])?.isEmpty == true)
    #expect((object["live_golden_speculative"] as? [String: Any])?.isEmpty == true)
    let hidden = try #require(object["hidden_correctness_golden"] as? [String: Any])
    #expect(hidden["sha256"] as? String == trackGoldenSentinel)
    #expect(hidden["bytes"] as? Int == 0)
    // No baseline pair is stored in the tree: the run times the reference leg
    // on the box instead.
    #expect(object["official_baseline"] == nil)
    #expect(object["baseline_prefill_seconds_per_token"] == nil)
    #expect(object["baseline_decode_seconds_per_token"] == nil)
}

@Test("the baseline reference names an engine commit, and the fork pin a TREE")
func baselineReferenceAndForkPin() throws {
    let object = try trackContractObject()
    let commit = try #require(object["baseline_reference_commit"] as? String)
    #expect(isHex40(commit), "baseline_reference_commit is a 40-hex engine commit")
    // The vendored tree is the local merge of two branches, so no commit names
    // it and the fixture records the TREE. Nothing reads it as a commit.
    #expect(object["mlx_swift_lm_revision"] as? String == "0391eacd84cb00ce4001d70763f2303fa2a01210")
    let note = try #require(object["mlx_swift_lm_revision_note"] as? String)
    #expect(note.contains("TREE"))
}

// MARK: - manifests

@Test("the target manifest pins the checkpoint's files and its header agrees")
func targetManifestIsConsistent() throws {
    let object = try trackContractObject()
    let target = try #require(object["target"] as? [String: Any])
    #expect(target["manifest_path"] as? String == "fixtures/reference_bonsai2_9b_2bit.sha256")
    let records = try targetManifestRecords()
    #expect(records.count == 13)
    let header = try pinnedManifestHeader(at: targetManifestURL)
    #expect(header.records == records.count)
    #expect(header.bytes == records.reduce(0) { $0 + $1.byteCount })
    for record in records {
        #expect(isHex64(record.sha256), Comment(rawValue: record.path))
        #expect(record.byteCount > 0, Comment(rawValue: record.path))
    }
    // LFS entries are pinned at their own object id; the omissions rule keeps
    // the sidecar, the licence and the image out of the list.
    for omitted in [".gitattributes", "LICENSE", "logo.png"] {
        #expect(
            !records.contains { $0.path.hasSuffix(omitted) },
            Comment(rawValue: "\(omitted) must not be pinned"))
    }
    let text = try String(contentsOf: targetManifestURL, encoding: .utf8)
    #expect(text.contains("nvidia/Muse-Glimmer-30B-NVFP4"))
}

@Test("the head slot pins no file, because the drafter is produced")
func headManifestPinsNothing() throws {
    let object = try trackContractObject()
    let head = try #require(object["mtp_head"] as? [String: Any])
    #expect(head["manifest_path"] as? String == "fixtures/reference_bonsai2_9b_mtp_head.sha256")
    let records = try headManifestRecords()
    #expect(records.isEmpty)
    let header = try pinnedManifestHeader(at: headManifestURL)
    #expect(header.records == 0)
    #expect(header.bytes == 0)
}

@Test("the shipped head declaration matches the fixture's decoder and cap")
func shippedDeclarationMatchesTheFixture() throws {
    // The tracked declaration is the tree's own state: decoder `dflash`,
    // switched OFF so the stock tree runs serial, and never above the cap the
    // fixture declares.
    let object = try trackContractObject()
    let head = try #require(object["mtp_head"] as? [String: Any])
    let declaration = try jsonObject(
        at: headDeclarationURL, describedAs: "mtp-head.manifest.json")
    #expect(declaration["source"] as? String == "pinned")
    let spec = try #require(declaration["spec"] as? [String: Any])
    #expect(spec["decoder"] as? String == head["decoder"] as? String)
    #expect(spec["enabled"] as? Bool == false)
    #expect(spec["num_speculative_tokens"] as? Int == 0)
    let declaredCap = try #require(declaration["max_bytes"] as? Int)
    #expect(declaredCap == head["byte_cap"] as? Int)
}

// MARK: - scoring, against the compiled window

@Test("official scoring is three pairs combined by the mean")
func officialScoringIsThreePairsByTheMean() throws {
    // The combine, the pair count and the live control leg are the ruled
    // shape of this track's score: the mean of three per-pair composites, each
    // pair a serial control leg and a candidate leg on the same box.
    let object = try trackContractObject()
    #expect(object["official_pairs"] as? Int == 3)
    #expect(object["official_pair_combine"] as? String == "mean")
    #expect(object["scores_against_live_control_leg"] as? Bool == true)
    #expect(object["scored_batch_size"] as? Int == 1)
}

@Test("scored_exponents is the ruled certify pair, exact field names")
func scoredExponentsAreTheRuledPair() throws {
    let object = try trackContractObject()
    let exponents = try #require(object["scored_exponents"] as? [String: Any])
    // Field names must match the benchmarker's declared struct exactly -- NOT
    // a shorthand "prefill"/"decode" spelling, which would read as absent.
    #expect(exponents["prefill_gain_exponent"] as? Double == 0.25)
    #expect(exponents["decode_gain_exponent"] as? Double == 0.75)
    #expect(
        exponents.count == 2,
        "scored_exponents must carry exactly the certify pair, no extra keys")
}

@Test("the contract and the compiled constants agree on the scored window")
func contractAgreesWithCompiledConstants() throws {
    // The fixture is the authority the benchmarker reads; these constants are
    // what the trusted Swift paths use. A window that moves on one side only
    // is a silent disagreement between the runner and the contract.
    let object = try trackContractObject()
    #expect(object["vocab_size"] as? Int == MLXFastConstants.vocabSize)
    #expect(object["num_hidden_layers"] as? Int == MLXFastConstants.numHiddenLayers)
    #expect(
        object["golden_model_type"] as? String == MLXFastConstants.requiredGoldenModelType)
    #expect(object["seed_tokens"] as? Int == MLXFastConstants.benchmarkDecodeSeedTokens)
    #expect(object["correctness_steps"] as? Int == MLXFastConstants.correctnessSteps)
    #expect(
        object["benchmark_decode_steps"] as? Int == MLXFastConstants.benchmarkDecodeSteps)
    #expect(
        object["local_submit_benchmark_decode_steps"] as? Int
            == MLXFastConstants.localSubmitBenchmarkDecodeSteps)
    #expect(object["score_decode_weight"] as? Double == MLXFastConstants.scoreDecodeWeight)
    #expect(object["score_prefill_weight"] as? Double == MLXFastConstants.scorePrefillWeight)
    #expect(
        object["decode_speedup_floor"] as? Double
            == MLXFastConstants.scoreDecodeSpeedupFloor)
    #expect(
        object["prefill_speedup_floor"] as? Double
            == MLXFastConstants.scorePrefillSpeedupFloor)
}

// MARK: - the packed-family substrate

@Test("the packed-family substrate fixtures are the files they cite")
func packedFamilyFixturesArePinned() throws {
    // The transform tests read these two. They are other repositories' published
    // configs, cited by repository and revision, and they are NOT this track's
    // target.
    let base = try Data(contentsOf: packedFamilyBaseConfigFixtureURL)
    #expect(base.count == packedFamilyBaseConfigByteCount)
    #expect(sha256Hex(base) == packedFamilyBaseConfigSHA256)
    let packing = try Data(contentsOf: packingReferenceConfigFixtureURL)
    #expect(packing.count == packingReferenceConfigByteCount)
    #expect(sha256Hex(packing) == packingReferenceConfigSHA256)
    #expect(packedFamilyBaseRepository == "Qwen/Qwen3.5-9B")
    #expect(packingReferenceRepository == "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit")
}
