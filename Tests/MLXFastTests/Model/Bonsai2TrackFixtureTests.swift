import Foundation
import MLXFastCore
import Testing

// Contract tests for `fixtures/muse_glimmer_30b_mlx_v1_track.json`.
//
// The track is STAMPED, NOT ARMED, AND ITS TARGET IS PENDING. prism-ml has not
// published the Bonsai 2 9B pack or an MTP head for it. So:
//
// - `official_scoring_enabled` is false, the timed pool is empty, `live_goldens`
//   is empty and the hidden oracle carries the sentinel. The goldens
//   are organizer material and are captured on the ranked box.
// - every pin that names the pack's or the head's bytes carries the exact
//   sentinel `BONSAI-MLX-PENDING-ORGANIZER`, and the two reference manifests
//   hold no record;
// - the geometry is the Qwen 3.5 9B backbone's, read off its own published
//   `config.json` (`fixtures/qwen35_9b_base_config.json`), and the compiled
//   constants agree with it.
//
// This suite asserts that state as a COHERENT WHOLE: a real pin beside a
// sentinel, or scoring switched on against a pending target, must red.
//
// This is a laptop-side JSON-shape test only. It does not exercise benchd's
// Rust contract parsers; those live in the bench repository.

@Suite("Bonsai 2 9B track contract fixture")
struct Bonsai2TrackFixtureTests {

    @Test("fixture parses as a JSON object")
    func fixtureParses() throws {
        let object = try bonsai2TrackContractObject()
        #expect(object["schema_version"] as? Int == 1)
    }

    @Test("track_id is pinned to the leaderboard / R2-prefix identity")
    func trackIdIsPinned() throws {
        let object = try bonsai2TrackContractObject()
        #expect(object["track_id"] as? String == "muse-glimmer-30b-mlx-v1")
    }

    @Test("benchmark_name matches the Yukon manifest name")
    func benchmarkNameMatchesManifest() throws {
        let object = try bonsai2TrackContractObject()
        #expect(object["benchmark_name"] as? String == "mlxfast-bonsai2-9b")
    }

    @Test("track_id is substring-clean against every retired MTP name")
    func trackIdIsCleanAgainstRetiredNames() throws {
        let object = try bonsai2TrackContractObject()
        let trackId = try #require(object["track_id"] as? String)
        let retired = [
            "MLXFAST_MTP_",
            "mtp-ranked",
            "measure-mtp-job",
            "mtp-weights",
            "laguna-xs-2.1-mtp",
            "gemma4-31b-it",
        ]
        for name in retired {
            #expect(
                !trackId.contains(name),
                "track_id must not substring-collide with retired name \(name)")
            #expect(
                !name.contains(trackId),
                "retired name \(name) must not substring-collide with track_id")
        }
    }

    /// THE UNARMED STATE IS ONE STATE, so it is asserted in one test. Scoring
    /// off, an empty pool, no live goldens, no per-depth oracles and a sentinel
    /// oracle must travel together. `live_goldens` is a list, and
    /// `live_golden_speculative` maps each depth to one pin per live golden.
    ///
    /// ARMED (the arming commit): scoring on, every live golden has a pinned
    /// pool entry, the per-depth oracle map names only live goldens, and the
    /// hidden oracle is a real digest equal to one pool pin (the warm-up pin).
    @Test("the fixture is coherently unarmed or coherently armed")
    func theFixtureIsCoherentlyUnarmed() throws {
        let object = try bonsai2TrackContractObject()
        if bonsai2TrackIsArmed(object) {
            try assertCoherentlyArmed(object)
            return
        }
        #expect(object["official_scoring_enabled"] as? Bool == false)
        let pool = try #require(object["timed_prompt_pool"] as? [[String: Any]])
        #expect(pool.isEmpty)
        let live = try #require(object["live_goldens"] as? [String])
        #expect(live.isEmpty)
        #expect(object["live_golden"] == nil)
        let oracles = try #require(
            object["live_golden_speculative"] as? [String: [String: [String: Any]]])
        #expect(oracles.isEmpty)
        // ROOT level -- benchd's `hidden_correctness_golden_pin_from_contract`
        // reads this key directly off the contract root.
        let golden = try #require(object["hidden_correctness_golden"] as? [String: Any])
        #expect(golden["sha256"] as? String == bonsai2PendingSentinel)
        #expect(golden["bytes"] as? Int == 0)
        #expect(object["hidden_material"] == nil)
    }

    private func assertCoherentlyArmed(_ object: [String: Any]) throws {
        let pool = try #require(object["timed_prompt_pool"] as? [[String: Any]])
        #expect(!pool.isEmpty)
        for entry in pool {
            #expect(bonsai2IsHex64(entry["sha256"]))
            #expect((entry["bytes"] as? Int ?? 0) > 0)
            #expect((entry["r2_path"] as? String)?.hasSuffix(".golden.json") == true)
        }
        let live = try #require(object["live_goldens"] as? [String])
        #expect(!live.isEmpty)
        let poolNames = pool.compactMap { ($0["r2_path"] as? String)?.split(separator: "/").last }
            .map { String($0.dropLast(".golden.json".count)) }
        for name in live {
            #expect(poolNames.contains(name), Comment(rawValue: "live golden \(name) has a pool pin"))
        }
        #expect(object["live_golden"] == nil)
        let oracles = try #require(
            object["live_golden_speculative"] as? [String: [String: [String: Any]]])
        for (_, perGolden) in oracles {
            for name in perGolden.keys {
                #expect(live.contains(name), Comment(rawValue: "oracle \(name) is a live golden"))
            }
        }
        let golden = try #require(object["hidden_correctness_golden"] as? [String: Any])
        #expect(bonsai2IsHex64(golden["sha256"]))
        #expect((golden["bytes"] as? Int ?? 0) > 0)
        #expect(pool.contains { ($0["sha256"] as? String) == (golden["sha256"] as? String)
            && ($0["bytes"] as? Int) == (golden["bytes"] as? Int) })
        #expect(object["hidden_material"] == nil)
    }

    /// MODE FENCE. This track has ONE speculative arm, the MTP head. `serial`
    /// must be present because the control leg is pinned serial. No DFlash 2
    /// drafter exists for this geometry, so `dflash` is absent and the
    /// contract carries no drafter block. `dspark` must stay absent too.
    @Test("allowed_modes declares serial and mtp only")
    func allowedModesDeclaresSerialAndMTP() throws {
        let object = try bonsai2TrackContractObject()
        let modes = try #require(object["allowed_modes"] as? [String])
        #expect(modes == ["serial", "mtp"])
        #expect(object["dflash_drafter"] == nil)
        let protocolBlock = try #require(object["protocol"] as? [String: Any])
        #expect(protocolBlock["dflash_envelope_constants"] == nil)
    }

    @Test("the MTP head declares draft depths 1 to 7")
    func theHeadDeclaresItsPermittedDraftDepths() throws {
        let object = try bonsai2TrackContractObject()
        let head = try #require(object["mtp_head"] as? [String: Any])
        #expect(head["permitted_draft_depths"] as? [Int] == Array(1 ... 7))
    }

    /// THE COMBINE. Three pairs, and the score is the MEAN of the per-pair
    /// composites (David 2026-10-06, the same rule as the Nemotron track).
    @Test("official scoring is three pairs combined by the mean")
    func officialPairsCombineByTheMean() throws {
        let object = try bonsai2TrackContractObject()
        #expect(object["official_pairs"] as? Int == 3)
        #expect(object["official_pair_combine"] as? String == "mean")
    }

    @Test("kv_backend is pinned explicitly to contiguous")
    func kvBackendIsContiguous() throws {
        let object = try bonsai2TrackContractObject()
        #expect(object["kv_backend"] as? String == "contiguous")
    }

    @Test("scored_batch_size is pinned to the ruled single-stream width")
    func scoredBatchSizeIsOne() throws {
        let object = try bonsai2TrackContractObject()
        #expect(object["scored_batch_size"] as? Int == 1)
    }

    @Test("scored_exponents equals the ruled certify pair, exact field names")
    func scoredExponentsMatchesRuledPair() throws {
        let object = try bonsai2TrackContractObject()
        let exponents = try #require(object["scored_exponents"] as? [String: Any])
        // Field names must match benchd's `DeclaredScoredExponents` struct
        // exactly -- NOT the shorthand "prefill"/"decode" spelling, which
        // `ScoredExponents::certify` would treat as absent.
        let prefill = try #require(exponents["prefill_gain_exponent"] as? Double)
        let decode = try #require(exponents["decode_gain_exponent"] as? Double)
        #expect(prefill == 0.25)
        #expect(decode == 0.75)
        #expect(
            exponents.count == 2,
            "scored_exponents must carry exactly the certify pair, no extra keys")
    }

    @Test("the contract and the compiled constants agree on the scored window")
    func contractAgreesWithCompiledConstants() throws {
        let object = try bonsai2TrackContractObject()
        #expect(object["vocab_size"] as? Int == MLXFastConstants.vocabSize)
        #expect(object["num_hidden_layers"] as? Int == MLXFastConstants.numHiddenLayers)
        #expect(
            object["golden_model_type"] as? String
                == MLXFastConstants.requiredGoldenModelType)
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

    /// PENDING TARGET. The repository and the revision carry the sentinel, and
    /// the compiled constants carry the same sentinel, so a golden recorded
    /// against ANY published checkpoint is refused until the pack is pinned.
    ///
    /// ARMED: every target pin is concrete. The compiled organizer constants
    /// stay the sentinel (the recorder gate in the binary is not re-pinned by
    /// a fixture arming), so the suite still reds if the binary is re-pinned
    /// to a checkpoint without this assertion moving with it.
    @Test("the target pins are coherently pending or coherently pinned")
    func targetPinsArePending() throws {
        let object = try bonsai2TrackContractObject()
        let target = try #require(object["target"] as? [String: Any])
        if bonsai2TrackIsArmed(object) {
            let revision = try #require(target["upstream_revision"] as? String)
            #expect(bonsai2IsConcretePin(target["upstream_model_id"]))
            #expect(revision.count == 40 && revision.allSatisfy(\.isHexDigit))
            #expect(MLXFastConstants.pendingOrganizerSentinel == bonsai2PendingSentinel)
            #expect(MLXFastConstants.referenceModelRepository == bonsai2PendingSentinel)
            #expect(MLXFastConstants.referenceModelRevision == bonsai2PendingSentinel)
            for key in [
                "bos_token_id", "mtp_num_hidden_layers", "raw_checkpoint_tensor_count",
                "raw_checkpoint_shard_count", "raw_checkpoint_total_size_bytes",
                "vision_tower_tensor_count", "config_fixture_path", "tensor_inventory_path",
            ] {
                #expect(
                    bonsai2IsConcretePin(target[key]),
                    Comment(rawValue: "target.\(key) is pinned"))
            }
            let components = try #require(target["components"] as? [String: Any])
            #expect(components["text"] as? Bool == true)
            #expect(components["mtp"] is Bool)
            #expect(components["vision"] is Bool)
            #expect(target["upstream_base_model"] as? String == bonsai2BaseRepository)
            return
        }
        #expect(target["upstream_model_id"] as? String == bonsai2PendingSentinel)
        #expect(target["upstream_revision"] as? String == bonsai2PendingSentinel)
        #expect(MLXFastConstants.pendingOrganizerSentinel == bonsai2PendingSentinel)
        #expect(MLXFastConstants.referenceModelRepository == bonsai2PendingSentinel)
        #expect(MLXFastConstants.referenceModelRevision == bonsai2PendingSentinel)
        // Every fact only the published pack can state waits with it.
        for key in [
            "bos_token_id", "mtp_num_hidden_layers", "raw_checkpoint_tensor_count",
            "raw_checkpoint_shard_count", "raw_checkpoint_total_size_bytes",
            "vision_tower_tensor_count", "config_fixture_path", "tensor_inventory_path",
        ] {
            #expect(
                target[key] as? String == bonsai2PendingSentinel,
                Comment(rawValue: "target.\(key) waits for the pack"))
        }
        let components = try #require(target["components"] as? [String: Any])
        #expect(components["text"] as? Bool == true)
        #expect(components["mtp"] as? String == bonsai2PendingSentinel)
        #expect(components["vision"] as? String == bonsai2PendingSentinel)
        #expect(target["upstream_base_model"] as? String == bonsai2BaseRepository)
    }

    @Test("the geometry source is the base config fixture, byte for byte")
    func geometrySourceIsTheBaseConfig() throws {
        let data = try Data(contentsOf: bonsai2BaseConfigFixtureURL)
        #expect(sha256Hex(data) == bonsai2BaseConfigSHA256)
        #expect(data.count == bonsai2BaseConfigByteCount)

        let object = try bonsai2TrackContractObject()
        let target = try #require(object["target"] as? [String: Any])
        let source = try #require(target["geometry_source"] as? [String: Any])
        #expect(source["upstream_model_id"] as? String == bonsai2BaseRepository)
        #expect(source["upstream_revision"] as? String == bonsai2BaseRevision)
        #expect(source["config_path"] as? String == "fixtures/qwen35_9b_base_config.json")
        #expect(source["sha256"] as? String == bonsai2BaseConfigSHA256)
        #expect(source["bytes"] as? Int == bonsai2BaseConfigByteCount)
    }

    /// The `target` geometry is READ OFF the backbone's own config.json, so
    /// every geometry field is asserted against that file rather than against
    /// a second hand-written copy of the same numbers.
    @Test("target geometry matches the Qwen 3.5 9B config.json")
    func targetGeometryMatchesConfig() throws {
        let object = try bonsai2TrackContractObject()
        let target = try #require(object["target"] as? [String: Any])
        let config = try bonsai2BaseConfigObject()
        let text = try #require(config["text_config"] as? [String: Any])
        let rope = try #require(text["rope_parameters"] as? [String: Any])

        #expect(target["model_type"] as? String == "prism_hadamard_qwen35")
        #expect(target["base_model_type"] as? String == config["model_type"] as? String)
        #expect(target["text_model_type"] as? String == text["model_type"] as? String)

        for key in [
            "num_hidden_layers", "hidden_size", "intermediate_size", "vocab_size",
            "max_position_embeddings", "full_attention_interval",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "linear_num_key_heads", "linear_num_value_heads",
            "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim",
        ] {
            #expect(
                target[key] as? Int == text[key] as? Int,
                Comment(rawValue: "target.\(key) must be the config's own"))
        }
        #expect(target["attention_bias"] as? Bool == text["attention_bias"] as? Bool)
        #expect(target["attn_output_gate"] as? Bool == text["attn_output_gate"] as? Bool)
        #expect(target["tie_word_embeddings"] as? Bool == config["tie_word_embeddings"] as? Bool)
        #expect(target["rope_theta"] as? Int == rope["rope_theta"] as? Int)
        #expect(target["partial_rotary_factor"] as? Double == rope["partial_rotary_factor"] as? Double)
        #expect(target["eos_token_ids"] as? [Int] == [try #require(text["eos_token_id"] as? Int)])
        #expect(target["mamba_ssm_dtype"] as? String == text["mamba_ssm_dtype"] as? String)

        // The compiled constants and the backbone are one set.
        #expect(target["hidden_size"] as? Int == MLXFastConstants.hiddenSize)
        #expect(target["num_hidden_layers"] as? Int == MLXFastConstants.numHiddenLayers)
        #expect(target["vocab_size"] as? Int == MLXFastConstants.vocabSize)
        // THE HEAD IS UNTIED: `lm_head` is a real packed module.
        #expect(target["tie_word_embeddings"] as? Bool == false)
    }

    @Test("the declared layer schedule is the config's own, and its summaries are derived")
    func layerScheduleSummariesAreDerived() throws {
        let object = try bonsai2TrackContractObject()
        let target = try #require(object["target"] as? [String: Any])
        let config = try bonsai2BaseConfigObject()
        let text = try #require(config["text_config"] as? [String: Any])
        let layerTypes = try #require(text["layer_types"] as? [String])
        let interval = try #require(text["full_attention_interval"] as? Int)

        #expect(layerTypes.count == target["num_hidden_layers"] as? Int)
        let counts = try #require(target["layer_type_counts"] as? [String: Int])
        for kind in ["full_attention", "linear_attention"] {
            #expect(
                counts[kind] == layerTypes.filter { $0 == kind }.count,
                Comment(rawValue: "layer_type_counts.\(kind)"))
        }
        #expect(counts.values.reduce(0, +) == layerTypes.count)

        let indices = try #require(target["full_attention_layer_indices"] as? [Int])
        #expect(indices == layerTypes.indices.filter { layerTypes[$0] == "full_attention" })
        // Every fourth layer, and nothing else.
        #expect(indices == (0 ..< layerTypes.count).filter { ($0 + 1) % interval == 0 })
    }

    /// The packing reference is the 27B pack's own config.json, verbatim, and
    /// the packing fields the contract declares are that file's.
    @Test("the packing reference is the 27B pack config, and its packing fields agree")
    func packingReferenceIsThe27BPackConfig() throws {
        let data = try Data(contentsOf: bonsai2PackingConfigFixtureURL)
        #expect(sha256Hex(data) == bonsai2PackingConfigSHA256)
        #expect(data.count == bonsai2PackingConfigByteCount)

        let object = try bonsai2TrackContractObject()
        let target = try #require(object["target"] as? [String: Any])
        let reference = try #require(target["packing_reference"] as? [String: Any])
        #expect(reference["track_id"] as? String == "bonsai2-27b-mlx-v1")
        #expect(reference["upstream_model_id"] as? String == bonsai2PackingRepository)
        #expect(reference["upstream_revision"] as? String == bonsai2PackingRevision)
        #expect(reference["config_path"] as? String == "fixtures/bonsai2_27b_config.json")
        #expect(reference["sha256"] as? String == bonsai2PackingConfigSHA256)
        #expect(reference["bytes"] as? Int == bonsai2PackingConfigByteCount)

        let packing = try bonsai2PackingConfigObject()
        #expect(target["model_type"] as? String == packing["model_type"] as? String)
        #expect(target["base_model_type"] as? String == packing["base_model_type"] as? String)
        #expect(target["schema_version"] as? Int == packing["schema_version"] as? Int)
        #expect(target["gdn_activation_layout"] as? String == packing["gdn_activation_layout"] as? String)
        #expect(target["tensor_namespace"] as? String == packing["tensor_namespace"] as? String)
        let quantization = try #require(target["quantization"] as? [String: Any])
        let published = try #require(packing["quantization"] as? [String: Any])
        #expect(quantization["mode"] as? String == published["mode"] as? String)
        #expect(quantization["bits"] as? Int == published["bits"] as? Int)
        #expect(quantization["group_size"] as? Int == published["group_size"] as? Int)
        let hadamard = try #require(target["hadamard"] as? [String: Any])
        #expect(hadamard["manifest_path_in_checkpoint"] as? String == packing["hadamard_config"] as? String)
        let modules = try #require(packing["modules"] as? [[String: Any]])
        #expect(modules.allSatisfy { $0["block"] as? Int == hadamard["block_size"] as? Int })
    }

    /// The synthetic 9B pack config the transform tests use carries one module
    /// per packed projection of the 9B schedule, and exactly one embedding.
    @Test("the synthetic 9B pack config agrees with the contract's derived counts")
    func syntheticPackConfigAgreesWithTheContract() throws {
        let object = try bonsai2TrackContractObject()
        let target = try #require(object["target"] as? [String: Any])
        let hadamard = try #require(target["hadamard"] as? [String: Any])
        let config = try bonsai2SyntheticPackConfigObject()
        let modules = try #require(config["modules"] as? [[String: Any]])
        #expect(modules.count == hadamard["packed_module_count"] as? Int)
        #expect(Set(modules.compactMap { $0["path"] as? String }).count == modules.count)
        let embeddings = modules.filter { $0["embedding"] as? Bool == true }
        #expect(embeddings.map { $0["path"] as? String } == [hadamard["embedding_module"] as? String])
        let text = try #require(config["text_config"] as? [String: Any])
        #expect(text["hidden_size"] as? Int == MLXFastConstants.hiddenSize)
        #expect(text["num_hidden_layers"] as? Int == MLXFastConstants.numHiddenLayers)
    }

    /// The packing is the Ternary Bonsai 2 27B pack's, assumed until the 9B
    /// pack is published: uniform affine 2-bit, group 128, block-1024 signed
    /// Hadamard. The derived counts follow from the geometry: one packed
    /// module each for the embedding and the head, three for each MLP, three
    /// for each linear-attention mixer and four for each full-attention mixer.
    @Test("the assumed packing is uniform 2-bit and its counts follow from the geometry")
    func assumedPackingFollowsFromTheGeometry() throws {
        let object = try bonsai2TrackContractObject()
        let target = try #require(object["target"] as? [String: Any])
        let quantization = try #require(target["quantization"] as? [String: Any])
        #expect(quantization["mode"] as? String == "affine")
        #expect(quantization["bits"] as? Int == 2)
        #expect(quantization["group_size"] as? Int == 128)
        #expect(quantization["mixed_precision"] as? Bool == false)

        let hadamard = try #require(target["hadamard"] as? [String: Any])
        #expect(hadamard["manifest_path_in_checkpoint"] as? String == "hadamard.json")
        #expect(hadamard["block_size"] as? Int == 1024)
        #expect(hadamard["sign_dtype"] as? String == "float32")
        #expect(hadamard["arithmetic_dtype"] as? String == "float32")
        #expect(hadamard["embedding_module"] as? String == "model.embed_tokens")

        let counts = try #require(target["layer_type_counts"] as? [String: Int])
        let linear = try #require(counts["linear_attention"])
        let full = try #require(counts["full_attention"])
        let packed = 2 + linear * (3 + 3) + full * (3 + 4)
        #expect(hadamard["packed_module_count"] as? Int == packed)
        #expect(hadamard["sign_vector_count"] as? Int == packed)
        // Nine top-level tensors and 32 per layer, of either kind.
        let layers = try #require(target["num_hidden_layers"] as? Int)
        #expect(target["language_model_tensor_count"] as? Int == 9 + 32 * layers)
        #expect(target["lm_head_tensor_count"] as? Int == 4)
        #expect(target["vision_tower_loaded"] as? Bool == false)
    }

    /// THE HEAD IS PENDING. Its pins carry the sentinel. What the geometry
    /// fixes is asserted: one `qwen3_5_mtp` layer, full attention, sharing the
    /// target's embedding and output head.
    ///
    /// ARMED SERIAL-ONLY: a track armed with no head (`target.components.mtp`
    /// false) declares the head absent: source/ids/prefix/quantization "none"
    /// and zero tensors. An armed track WITH a head pins every field.
    @Test("mtp_head is a pending, absent or pinned qwen3_5_mtp layer for the 9B geometry")
    func mtpHeadIsPending() throws {
        let object = try bonsai2TrackContractObject()
        let head = try #require(object["mtp_head"] as? [String: Any])
        let config = try bonsai2BaseConfigObject()
        let text = try #require(config["text_config"] as? [String: Any])
        #expect(head["role"] as? String == "spec_decode_mtp_head")
        let pinKeys = [
            "source", "upstream_model_id", "upstream_revision", "tensor_prefix",
            "tensor_count", "tensor_bytes", "quantization",
        ]
        let components = (object["target"] as? [String: Any])?["components"] as? [String: Any]
        if bonsai2TrackIsArmed(object), components?["mtp"] as? Bool == false {
            for key in ["source", "upstream_model_id", "upstream_revision", "tensor_prefix", "quantization"] {
                #expect(head[key] as? String == "none", Comment(rawValue: "mtp_head.\(key) is absent"))
            }
            #expect(head["tensor_count"] as? Int == 0)
            #expect(head["tensor_bytes"] as? Int == 0)
        } else if bonsai2TrackIsArmed(object) {
            for key in pinKeys {
                #expect(bonsai2IsConcretePin(head[key]), Comment(rawValue: "mtp_head.\(key) is pinned"))
            }
        } else {
            for key in pinKeys {
                #expect(
                    head[key] as? String == bonsai2PendingSentinel,
                    Comment(rawValue: "mtp_head.\(key) waits for the head"))
            }
        }
        #expect(head["model_type"] as? String == "qwen3_5_mtp")
        #expect(head["num_hidden_layers"] as? Int == text["mtp_num_hidden_layers"] as? Int)
        #expect(head["num_hidden_layers"] as? Int == 1)
        #expect(head["layer_types"] as? [String] == ["full_attention"])
        #expect(head["use_dedicated_embeddings"] as? Bool
            == text["mtp_use_dedicated_embeddings"] as? Bool)
        #expect(head["shares_target_embed_tokens"] as? Bool == true)
        #expect(head["shares_target_lm_head"] as? Bool == true)
        #expect(head["manifest_path"] as? String == "fixtures/reference_bonsai2_9b_mtp_head.sha256")
        #expect(head["declaration_path"] as? String == "mtp-head.manifest.json")
    }

    @Test("permitted draft depths fit the declared MTP envelope")
    func permittedDraftDepthsFitTheEnvelope() throws {
        let object = try bonsai2TrackContractObject()
        let head = try #require(object["mtp_head"] as? [String: Any])
        let depths = try #require(head["permitted_draft_depths"] as? [Int])
        let protocolBlock = try #require(object["protocol"] as? [String: Any])
        let envelope = try #require(
            protocolBlock["mtp_envelope_constants_from_darkbloom"] as? [String: Any])
        let maxDraft = try #require(envelope["max_draft_tokens"] as? Int)

        #expect(!depths.isEmpty)
        #expect(depths == Array(1 ... maxDraft))
        #expect(envelope["max_speculative_batch"] as? Int == object["scored_batch_size"] as? Int)
    }

    /// Both manifests are PENDING: they name the sentinel and hold no record,
    /// and their header pins are summed from the (zero) records.
    @Test("both reference manifests are pending and hold no record")
    func referenceManifestsArePending() throws {
        for url in [bonsai2ReferenceManifestURL, bonsai2HeadManifestURL] {
            let records = try bonsai2ManifestRecords(at: url)
            let pins = try bonsai2ManifestHeaderPins(at: url)
            #expect(records.isEmpty, Comment(rawValue: url.lastPathComponent))
            #expect(pins.records == 0, Comment(rawValue: url.lastPathComponent))
            #expect(pins.bytes == 0, Comment(rawValue: url.lastPathComponent))
            let text = try String(contentsOf: url, encoding: .utf8)
            #expect(
                text.contains("# Revision: \(bonsai2PendingSentinel)\n"),
                Comment(rawValue: url.lastPathComponent))
        }
        let object = try bonsai2TrackContractObject()
        let target = try #require(object["target"] as? [String: Any])
        #expect(target["manifest_path"] as? String == "fixtures/reference_bonsai2_9b_2bit.sha256")
    }

    /// The shipped declaration is the stock tree: the MTP decoder, switched
    /// off, so the stock submission runs serial.
    @Test("the shipped head declaration is pinned, MTP and serial")
    func theShippedDeclarationIsSerial() throws {
        let declaration = try #require(
            try JSONSerialization.jsonObject(
                with: Data(contentsOf: bonsai2HeadDeclarationURL)) as? [String: Any])
        #expect(declaration["source"] as? String == "pinned")
        let spec = try #require(declaration["spec"] as? [String: Any])
        #expect(spec["decoder"] as? String == "mtp")
        #expect(spec["enabled"] as? Bool == false)
        #expect(spec["num_speculative_tokens"] as? Int == 0)
        #expect(declaration["max_bytes"] as? Int == 2_147_483_648)
    }

    @Test("scoring_semantics records the ruled composite formula")
    func scoringSemanticsRecordsRuledFormula() throws {
        let object = try bonsai2TrackContractObject()
        let semantics = try #require(object["scoring_semantics"] as? [String: Any])
        let formula = try #require(semantics["formula"] as? String)
        #expect(formula.contains("0.25"))
        #expect(formula.contains("0.75"))
        #expect(formula.contains("prefill_gain"))
        #expect(formula.contains("decode_gain"))
    }

    /// The license is read off the published pack, so it waits with it, and
    /// no repository is verified yet.
    @Test("the license block is pending (or pinned when armed) and verifies no repository yet")
    func licenseIsPending() throws {
        let object = try bonsai2TrackContractObject()
        let license = try #require(object["license"] as? [String: Any])
        if bonsai2TrackIsArmed(object) {
            #expect(bonsai2IsConcretePin(license["spdx"]))
            #expect((license["url"] as? String)?.hasPrefix("https://") == true)
            #expect(license["verified_repositories"] is [String])
            return
        }
        #expect(license["spdx"] as? String == bonsai2PendingSentinel)
        #expect(license["url"] as? String == bonsai2PendingSentinel)
        #expect((license["verified_repositories"] as? [String])?.isEmpty == true)
    }
}
