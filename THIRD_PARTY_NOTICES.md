# Third-Party Notices

This repository's own code is licensed per the top-level `LICENSE` file. The
challenge additionally depends on third-party model artifacts and Swift
packages credited below. Nothing in this file grants rights to the mlxfast
harness itself, and no model weights are distributed inside this repository —
the setup script (`setup.sh`) downloads the pinned artifacts
from their upstream repositories and verifies them against checked-in SHA256
manifests.

## Bonsai 2 models

The target model is the Bonsai 2 9B MLX pack of `Qwen/Qwen3.5-9B`. prism-ml has
not published it, so its license, its pin and its attribution are pending. The
MTP head is pending too. This file names the pack, the head and their licenses
when they are pinned.

Artifacts referenced by this repository:

| Artifact | Upstream | License |
|---|---|---|
| `Qwen/Qwen3.5-9B` @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a` (the backbone; its `config.json` is checked in as `fixtures/qwen35_9b_base_config.json`) | Qwen | Apache-2.0 |
| `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` @ `3f926b415992eaa2ae9dd7b573706494d6bbf787` (the packing reference; its `config.json` is checked in as `fixtures/bonsai2_27b_config.json`) | Prism ML | Apache-2.0 |
| The Bonsai 2 9B pack (the track's target) | Prism ML | Pending |
| The MTP head | Pending | Pending |

Attribution statement: "Qwen3.5-9B © Qwen, licensed Apache-2.0. Ternary Bonsai
2 27B configuration by Prism ML, licensed Apache-2.0."

Compliance notes for anyone redistributing these models or derivative
weights obtained through this harness (for example a transformed `weights/`
tree):

- Review and satisfy the Apache-2.0 terms published on the model card before
  redistribution. The license URL the contract fixture records is the one on
  the pinned revision.
- Mark modified files as changed. The harness's `transform` output is a
  repacked derivative of the pinned checkpoint, and its provenance is recorded
  in the emitted `config.json` and hash manifests.
- Retain the attribution statement above.
- The Apache License 2.0 appendix below covers these models and the Swift
  package dependencies listed in the next section.

## Swift package dependencies

The harness links the following packages via SwiftPM; each license text is
available in the corresponding package repository and, after a build, under
`.build/checkouts/<package>/LICENSE*`. The first two are local, not fetched
from the network: see "Vendored trees" below.

| Package | License |
|---|---|
| `Layr-Labs/mlx-swift` (fork of `ml-explore/mlx-swift`, incl. the MLX C++ core and Metal kernels) | MIT (© 2023 ml-explore / Apple) |
| `Layr-Labs/mlx-swift-lm` (fork of `ml-explore/mlx-swift-examples` LM libraries) | MIT (© 2024 ml-explore / Apple) |
| `huggingface/swift-transformers`, `huggingface/swift-huggingface`, `huggingface/swift-jinja` | Apache-2.0 |
| Apple `swift-*` packages (`argument-parser`, `algorithms`, `asn1`, `async-algorithms`, `atomics`, `certificates`, `collections`, `configuration`, `crypto`, `distributed-tracing`, `http-structured-headers`, `http-types`, `log`, `metrics`, `nio` family, `numerics`, `service-context`, `syntax`, `system`) | Apache-2.0 |
| `swift-server/async-http-client`, `swift-server/swift-service-lifecycle`, `hummingbird-project/hummingbird` | Apache-2.0 |
| `ibireme/yyjson` | MIT |
| `mattt/EventSource` | MIT |

## Vendored trees

`Vendor/mlx-swift` and `Vendor/mlx-swift-lm` are COPIES, not submodules and not
network dependencies. They are copies for one reason: parts of each are the
track's editable surface, and `benchmark.json` cannot make a submodule
editable. An editable path names bytes in this tree; a gitlink names a commit.
The `mlx-swift` copy includes the two nested submodules of that repository as
plain files.

| Tree | Upstream | Commit |
|---|---|---|
| `Vendor/mlx-swift` | `Layr-Labs/mlx-swift` (fork of `ml-explore/mlx-swift`) | `b1c94d29c5ddff16838fdc25b83fbe8fde48b3e8` |
| `Vendor/mlx-swift/Source/Cmlx/mlx` | `Layr-Labs/mlx` (fork of `ml-explore/mlx`), MLX 0.32.5 | `72dffa2dbbdce95601fd84f5e8b6d7b4ec97a11b` |
| `Vendor/mlx-swift/Source/Cmlx/mlx-c` | `Layr-Labs/mlx-c` (fork of `ml-explore/mlx-c`) | `2ac897e34495c924f9cecfa93fffaf4316bbcb91` |
| `Vendor/mlx-swift-lm` | `Layr-Labs/mlx-swift-lm` (fork of `ml-explore/mlx-swift-examples`), the local merge of `feat/muse-glimmer-speed` `62cfabd5c67c6042edcfd812698bd34e4b5df7d7` and `feat/muse-glimmer-rowexact-m5-d5` `443789360ddbf804d55d063ca9cda4ea2d3013a9`; engine-local changes on top | tree `0391eacd84cb00ce4001d70763f2303fa2a01210` (merge of those two commits; not pushed upstream) |

`b1c94d29c5ddff16838fdc25b83fbe8fde48b3e8` with its nested `72dffa2dbbdce95601fd84f5e8b6d7b4ec97a11b` is the core pair this engine is built
and measured on: it is the pair the serving provider (`Layr-Labs/d-inference`)
builds the same engine fork against. `b1c94d29c5ddff16838fdc25b83fbe8fde48b3e8` descends from `6d6796d7`, the
base the earlier pin sat on, and it carries `Source/MLXNN/Hadamard.swift`, which
holds the signed block-Hadamard transform and the packed
`HadamardQuantizedLinear` and `HadamardQuantizedEmbedding` layers the Bonsai 2
pack loads into. The nested `mlx` reports MLX 0.32.5. The engine fork's
`Package.swift` uses a sibling `../mlx-swift` checkout when one exists, so the
copy here IS the core the fork compiles against. Keep the two in step: one
runner builds against one core.

`Vendor/mlx-swift/Source/Cmlx` carries further third-party code that upstream
vendors in the same way. `Vendor/mlx-swift/Source/Cmlx/vendor-README.md` names
the sources: `nlohmann/json` v3.11.3 (MIT), Apple `metal-cpp` for macOS 15 /
iOS 18 (Apache-2.0), and `fmtlib/fmt` tag 12.1.0 (MIT).

The vendored `Vendor/mlx-swift-lm` tree is a fresh cut at that merge and is
byte-identical to it, with no commits of this repository on top. The changes the
earlier base carried -- the merge of fork `main` `0607011`, two build fixes, and
the Bonsai 2 support files taken from fork `main`
`fd0eaac7b5ec5c199f1595d29be011f66343f1db` (pull requests 154 and 155) -- were
discarded by the cut and are re-applied by later commits in this repository.
Its Bonsai 2 and Qwen 3.5 model files are the editable surface; the rest of the
fork is trusted.

### Engine-local changes to the vendored SDK

One change sits on top of the cut, in the MTP path. It is **record-only**: it
stores numbers the round driver already computes at its finalize boundary and
changes neither the tokens a window emits, their order, the work inside a round,
nor any default. Retention is a ring of 8192 records: a full ring overwrites its
oldest entry in constant time and counts the evictions, so a reader can tell a
full log (nothing lost) from a truncated one.

| File | Added | Removed | What |
|---|---|---|---|
| `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/MTP/MTPContractsV2.swift` | 66 | 0 | one per-round record type, the bounded log that holds it, and one counter for the engine steps that seeded a row |
| `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/MTP/CBv2MTPRoundDriver.swift` | 8 | 1 | `recordRound` accepts the record and appends it to the log under the lock it already takes; `recordSeedSteps` also counts the STEP |
| `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/MTP/EngineLoopV2+MTPFinalize.swift` | 5 | 1 | the finalize boundary passes `(requestID, drafted, accepted, committed)` |

Each changed block carries the comment line
`// engine-local: per-round audit record (benchmark free-run audit)`. The runner
layer reads the records through `Vendor/mlx-swift-lm/Libraries/MLXRunners/RunnerRoundJournal.swift`.

The signed block-Hadamard layers in `Vendor/mlx-swift/Source/MLXNN/Hadamard.swift`
are adapted from Prism's Swift work at
`PrismML-Eng/mlx-swift` `6d3a84de28225d1f5bc0a56f5c781596997242f9`, MIT.

## Tooling

- `macmon` (GPU telemetry used by the local and ranked thermal gate) is
  <https://github.com/vladkens/macmon>, MIT. `setup.sh` installs it from a
  pinned release tarball, verified against a pinned sha256. It is not part of
  this repository.

---

## Ported reference implementations

The model sources in the vendored engine fork are Swift ports of permissively
licensed reference implementations. No source under a copyleft licence was read
or ported.

| Ported from | Upstream | Licence |
|---|---|---|
| `mlx_lm/models/qwen3_5.py` — the Qwen 3.5 hybrid tower, ported in `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35*.swift` | ml-explore/mlx-lm | MIT, Copyright (c) 2023 Apple Inc. |
| `mlx_lm/models/qwen4_exp.py` @ `c961f839` (ml-explore/mlx-lm pull request 1788) — the Qwen 3.8 Flash-Next text tower, ported in `Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Qwen4Exp*.swift` and `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/Qwen4ExpCaches.swift`. Those files serve another track and are trusted here. | ml-explore/mlx-lm | MIT, Copyright (c) 2023 Apple Inc. |
| `vllm/models/qwen4_exp/nvidia/mtp.py` @ `2a4cd640` (vllm-project/vllm pull request 53896) — the wiring of the native multi-token-prediction head | vllm-project/vllm | Apache-2.0 |

The Apache License 2.0 text in the appendix below covers the third entry and
the pinned models above.

## Appendix: Apache License, Version 2.0

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS
