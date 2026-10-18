# LinuxONE experiment records

Numeric records from the team's 2026-10-17 s390x execution. The
[research deck](../Revolution_IBM_Z_Research_Deck.pptx) is the submission narrative;
these files preserve one recorded run without changing its measurements.

| File | Contents |
| --- | --- |
| [results.json](results.json) | Platform, security checks, parity, latency, Granite and Telum summaries |
| [telum_cpu.json](telum_cpu.json) | TensorFlow CPU benchmark |
| [telum_nnpa.json](telum_nnpa.json) | TensorFlow NNPA benchmark |

This snapshot records 11 PASS results, a maximum secure/clear difference of
0.0000155198, 1001.55 ms median clear inference, 2414.4 ms guest request time,
and 2.51× batch-eight Telum speedup. Parity uses one public NIH sample; the batch
benchmark repeats that sample. It also records three Granite explanations, one withheld.

The deck summarizes multiple reported timings with rounded values and ranges. Its
approximately 23 ms security figure is a reported stage-overhead figure, not total
request time. In this snapshot, key release, decrypt and sign total 21.4 ms;
measurement and process/model startup are separate. Raw values are retained as recorded.

Notebook execution generates new `crai_data/results.json` records. The source-file hash
is recomputed and approved in section 2, including after source changes.
