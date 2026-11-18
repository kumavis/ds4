## Every run

| tag | tok/s | output_class | ref_match | ref_line | token digest | rounds/accepted | decode+batch misses | MemAvailable min (MiB) | GTT max (MiB) | NVMe max (°C) |
|---|---:|---|---|---|---|---|---|---:|---:|---:|
| d15-q8-smoke-on-8 | 8.9 | fp-order | False | None | a8b620760253/7 | None/None | 5642+0 | 13938 | 110433 | 75 |
| d15-q8-smoke-on-64 | 9.21 | fp-order | False | None | c2bdcaff1e51/63 | None/None | 6924+0 | 13842 | 110541 | 75 |
| d15-q8-mx-code-off-r1 | 9.15 | plain | True | fork | 8290541d324b/511 | None/None | 17462+0 | 13871 | 110545 | 76 |
| d15-q8-mx-code-on-r1 | 9.1 | fp-order | False | None | 3c26a9d373a1/511 | None/None | 17685+0 | 13795 | 110493 | 77 |
| d15-q8-mx-prose-off-r1 | 9.8 | plain | True | fork | 88ed0e11adfd/511 | None/None | 15149+0 | 13668 | 110743 | 72 |
| d15-q8-mx-prose-on-r1 | 9.56 | fp-order | False | None | 8520725fd8f9/511 | None/None | 16075+0 | 13865 | 110529 | 72 |
| d15-q8-mx-hash-off-r1 | 9.23 | plain | True | fork | 4c37f5fee667/410 | None/None | 13668+0 | 13886 | 110529 | 73 |
| d15-q8-mx-hash-on-r1 | 9.29 | fp-order | False | None | c669c86b6e97/413 | None/None | 13493+0 | 13560 | 110761 | 73 |
| d15-q8-mx-long-off-r1 | 8.78 | plain | True | fork | c27a3b087fb5/255 | None/None | 6290+0 | 13439 | 110527 | 79 |
| d15-q8-mx-long-on-r1 | 8.82 | fp-order | False | None | 4bdf471a39c2/255 | None/None | 6240+0 | 13504 | 110455 | 79 |
| d15-q8-dsp-code-off-r1 | 11.87 | dspark-fast | None | None | None/None | 172/340 | 7575+13047 | 13287 | 110869 | 79 |
| d15-q8-dsp-code-on-r1 | 12.31 | fp-order | None | None | None/None | 133/379 | 5541+15482 | 13324 | 110829 | 79 |
| d15-q8-dsp-json-off-r1 | 11.99 | dspark-fast | None | None | None/None | 207/305 | 1228+10283 | 12890 | 110833 | 84 |
| d15-q8-dsp-json-on-r1 | 12.18 | fp-order | None | None | None/None | 209/303 | 990+9834 | 13033 | 110689 | 83 |
| d15-q8-q2-code-off-r1 | 12.14 | plain | True | fork | f0a2cd53a247/511 | None/None | 9818+0 | 13971 | 110499 | 72 |
| d15-q8-q2-code-on-r1 | 12.17 | fp-order | False | None | e4e223807e97/511 | None/None | 9736+0 | 13699 | 110775 | 75 |
| d15-q8-mx-code-off-r2 | 9.15 | plain | True | fork | 8290541d324b/511 | None/None | 17467+0 | 13907 | 110511 | 74 |
| d15-q8-mx-code-on-r2 | 9.09 | fp-order | False | None | 3c26a9d373a1/511 | None/None | 17602+0 | 13560 | 110743 | 75 |
| d15-q8-mx-prose-off-r2 | 9.76 | plain | True | fork | 88ed0e11adfd/511 | None/None | 15185+0 | 13923 | 110511 | 75 |
| d15-q8-mx-prose-on-r2 | 9.48 | fp-order | False | None | 8520725fd8f9/511 | None/None | 16095+0 | 13930 | 110385 | 60 |
| d15-q8-mx-hash-off-r2 | 9.24 | plain | True | fork | 4c37f5fee667/410 | None/None | 13679+0 | 13880 | 110457 | 74 |
| d15-q8-mx-hash-on-r2 | 9.3 | fp-order | False | None | c669c86b6e97/413 | None/None | 13516+0 | 13658 | 110599 | 75 |
| d15-q8-mx-long-off-r2 | 8.76 | plain | True | fork | c27a3b087fb5/255 | None/None | 6315+0 | 13528 | 110455 | 78 |
| d15-q8-mx-long-on-r2 | 8.82 | fp-order | False | None | 4bdf471a39c2/255 | None/None | 6232+0 | 13423 | 110543 | 78 |
| d15-q8-dsp-code-off-r2 | 11.94 | dspark-fast | None | None | None/None | 172/340 | 7586+13064 | 13436 | 110743 | 75 |
| d15-q8-dsp-code-on-r2 | 12.3 | fp-order | None | None | None/None | 133/379 | 5542+15511 | 13496 | 110671 | 75 |
| d15-q8-dsp-json-off-r2 | 11.86 | dspark-fast | None | None | None/None | 207/305 | 1230+10318 | 13012 | 110707 | 80 |
| d15-q8-dsp-json-on-r2 | 12.08 | fp-order | None | None | None/None | 209/303 | 990+9826 | 13097 | 110707 | 80 |
| d15-q8-q2-code-off-r2 | 12.09 | plain | True | fork | f0a2cd53a247/511 | None/None | 9818+0 | 13929 | 110565 | 74 |
| d15-q8-q2-code-on-r2 | 11.99 | fp-order | False | None | e4e223807e97/511 | None/None | 9737+0 | 13884 | 110679 | 72 |
| d15-q8-ref-code | 9.15 | plain | True | fork | 8290541d324b/511 | None/None | 17417+0 | 13660 | 110743 | 76 |
| d15-q8-tf-code | 9.05 | fp-order | False | None | 3c26a9d373a1/511 | None/None | 17751+0 | 14026 | 110223 | 74 |
| d15-q8-fr-code | 9.08 | fp-order | False | None | 3c26a9d373a1/511 | None/None | 17613+0 | 13697 | 110725 | 73 |
| d15-q8-ref-prose | 9.78 | plain | True | fork | 88ed0e11adfd/511 | None/None | 15147+0 | 13611 | 110761 | 73 |
| d15-q8-tf-prose | 9.51 | fp-order | False | None | 8520725fd8f9/511 | None/None | 16097+0 | 13971 | 110331 | 75 |
| d15-q8-fr-prose | 9.54 | fp-order | False | None | 8520725fd8f9/511 | None/None | 16083+0 | 13934 | 110439 | 76 |
| d15-q8-prof-off | 9.23 | plain | None | None | 726ef850ae7b/127 | None/None | 8400+0 | 13689 | 110739 | 75 |
| d15-q8-prof-on | 9.25 | fp-order | None | None | 03379cefb9d0/127 | None/None | 8334+0 | 13677 | 110757 | 75 |
| d15-q8-tfx-code | 7.42 | fp-order | None | None | None/None | 120/392 | 20888+0 | 13682 | 110367 | 70 |
| d15-q8-tfx-prose | 8.39 | fp-order | None | None | None/None | 440/72 | 17339+0 | 13906 | 110313 | 60 |

## Labels

* switch-on runs: 23, labelled fp-order: 23; not fp-order: none
* switch-off runs labelled fp-order: none
* ref_match on switch-on runs: ['False', 'None']; on switch-off runs: ['None', 'True']
    d15-q8-smoke-on-8: class=fp-order ref_match=False ref_line=None
    d15-q8-smoke-on-64: class=fp-order ref_match=False ref_line=None
    d15-q8-mx-code-on-r1: class=fp-order ref_match=False ref_line=None
    d15-q8-mx-prose-on-r1: class=fp-order ref_match=False ref_line=None
    d15-q8-mx-hash-on-r1: class=fp-order ref_match=False ref_line=None
    d15-q8-mx-long-on-r1: class=fp-order ref_match=False ref_line=None
    d15-q8-dsp-code-on-r1: class=fp-order ref_match=None ref_line=None
    d15-q8-dsp-json-on-r1: class=fp-order ref_match=None ref_line=None
    d15-q8-q2-code-on-r1: class=fp-order ref_match=False ref_line=None
    d15-q8-mx-code-on-r2: class=fp-order ref_match=False ref_line=None
    d15-q8-mx-prose-on-r2: class=fp-order ref_match=False ref_line=None
    d15-q8-mx-hash-on-r2: class=fp-order ref_match=False ref_line=None
    d15-q8-mx-long-on-r2: class=fp-order ref_match=False ref_line=None
    d15-q8-dsp-code-on-r2: class=fp-order ref_match=None ref_line=None
    d15-q8-dsp-json-on-r2: class=fp-order ref_match=None ref_line=None
    d15-q8-q2-code-on-r2: class=fp-order ref_match=False ref_line=None
    d15-q8-tf-code: class=fp-order ref_match=False ref_line=None
    d15-q8-fr-code: class=fp-order ref_match=False ref_line=None
    d15-q8-tf-prose: class=fp-order ref_match=False ref_line=None
    d15-q8-fr-prose: class=fp-order ref_match=False ref_line=None
    d15-q8-prof-on: class=fp-order ref_match=None ref_line=None
    d15-q8-tfx-code: class=fp-order ref_match=None ref_line=None
    d15-q8-tfx-prose: class=fp-order ref_match=None ref_line=None
    d15-q8-mx-code-off-r1: class=plain ref_match=True ref_line=fork
    d15-q8-mx-prose-off-r1: class=plain ref_match=True ref_line=fork
    d15-q8-mx-hash-off-r1: class=plain ref_match=True ref_line=fork
    d15-q8-mx-long-off-r1: class=plain ref_match=True ref_line=fork
    d15-q8-dsp-code-off-r1: class=dspark-fast ref_match=None ref_line=None
    d15-q8-dsp-json-off-r1: class=dspark-fast ref_match=None ref_line=None
    d15-q8-q2-code-off-r1: class=plain ref_match=True ref_line=fork
    d15-q8-mx-code-off-r2: class=plain ref_match=True ref_line=fork
    d15-q8-mx-prose-off-r2: class=plain ref_match=True ref_line=fork
    d15-q8-mx-hash-off-r2: class=plain ref_match=True ref_line=fork
    d15-q8-mx-long-off-r2: class=plain ref_match=True ref_line=fork
    d15-q8-dsp-code-off-r2: class=dspark-fast ref_match=None ref_line=None
    d15-q8-dsp-json-off-r2: class=dspark-fast ref_match=None ref_line=None
    d15-q8-q2-code-off-r2: class=plain ref_match=True ref_line=fork
    d15-q8-ref-code: class=plain ref_match=True ref_line=fork
    d15-q8-ref-prose: class=plain ref_match=True ref_line=fork
    d15-q8-prof-off: class=plain ref_match=None ref_line=None

## Speed, off vs on (same binary, two runs each, interleaved)

| config | off r1 | off r2 | off mean | on r1 | on r2 | on mean | change | MemAvail min off/on | NVMe max off/on | first differing token, on vs off (r1, r2) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| mx-code | 9.15 | 9.15 | 9.150 | 9.10 | 9.09 | 9.095 | -0.6% | 13871/13560 | 76/77 | 3, 3 |
| mx-prose | 9.80 | 9.76 | 9.780 | 9.56 | 9.48 | 9.520 | -2.7% | 13668/13865 | 75/72 | 108, 108 |
| mx-hash | 9.23 | 9.24 | 9.235 | 9.29 | 9.30 | 9.295 | +0.6% | 13880/13560 | 74/75 | 18, 18 |
| mx-long | 8.78 | 8.76 | 8.770 | 8.82 | 8.82 | 8.820 | +0.6% | 13439/13423 | 79/79 | 112, 112 |
| dsp-code | 11.87 | 11.94 | 11.905 | 12.31 | 12.30 | 12.305 | +3.4% | 13287/13324 | 79/79 | no per-token ids; rounds/accepted 172/340 vs 133/379, no per-token ids; rounds/accepted 172/340 vs 133/379 |
| dsp-json | 11.99 | 11.86 | 11.925 | 12.18 | 12.08 | 12.130 | +1.7% | 12890/13033 | 84/83 | no per-token ids; rounds/accepted 207/305 vs 209/303, no per-token ids; rounds/accepted 207/305 vs 209/303 |
| q2-code | 12.14 | 12.09 | 12.115 | 12.17 | 11.99 | 12.080 | -0.3% | 13929/13699 | 74/75 | 3, 3 |

## Divergence against the off reference (record fields from diverge.py)

* `d15-q8-ref-code` (9.15 tok/s): ref_tag=None, div_forced=None, div_tf_positions=None, div_tf_agree=None, div_tf_flips=None, div_flip_margin_max=None, div_flip_margin_med=None, div_logit_maxabs=None, div_logit_meanabs=None, div_logit_p99=None, div_bitexact_frac=None, div_first_div=None, div_common_prefix=None, div_compared=None, div_error=None
* `d15-q8-tf-code` (9.05 tok/s): ref_tag=d15-q8-ref-code, div_forced=False, div_tf_positions=3, div_tf_agree=0.666667, div_tf_flips=1, div_flip_margin_max=0.017719268798828125, div_flip_margin_med=0.017719268798828125, div_logit_maxabs=3.071782112121582, div_logit_meanabs=2.166362, div_logit_p99=3.071782112121582, div_bitexact_frac=0.0, div_first_div=3, div_common_prefix=3, div_compared=511, div_error=None
* `d15-q8-fr-code` (9.08 tok/s): ref_tag=d15-q8-ref-code, div_forced=False, div_tf_positions=3, div_tf_agree=0.666667, div_tf_flips=1, div_flip_margin_max=0.017719268798828125, div_flip_margin_med=0.017719268798828125, div_logit_maxabs=3.071782112121582, div_logit_meanabs=2.166362, div_logit_p99=3.071782112121582, div_bitexact_frac=0.0, div_first_div=3, div_common_prefix=3, div_compared=511, div_error=None
* `d15-q8-ref-prose` (9.78 tok/s): ref_tag=None, div_forced=None, div_tf_positions=None, div_tf_agree=None, div_tf_flips=None, div_flip_margin_max=None, div_flip_margin_med=None, div_logit_maxabs=None, div_logit_meanabs=None, div_logit_p99=None, div_bitexact_frac=None, div_first_div=None, div_common_prefix=None, div_compared=None, div_error=None
* `d15-q8-tf-prose` (9.51 tok/s): ref_tag=d15-q8-ref-prose, div_forced=False, div_tf_positions=108, div_tf_agree=0.990741, div_tf_flips=1, div_flip_margin_max=0.35112762451171875, div_flip_margin_med=0.35112762451171875, div_logit_maxabs=6.49397611618042, div_logit_meanabs=1.512197, div_logit_p99=5.226758241653442, div_bitexact_frac=0.0, div_first_div=108, div_common_prefix=108, div_compared=511, div_error=None
* `d15-q8-fr-prose` (9.54 tok/s): ref_tag=d15-q8-ref-prose, div_forced=False, div_tf_positions=108, div_tf_agree=0.990741, div_tf_flips=1, div_flip_margin_max=0.35112762451171875, div_flip_margin_med=0.35112762451171875, div_logit_maxabs=6.49397611618042, div_logit_meanabs=1.512197, div_logit_p99=5.226758241653442, div_bitexact_frac=0.0, div_first_div=108, div_common_prefix=108, div_compared=511, div_error=None
* `d15-q8-tfx-code` (7.42 tok/s): ref_tag=d15-q8-ref-code, div_forced=True, div_tf_positions=511, div_tf_agree=0.992172, div_tf_flips=4, div_flip_margin_max=0.17066383361816406, div_flip_margin_med=0.09863567352294922, div_logit_maxabs=11.490501403808594, div_logit_meanabs=1.584953, div_logit_p99=5.504632949829102, div_bitexact_frac=0.0, div_first_div=None, div_common_prefix=511, div_compared=511, div_error=None
* `d15-q8-tfx-prose` (8.39 tok/s): ref_tag=d15-q8-ref-prose, div_forced=True, div_tf_positions=511, div_tf_agree=0.976517, div_tf_flips=12, div_flip_margin_max=1.8532562255859375, div_flip_margin_med=0.38953590393066406, div_logit_maxabs=18.462955474853516, div_logit_meanabs=1.543067, div_logit_p99=3.984459638595581, div_bitexact_frac=0.0, div_first_div=None, div_common_prefix=511, div_compared=511, div_error=None

## C1 coarse profile

    d15-q8-prof-off: C1 report mode=coarse tokens=123 (first 4 skipped)
    d15-q8-prof-off: C1 step_span_ms_per_tok 108.157 gpu_span_ms_per_tok 0.000 outside_step_ms_per_tok 0.056
    d15-q8-prof-off: C1 segment          gpu_ms/tok host_ms/tok  rdwait_ms    n/tok
    d15-q8-prof-off: C1 layer_drain           0.000    104.979     43.188    40.00
    d15-q8-prof-off: C1 quiesce               0.000      0.002      0.000     1.00
    d15-q8-prof-off: C1 logits_read           0.000      3.176      0.000     1.00
    d15-q8-prof-off: C1 step_tail             0.000      0.000      0.000     1.00
    d15-q8-prof-off: C1 TOTAL                 0.000    108.157     43.188
    d15-q8-prof-on: C1 report mode=coarse tokens=123 (first 4 skipped)
    d15-q8-prof-on: C1 step_span_ms_per_tok 108.044 gpu_span_ms_per_tok 0.000 outside_step_ms_per_tok 0.062
    d15-q8-prof-on: C1 segment          gpu_ms/tok host_ms/tok  rdwait_ms    n/tok
    d15-q8-prof-on: C1 layer_drain           0.000    104.843     42.899    40.00
    d15-q8-prof-on: C1 quiesce               0.000      0.002      0.000     1.00
    d15-q8-prof-on: C1 logits_read           0.000      3.198      0.000     1.00
    d15-q8-prof-on: C1 step_tail             0.000      0.000      0.000     1.00
    d15-q8-prof-on: C1 TOTAL                 0.000    108.044     42.899
