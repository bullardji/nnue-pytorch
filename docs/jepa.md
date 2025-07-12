# JEPA Overview

This document provides a brief reference for the Joint-Embedding Predictive Architecture (JEPA) and its video variant V-JEPA 2. These papers are included in the `docs/papers` directory.

## JEPA

JEPA describes an agent composed of differentiable modules:

- **Perception** encodes observations into a latent state `s[t]`.
- **World Model** predicts future latent states given action sequences.
- **Cost Module** assigns an "energy" value measuring discomfort.
- **Actor** plans actions to minimize predicted future cost.
- **Short-Term Memory** stores past state–cost pairs to train the critic.
- **Configurator** adjusts module parameters depending on the task.

Two execution modes are outlined:

1. **Reactive**: compute `s[0] = Enc(x)` and act with a feed-forward policy.
2. **Model-Predictive Control**: iteratively predict states `s[t+1] = Pred(s[t], a[t])`, sum costs `F = sum_t C(s[t])`, and optimize actions to minimize `F`.

The cost module factorizes intrinsic and learned terms:

```
C(s) = IC(s) + TC(s)
IC(s) = sum_i u_i * IC_i(s)
TC(s) = sum_j v_j * TC_j(s)
```

## V-JEPA 2

V-JEPA 2 scales JEPA for video by predicting masked tokens in feature space. A student encoder is trained to match the output of a slowly updated teacher encoder. Two randomly augmented views of the same clip are used, and the predictor reconstructs the masked tokens of one view from the other. The paper explores larger models, longer training and progressive resolution.

V-JEPA 2-AC further introduces an action-conditioned predictor trained on robot demonstrations to enable model-predictive control.

## Applying to NNUE

When adapting JEPA to chess, the NNUE feature extractor can serve as the encoder and JEPA predictors can forecast embeddings of future board states given move sequences. Intrinsic costs might capture material or king safety, while a critic learns trajectory-based objectives.

### Masking Strategies

The training script exposes a `--jepa-mask-mode` option controlling how latent
features are masked when computing the JEPA loss. The default `random` strategy
masks individual embedding dimensions. The alternative `board_group` strategy
masks whole piece/square groups inferred from the active feature set.

## Further Reading

- [JEPA paper](papers/JEPA_paper.pdf)
- [V-JEPA2 paper](papers/V-JEPA2_paper.pdf)

