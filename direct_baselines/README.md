# Direct Terminal-State Baseline Models

This section contains model code used for the full-dataset direct-prediction
comparison. It does not include baseline checkpoints, data, training scripts, or
evaluation outputs.

## Rigid-Pose MLP

`rigid_pose_mlp/model.py` exports `RigidPoseMLP`. The constructor defaults match the
paper: a 30-dimensional object descriptor, 16-dimensional object embedding,
256-dimensional object embedding/context width, one residual MLP block, and zero
dropout. The model predicts a center displacement and continuous 6D rotation for every
object, so its output is rigid by construction.

## Vertex Transformer

`vertex_transformer/model.py` exports `VertexTransformer`. The paper-matched defaults
are hidden width 384, six attention heads, one intra-object KNN attention block, one
object-token Transformer layer, no post-context vertex block, KNN size supplied by the
input cache, feed-forward expansion four, a 16-dimensional object embedding, and three
Fourier-coordinate bands. The decoder predicts one displacement per input vertex.

These are original baselines developed for the PIM study. **Vertex Transformer** is the
public name used in the submission; it is not an implementation of the Point
Transformer architecture of Zhao et al.

