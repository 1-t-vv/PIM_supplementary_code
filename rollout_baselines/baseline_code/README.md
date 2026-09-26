# Autoregressive Baseline Models

This directory contains source code only for the six autoregressive comparison rows:
HOPNet, HOPNet-Adapt, FIGNet-Reimplemented, FIGNet-Reimplemented-Adapt,
MeshGraphNet-Reimplemented, and MeshGraphNet-Reimplemented-Adapt.

It contains model modules only: no trajectory dataset, data loader,
preprocessing, configuration file, train/validation/test split, checkpoint, evaluator,
prediction archive, metric file, or data generator. The two variants in each family
share the listed model implementation; the Adapt variant changes the feature dimension
supplied by the caller.

## HOPNet and HOPNet-Adapt

- `hopnet/models/`: HOPNet neural modules.
- `hopnet/LICENSE.md`: upstream redistribution license.

## FIGNet-Reimplemented variants

- `fignet/fignet/`: the model modules required by the task implementation.
- `fignet/adapter/model.py`: controlled-task model construction and normalizer handling.
- `fignet/LICENSE`: upstream redistribution license.

## MeshGraphNet-Reimplemented variants

- `meshgraphnet/meshGraphNets_pytorch/model/`: message-passing model implementation.
- `meshgraphnet/meshGraphNets_pytorch/utils/normalization.py`: normalizer required by
  the model simulator.
- `meshgraphnet/adapter/model.py`: controlled-task model and running normalizers.
- `meshgraphnet/LICENSE`: upstream redistribution license.

The supplementary archive does not provide a data path or commands for training,
evaluating, or regenerating results for these non-PIM models.
