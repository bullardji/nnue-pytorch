# Data Loader Overview

This short note outlines how batches are produced and transferred to the GPU during training.

## C++ stream

The dataset is read using a C++ library (`training_data_loader.*`). It assembles complete `SparseBatch` structures that contain indices, values and metadata for an entire batch. Batching on the C++ side avoids Python overhead and ensures the data is already contiguous in memory.

## Python wrappers

`nnue_dataset.SparseBatchDataset` exposes the C++ stream as an `IterableDataset`. Every item is a `SparseBatch` object. The `get_tensors(device)` method converts the raw pointers inside that structure into PyTorch tensors using pinned memory and non‑blocking transfers:

```python
white_values = torch.from_numpy(...).pin_memory().to(device, non_blocking=True)
```

This lets the loader overlap GPU copies with computation.

`FixedNumBatchesDataset` wraps another dataset to provide a fixed number of batches per epoch. It starts a background thread that prefetches batches into a queue so that `DataLoader` can yield them without waiting on the C++ stream.

## DataLoader usage

`train.py` creates `SparseBatchDataset` instances for training and validation and wraps them with `FixedNumBatchesDataset`. PyTorch's `DataLoader` is then constructed with `batch_size=None` so each yielded item is already a preassembled batch on the target device.
