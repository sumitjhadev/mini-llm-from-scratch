# Mini LLM From Scratch

A modern transformer language model implemented in PyTorch.

## Features

- Character-level tokenization
- RMSNorm
- RoPE (Rotary Positional Embeddings)
- Grouped Query Attention (GQA)
- SwiGLU
- Weight Tying
- Shakespeare Training
- Autoregressive Text Generation

## Model Configuration

| Parameter | Value |
|------------|---------|
| Parameters | 2.76M |
| Layers | 4 |
| Embedding Dimension | 256 |
| Query Heads | 8 |
| KV Heads | 2 |
| Context Length | 256 |

## Results

- Train Loss: 1.0163
- Validation Loss: 1.5086

## Sample Output

ROMEO:

That she is come to me at once so strength
To shake the contrary of the most dead.

## Future Improvements

- Byte Pair Encoding (BPE)
- Larger datasets
- Bigger model sizes
- Fine-tuning experiments
