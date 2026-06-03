# Mini LLM From Scratch

A modern transformer language model implemented in PyTorch from scratch.

## Features

- Character-level tokenization
- RMSNorm
- RoPE (Rotary Positional Embeddings)
- Grouped Query Attention (GQA)
- SwiGLU Feed Forward Network
- Weight Tying
- Shakespeare Dataset Training
- Autoregressive Text Generation

## Architecture

Modern Transformer Stack:

Token Embedding
→ RMSNorm
→ Grouped Query Attention (GQA)
→ Residual Connection
→ RMSNorm
→ SwiGLU
→ Residual Connection
→ Final RMSNorm
→ Language Modeling Head

## Model Configuration

| Parameter | Value |
|-----------|---------|
| Parameters | 2.76M |
| Layers | 4 |
| Embedding Dimension | 256 |
| Query Heads | 8 |
| KV Heads | 2 |
| Context Length | 256 |

## Training Results

| Metric | Value |
|---------|---------|
| Train Loss | 1.0163 |
| Validation Loss | 1.5086 |

## Sample Output

```text
ROMEO:

That she is come to me at once so strength
To shake the contrary of the most dead.
```

## Technologies Used

- Python
- PyTorch
- CUDA
- Google Colab

## Future Improvements

- Implement Byte Pair Encoding (BPE)
- Train on larger datasets
- Increase model size
- Fine-tuning experiments
- Convert notebook into modular codebase

## Repository Structure

```text
mini-llm-from-scratch/
│
├── notebook.ipynb
├── README.md
```

## Author

Sumit Jha
