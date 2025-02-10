# MicroQwen

This is an ongoing project to implement inference for the smallest of the Qwen2.5 LLM models (0.5B params).

The purpose of this project is to learn more about the low-level implementation details of auto-regressive transformers,
as well as practice optimising computationally expensive arithmetic operations in C.

I'm referencing the following documents for implementation details:
- <a href="https://arxiv.org/abs/2412.15115">Qwen2.5 Technical Report</a>
- <a href="https://arxiv.org/abs/1706.03762">Attention Is All You Need</a>
- <a href="https://jalammar.github.io/illustrated-transformer/">The Illustrated Transformer</a>
- <a href="https://arxiv.org/pdf/2305.13245">GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints</a>

**Note**:
There are *many* areas for improvement in the model's implementation at the moment, but my goal is just to get it working first 
and I'll go back to improve and optimise things later.

Also, I'm currently using Python scripts (located in the `/scripts` directory) to:
- Pre-tokenise inputs
- Simplify the safetensor format to be easier to parse (essentially just extracting weight tensor shapes and byte offsets)

Both of these scripts exist because I don't feel like writing the byte-pair decoder and JSON parser to do these things at the moment,
and I'm trying to avoid external dependencies for the core implementation. Hopefully in the future I can integrate everything into the main 
program.


## Implementation To-do List:
- [x] bfloat16 type
- [x] Basic linear algebra library (bfloat16 matrix operations)
- [x] Feed forward neural networks (including relevant activation functions)
- [x] Dot product self-attention head
- [x] Grouped Query Attention mechanism
- [ ] Attention sub-layer masking (to preseve auto-regression)
- [x] SwiGLU non-linear activation
- [ ] RoPE Rotary positional embeddings
- [ ] Layer RMS normalisation
- [ ] Complete transformer layer struct
- [ ] Un-embedding layer
- [ ] Benchmark token generation performance and start multithreading computations


After I finish implementing the core transformer decoder architecture, I'll go back through to integrate
tokenisation into the model - so I don't need to pretokenise everything with a Python script.



