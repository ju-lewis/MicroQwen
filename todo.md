
# MicroQwen

**Goal:**
Write a minimal Qwen2-0.5b inference engine from scratch in C, with the goal of targetting minimal hardware requirements.



## Sub-Goal 1:
- Pre-tokenise input stream using an open source tokeniser (tiktoken)
- MicroQwen handles embedding and inference up until final unembedding stage
- Decode produced tokens using tiktoken


**Todo:**
(Embedding)
- [x] Lookup vector embedding for parsed token

(Inference)
- [ ]





(Tokenising)
- [ ] Parse tokeniser JSON into more compact representation
- [ ] Tokenise string into token IDs
