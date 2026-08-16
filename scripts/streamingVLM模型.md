# streamingVLM模型

```YAML
inference.py:
        model.generate(**inputs, streaming_args=streaming_args)
            │
            ▼  [patched] streaming_generate
        streaming_generate_qwen.py:
            StreamingCache() 创建
            self._sample(input_ids, **model_kwargs)
                │
                ▼  [patched] _sample
            streaming_generate_qwen.py:
                self.prepare_inputs_for_generation(input_ids, **model_kwargs)
                    │
                    ▼  [patched] prepare_multiturn_multimodal_inputs_for_generation
                prepare_generation.py:
                    截断 input_ids + 清空 decode 视觉输入
                    return model_inputs (streaming_args 在 kwargs 中)
                
                self(**model_inputs)   ← 即 model.forward(...)
                    │
                    ▼  [patched] qwen2_5_vl_forward
                model_forward.py:
                    self.model(...)    ← 即 model.model.forward(...)
                        │
                        ▼  [patched] model_forward
                    model_forward.py:
                        embed_tokens + masked_scatter 视觉
                        get_rope_index(streaming_args.input_ids, ...)  ← [patched]
                        │                                    │
                        │                           pos_emb.py:
                        │                               3D position_ids 计算
                        │
                        self.language_model(...)
                            │
                            ▼  [patched] streaming_language_model_forward
                        language_forward.py:
                            rotary_emb → position_embeddings
                            self._update_causal_mask(...)  ← [patched]
                            for layer in self.layers:
                                layer(...)
                                    │
                                    ▼  [patched] streaming_text_decoder_layer_forward
                                language_forward.py:
                                    self.self_attn(...)
                                        │
                                        ▼  [patched] streaming_text_flash_attn_forward
                                    language_forward.py:
                                        Q/K/V proj
                                        cache.update(K, V)
                                        apply_multimodal_rotary_pos_emb(Q, K)  ← RoPE
                                        _flash_attention_forward(Q, K, V, None) ← 计算注意力
                                        return attn_output
    
                        (visual encoder 分支，只在 prefill 时执行)
                        self.get_video_features(...)
                            │
                            ▼  self.visual(...)  ← [patched] streaming_visual_encoder_forward
                        vision_forward.py:
                            patch_embed → blocks → merger
                            for blk in blocks:
                                blk(...)  ← [patched] streaming_visual_block_forward
                                    blk.attn(...)  ← [patched] streaming_visual_attention_forward
    
    
    
    
    streaming_args 的穿透路径
    
    streaming_args 从 inference.py 出发，经过 7 层 函数调用到达最底层的 attention：
    
    
    inference.py: model.generate(streaming_args=...)
        ↓ **kwargs 传递
    streaming_generate: model_kwargs 里带 streaming_args
        ↓ **model_kwargs 传递
    _sample: self(**model_inputs) 里带 streaming_args
        ↓ **kwargs 传递
    qwen2_5_vl_forward(streaming_args=...)
        ↓ 显式参数
    model_forward(streaming_args=...)
        ↓ 显式参数
    streaming_language_model_forward(streaming_args=...)
        ↓ 显式参数
    streaming_text_decoder_layer_forward(streaming_args=...)
        ↓ 显式参数
    streaming_text_flash_attn_forward(streaming_args=...)  ← 最终消费点
    

```



