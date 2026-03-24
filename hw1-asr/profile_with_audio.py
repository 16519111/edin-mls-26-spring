#!/usr/bin/env python3                                                                                                                    
"""Profile kernel execution times during real audio inference (low memory)."""                                                            
                                                                                                                                        
import sys                                                                                                                                
import os                                                                                                                                 
import gc                                                                                                                                 
import time                                                                                                                               
import torch
import numpy as np                                                                                                                        
import struct                                                                                                                             
import wave                                                                                                                               
                                                                                                                                        
                                                                                                                                        
def load_audio(audio_path):                                                                                                             
    with wave.open(audio_path, 'rb') as wav:                                                                                              
        sr = wav.getframerate()                                                                                                         
        n_frames = wav.getnframes()                                                                                                       
        n_channels = wav.getnchannels()                                                                                                   
        sample_width = wav.getsampwidth()                                                                                                 
        raw_data = wav.readframes(n_frames)                                                                                               
        if sample_width == 2:                                                                                                             
            audio = np.array(                                                                                                             
                struct.unpack(f'<{n_frames * n_channels}h', raw_data),                                                                    
                dtype=np.float32,                                                                                                         
            )                                                                                                                           
            audio = audio / 32768.0
        else:                                                                                                                             
            audio = np.zeros(n_frames, dtype=np.float32)
        if n_channels > 1:                                                                                                                
            audio = audio.reshape(-1, n_channels).mean(axis=1)                                                                            
    return audio, sr                                                                                                                      
                                                                                                                                        
                                                                                                                                        
def cuda_time(fn, warmup=1, runs=3):                                                                                                      
    """Time a function using CUDA events. Returns mean ms."""                                                                             
    start = torch.cuda.Event(enable_timing=True)                                                                                          
    end = torch.cuda.Event(enable_timing=True)
                                                                                                                                        
    for _ in range(warmup):                                                                                                               
        with torch.no_grad():                                                                                                             
            _ = fn()                                                                                                                      
        torch.cuda.synchronize()                                                                                                          
    del _                                                                                                                                 
    gc.collect()                                                                                                                          
    torch.cuda.empty_cache()                                                                                                              
                                                                                                                                        
    times = []                                                                                                                            
    for _ in range(runs):                                                                                                                 
        torch.cuda.synchronize()                                                                                                          
        start.record()                                                                                                                    
        with torch.no_grad():
            result = fn()                                                                                                                 
        end.record()                                                                                                                    
        torch.cuda.synchronize()                                                                                                          
        times.append(start.elapsed_time(end))                                                                                             
        del result                                                                                                                        
    return np.mean(times), np.std(times)                                                                                                  
                                                                                                                                        

def main():                                                                                                                               
    folder = sys.argv[1] if len(sys.argv) > 1 else "glm_asr_triton_template"                                                            
    audio_path = sys.argv[2] if len(sys.argv) > 2 else "test_audio.wav"                                                                   
    num_runs = int(sys.argv[3]) if len(sys.argv) > 3 else 3                                                                               
                                                                                                                                        
    script_dir = os.path.dirname(os.path.abspath(__file__))                                                                               
    folder_path = os.path.join(script_dir, folder)                                                                                        
    sys.path.insert(0, folder_path)                                                                                                       
                                                                                                                                        
    for mod in ['weight_loader', 'model', 'layers', 'attention', 'rope',                                                                  
                'conv', 'torch_glm', 'encoder', 'decoder', 'config', 'tokenizer']:                                                        
        sys.modules.pop(mod, None)                                                                                                        
                                                                                                                                        
    is_scratch = 'scratch' in folder.lower()                                                                                              
    device = torch.device("cuda")                                                                                                         
                                                                                                                                        
    # ================================================================                                                                    
    # Load model                                                                                                                          
    # ================================================================                                                                    
    print(f"Loading model from {folder}...")                                                                                            
    if is_scratch:
        from torch_glm import load_model_and_processor                                                                                    
        model, processor = load_model_and_processor(dtype='float32')                                                                      
    else:                                                                                                                                 
        from weight_loader import load_model_from_hf                                                                                      
        model, processor = load_model_from_hf("zai-org/GLM-ASR-Nano-2512")                                                                
                                                                                                                                        
        import layers                                                                                                                     
        print(f"Linear.BACKEND = {layers.Linear.BACKEND}")                                                                                
        print(f"MLP.FUSED = {layers.MLP.FUSED}")                                                                                          
        print(f"EncoderMLP.FUSED = {getattr(layers, 'EncoderMLP', type('X', (), {'FUSED': 'N/A'})).FUSED}")                               
                                                                                                                                        
    # ================================================================                                                                    
    # Load audio and prepare inputs                                                                                                       
    # ================================================================                                                                    
    print(f"\nLoading audio from {audio_path}...")
    audio_array, sr = load_audio(                                                                                                         
        audio_path if os.path.isabs(audio_path)                                                                                           
        else os.path.join(script_dir, audio_path)                                                                                         
    )                                                                                                                                     
    print(f"Audio: {len(audio_array)/sr:.2f}s")                                                                                           
                                                                                                                                        
    if is_scratch:                                                                                                                        
        inputs = processor.apply_transcription_request(audio_array)                                                                       
        input_features = inputs['input_features'].to(device=device, dtype=torch.float32)                                                  
        input_ids = inputs['input_ids'].to(device=device)                                                                                 
        attention_mask = inputs['attention_mask'].to(device=device)                                                                       
        input_features_mask = None                                                                                                        
    else:                                                                                                                                 
        inputs = processor.apply_transcription_request(audio_array)                                                                       
        input_features = inputs.input_features.to(device=device, dtype=torch.float32)                                                     
        input_ids = inputs.input_ids.to(device=device, dtype=torch.int64)                                                                 
        input_features_mask = None                                                                                                        
        attention_mask = None                                                                                                             
        if hasattr(inputs, 'input_features_mask') and inputs.input_features_mask is not None:                                             
            input_features_mask = inputs.input_features_mask.to(device=device, dtype=torch.float32)                                       
                                                                                                                                        
    print(f"Input features: {input_features.shape}")                                                                                      
    print(f"Input IDs: {input_ids.shape}")                                                                                                
                                                                                                                                        
    # ================================================================
    # Determine generate function                                                                                                         
    # ================================================================                                                                    
    if is_scratch:
        generate_fn = model.generate                                                                                                      
        print(f"Using: model.generate (PyTorch)")                                                                                         
    else:                                                                                                                                 
        generate_fn = model.generate                                                                                                      
        for name in ['generate_v8b', 'generate_v8', 'generate_v6']:                                                                       
            if hasattr(model, name):                                                                                                      
                generate_fn = getattr(model, name)                                                                                        
                break                                                                                                                     
        print(f"Using: {generate_fn.__name__}")                                                                                         
                                                                                                                                        
    # ================================================================                                                                  
    # 1. End-to-end inference                                                                                                             
    # ================================================================                                                                    
    print("\n" + "=" * 70)
    print("END-TO-END INFERENCE")                                                                                                         
    print("=" * 70)                                                                                                                     
                                                                                                                                        
    if is_scratch:                                                                                                                      
        def run_generate():                                                                                                               
            return model.generate(                                                                                                        
                input_ids=input_ids,                                                                                                      
                input_features=input_features,                                                                                            
                attention_mask=attention_mask,                                                                                            
                max_new_tokens=50,                                                                                                        
                do_sample=False,                                                                                                        
            )                                                                                                                             
    else:                                                                                                                                 
        def run_generate():                                                                                                               
            try:                                                                                                                          
                return generate_fn(                                                                                                     
                    input_features, input_ids=input_ids,                                                                                  
                    input_features_mask=input_features_mask,                                                                              
                    max_new_tokens=50, temperature=1.0, top_k=1,                                                                          
                )                                                                                                                         
            except TypeError:                                                                                                             
                return generate_fn(                                                                                                       
                    input_features, input_ids=input_ids,
                    max_new_tokens=50, temperature=1.0, top_k=1,                                                                          
                )                                                                                                                         
                                                                                                                                        
    mean, std = cuda_time(run_generate, warmup=1, runs=num_runs)                                                                          
    print(f"  Total inference:    {mean:.2f} ms (+/- {std:.2f})")                                                                       
                                                                                                                                        
    gc.collect()                                                                                                                          
    torch.cuda.empty_cache()                                                                                                              
                                                                                                                                        
    # # ================================================================                                                                    
    # # 2. Component-level profiling
    # # ================================================================                                                                    
    # print("\n" + "=" * 70)                                                                                                                
    # print("COMPONENT-LEVEL PROFILING")                                                                                                    
    # print("=" * 70)                                                                                                                       
                                                                                                                                        
    # if is_scratch:                                                                                                                        
    #     # Scratch model: audio_encoder, multi_modal_projector, language_model                                                           
    #     # Audio Encoder                                                                                                                   
    #     mean, std = cuda_time(                                                                                                            
    #         lambda: model.audio_encoder(input_features),                                                                                  
    #         warmup=1, runs=num_runs,                                                                                                      
    #     )                                                                                                                                 
    #     print(f"  Audio Encoder:      {mean:.2f} ms (+/- {std:.2f})")                                                                     
                                                                                                                                        
    #     with torch.no_grad():                                                                                                             
    #         audio_hidden = model.audio_encoder(input_features)                                                                            
    #     torch.cuda.synchronize()                                                                                                          
                                                                                                                                        
    #     # Audio reshape (same as in model.forward)                                                                                        
    #     factor = model.audio_reshape_factor                                                                                               
    #     batch_size, seq_len, hidden_size = audio_hidden.shape                                                                             
    #     truncated_len = (seq_len // factor) * factor                                                                                      
    #     if truncated_len < seq_len:                                                                                                       
    #         audio_hidden = audio_hidden[:, :truncated_len, :]                                                                             
    #     audio_reshaped = audio_hidden.reshape(                                                                                            
    #         batch_size, -1, model.config.audio_config.intermediate_size                                                                   
    #     )                                                                                                                                 
                                                                                                                                        
    #     # Multi-modal Projector                                                                                                           
    #     mean, std = cuda_time(                                                                                                          
    #         lambda: model.multi_modal_projector(audio_reshaped),                                                                          
    #         warmup=1, runs=num_runs,                                                                                                      
    #     )                                                                                                                                 
    #     print(f"  Projector:          {mean:.2f} ms (+/- {std:.2f})")                                                                     
                                                                                                                                        
    #     with torch.no_grad():                                                                                                             
    #         audio_features_proj = model.multi_modal_projector(audio_reshaped)                                                             
    #     del audio_hidden, audio_reshaped                                                                                                  
    #     gc.collect()                                                                                                                      
    #     torch.cuda.empty_cache()                                                                                                          
                                                                                                                                        
    #     # Build combined embeddings                                                                                                       
    #     with torch.no_grad():                                                                                                           
    #         text_embeds = model.language_model.model.embed_tokens(input_ids)                                                              
    #         audio_token_id = model.audio_token_id                                                                                         
    #         audio_mask_bool = (input_ids == audio_token_id)                                                                               
    #         audio_positions = torch.where(audio_mask_bool[0])[0]                                                                          
                                                                                                                                        
    #         if len(audio_positions) > 0:                                                                                                  
    #             first_pad = int(audio_positions[0].item())                                                                                
    #             last_pad = int(audio_positions[-1].item())                                                                                
    #             before = text_embeds[0, :first_pad, :]                                                                                    
    #             after = text_embeds[0, last_pad + 1:, :]                                                                                  
    #             combined_embeds = torch.cat(                                                                                              
    #                 [before[None], audio_features_proj, after[None]], dim=1                                                               
    #             )                                                                                                                         
    #         else:                                                                                                                         
    #             combined_embeds = text_embeds                                                                                             
                                                                                                                                        
    #     del audio_features_proj, text_embeds                                                                                              
    #     gc.collect()                                                                                                                    
    #     torch.cuda.empty_cache()                                                                                                          
                                                                                                                                        
    #     # Decoder Prefill                                                                                                                 
    #     mean, std = cuda_time(                                                                                                          
    #         lambda: model.language_model(inputs_embeds=combined_embeds),                                                                  
    #         warmup=1, runs=num_runs,                                                                                                      
    #     )                                                                                                                                 
    #     print(f"  Decoder Prefill:    {mean:.2f} ms (+/- {std:.2f})")                                                                     
                                                                                                                                        
    #     with torch.no_grad():                                                                                                             
    #         outputs = model.language_model(inputs_embeds=combined_embeds)                                                                 
    #     hidden_states = outputs['logits'] if isinstance(outputs, dict) else outputs                                                       
    #     torch.cuda.synchronize()                                                                                                          
                                                                                                                                        
    #     # Single Decode Step                                                                                                              
    #     with torch.no_grad():                                                                                                             
    #         if isinstance(outputs, dict):                                                                                                 
    #             logits = outputs['logits']                                                                                                
    #         else:                                                                                                                         
    #             logits = outputs                                                                                                          
    #         next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)                                                           
    #     del outputs, logits                                                                                                               
    #     gc.collect()                                                                                                                      
    #     torch.cuda.empty_cache()                                                                                                          
                                                                                                                                        
    #     def single_decode_step():                                                                                                         
    #         embed = model.language_model.model.embed_tokens(next_token)                                                                 
    #         out = model.language_model(inputs_embeds=embed)                                                                               
    #         if isinstance(out, dict):                                                                                                     
    #             l = out['logits']                                                                                                         
    #         else:                                                                                                                         
    #             l = out                                                                                                                   
    #         return torch.argmax(l[:, -1, :], dim=-1, keepdim=True)                                                                      
                                                                                                                                        
    #     mean, std = cuda_time(single_decode_step, warmup=1, runs=num_runs)                                                                
    #     print(f"  Single Decode Step: {mean:.2f} ms (+/- {std:.2f})")                                                                     
    #     print(f"  50 Decode Steps:    {mean * 50:.2f} ms (estimated)")                                                                    
                                                                                                                                        
    # else:                                                                                                                                 
    #     # Triton model: audio_encoder, multi_modal_projector, text_decoder, lm_head                                                       
    #     mean, std = cuda_time(                                                                                                            
    #         lambda: model.audio_encoder(input_features),                                                                                  
    #         warmup=1, runs=num_runs,                                                                                                      
    #     )                                                                                                                                 
    #     print(f"  Audio Encoder:      {mean:.2f} ms (+/- {std:.2f})")                                                                   
                                                                                                                                        
    #     with torch.no_grad():                                                                                                             
    #         audio_features = model.audio_encoder(input_features)                                                                          
    #     torch.cuda.synchronize()                                                                                                          
                                                                                                                                        
    #     mean, std = cuda_time(                                                                                                            
    #         lambda: model.multi_modal_projector(audio_features),                                                                          
    #         warmup=1, runs=num_runs,                                                                                                      
    #     )                                                                                                                               
    #     print(f"  Projector:          {mean:.2f} ms (+/- {std:.2f})")                                                                     
                                                                                                                                        
    #     with torch.no_grad():                                                                                                             
    #         projected = model.multi_modal_projector(audio_features)                                                                       
    #     del audio_features                                                                                                                
    #     gc.collect()                                                                                                                    
    #     torch.cuda.empty_cache()                                                                                                          
                                                                                                                                        
    #     with torch.no_grad():                                                                                                             
    #         text_embeds = model.text_decoder.embed_tokens(input_ids)                                                                      
    #         combined_embeds = text_embeds.clone()                                                                                         
    #         audio_token_id = 59260                                                                                                        
    #         audio_mask_bool = (input_ids == audio_token_id)                                                                               
    #         if torch.any(audio_mask_bool):                                                                                                
    #             audio_positions = torch.where(audio_mask_bool[0])[0]                                                                      
    #             first_pad = int(audio_positions[0].item())                                                                                
    #             last_pad = int(audio_positions[-1].item())                                                                                
    #             before = text_embeds[0, :first_pad, :]                                                                                    
    #             after = text_embeds[0, last_pad + 1:, :]                                                                                  
    #             if projected.ndim == 3:                                                                                                   
    #                 proj_2d = projected[0]                                                                                                
    #             else:                                                                                                                     
    #                 proj_2d = projected                                                                                                   
    #             combined_embeds = torch.cat(                                                                                              
    #                 [before[None], proj_2d[None], after[None]], dim=1                                                                     
    #             )                                                                                                                         
    #     del projected, text_embeds                                                                                                      
    #     gc.collect()                                                                                                                      
    #     torch.cuda.empty_cache()                                                                                                          
                                                                                                                                        
    #     mean, std = cuda_time(                                                                                                            
    #         lambda: model.text_decoder(inputs_embeds=combined_embeds),                                                                  
    #         warmup=1, runs=num_runs,                                                                                                      
    #     )                                                                                                                                 
    #     print(f"  Decoder Prefill:    {mean:.2f} ms (+/- {std:.2f})")                                                                     
                                                                                                                                        
    #     with torch.no_grad():                                                                                                             
    #         hidden_states = model.text_decoder(inputs_embeds=combined_embeds)                                                             
    #     torch.cuda.synchronize()                                                                                                          
                                                                                                                                        
    #     mean, std = cuda_time(                                                                                                            
    #         lambda: model.lm_head(hidden_states[:, -1:, :]),                                                                              
    #         warmup=1, runs=num_runs,                                                                                                      
    #     )                                                                                                                                 
    #     print(f"  LM Head:            {mean:.2f} ms (+/- {std:.2f})")                                                                     
                                                                                                                                        
    #     with torch.no_grad():                                                                                                             
    #         logits = model.lm_head(hidden_states[:, -1:, :])                                                                              
    #         next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)                                                             
    #     del hidden_states, logits                                                                                                         
    #     gc.collect()                                                                                                                      
    #     torch.cuda.empty_cache()                                                                                                          
                                                                                                                                        
    #     def single_decode_step():                                                                                                         
    #         embed = model.text_decoder.embed_tokens(next_token)                                                                         
    #         h = model.text_decoder(inputs_embeds=embed)                                                                                   
    #         l = model.lm_head(h)                                                                                                          
    #         return torch.argmax(l[:, -1, :], dim=-1, keepdim=True)                                                                        
                                                                                                                                        
    #     mean, std = cuda_time(single_decode_step, warmup=1, runs=num_runs)                                                                
    #     print(f"  Single Decode Step: {mean:.2f} ms (+/- {std:.2f})")                                                                     
    #     print(f"  50 Decode Steps:    {mean * 50:.2f} ms (estimated)")                                                                    
                                                                                                                                        
    # # ================================================================                                                                    
    # # 3. Per-layer profiling (first 5 decoder layers)                                                                                     
    # # ================================================================                                                                    
    # print("\n" + "=" * 70)                                                                                                                
    # print("PER-LAYER DECODER PROFILING (first 5 layers)")                                                                                 
    # print("=" * 70)                                                                                                                       
                                                                                                                                        
    # gc.collect()                                                                                                                          
    # torch.cuda.empty_cache()                                                                                                              
                                                                                                                                        
    # test_input = combined_embeds                                                                                                          
                                                                                                                                        
    # if is_scratch:                                                                                                                        
    #     decoder_layers = model.language_model.model.layers                                                                              
    # else:                                                                                                                                 
    #     decoder_layers = model.text_decoder.layers if hasattr(model.text_decoder, 'layers') else []                                     
                                                                                                                                        
    # for i, layer in enumerate(decoder_layers[:5]):                                                                                        
    #     current_input = test_input                                                                                                        
                                                                                                                                        
    #     def run_layer(inp=current_input, lay=layer):                                                                                      
    #         try:                                                                                                                          
    #             return lay(inp)                                                                                                           
    #         except TypeError:                                                                                                             
    #             seq_len = inp.shape[1]                                                                                                    
    #             pos_ids = torch.arange(                                                                                                   
    #                 seq_len, dtype=torch.int64, device=inp.device                                                                         
    #             ).reshape(1, -1)                                                                                                          
    #             try:                                                                                                                      
    #                 return lay(inp, position_ids=pos_ids)                                                                                 
    #             except TypeError:                                                                                                         
    #                 return lay(inp)                                                                                                       
                                                                                                                                        
    #     mean, std = cuda_time(run_layer, warmup=1, runs=num_runs)                                                                         
    #     print(f"  Decoder Layer {i}:    {mean:.2f} ms (+/- {std:.2f})")                                                                   
                                                                                                                                        
    #     with torch.no_grad():                                                                                                             
    #         out = run_layer()                                                                                                             
    #         if isinstance(out, tuple):                                                                                                    
    #             test_input = out[0]                                                                                                       
    #         else:                                                                                                                         
    #             test_input = out                                                                                                          
    #     torch.cuda.synchronize()                                                                                                          
                                                                                                                                        
    # ================================================================                                                                    
    # 4. Operator-level profiling (individual kernels)                                                                                    
    # ================================================================                                                                    
    if is_scratch:                                                                                                                        
        print("\n" + "=" * 70)                                                                                                            
        print("OPERATOR-LEVEL PROFILING: Skipped (pure PyTorch, no Triton kernels)")                                                      
        print("=" * 70)                                                                                                                   
    else:                                                                                                                                 
        print("\n" + "=" * 70)                                                                                                            
        print("OPERATOR-LEVEL PROFILING")                                                                                                 
        print("=" * 70)                                                                                                                   
                                                                                                                                        
        del test_input, combined_embeds                                                                                                   
        gc.collect()                                                                                                                      
        torch.cuda.empty_cache()                                                                                                          
                                                                                                                                        
        B, S = 1, 256                                                                                                                     
        H_enc, H_dec = 1280, 3584
        I_enc, I_dec = 5120, 18944                                                                                                        
        N_heads_enc, N_heads_dec = 20, 28                                                                                                 
        N_kv_heads = 4                                                                                                                    
        HD_enc = H_enc // N_heads_enc                                                                                                     
        HD_dec = H_dec // N_heads_dec                                                                                                     
                                                                                                                                        
        import triton                                                                                                                     
        from layers import (                                                                                                              
            rmsnorm_kernel, layernorm_kernel, softmax_kernel,                                                                             
            gelu_kernel, silu_kernel, linear_kernel_tf32,                                                                                 
            pad_to_multiple,                                                                                                              
        )                                                                                                                                 
        from attention import (                                                                                                           
            attention_scores_kernel,                                                                                                      
            softmax_inplace_kernel, attention_output_kernel,                                                                              
        )                                                                                                                                 
        from rope import compute_freqs_kernel                                                                                             
                                                                                                                                        
        print("\n  --- Element-wise Kernels ---")                                                                                         
                                                                                                                                        
        x = torch.randn(B * S * H_dec, device=device, dtype=torch.float32)                                                                
        y = torch.empty_like(x)                                                                                                         
        n = x.numel()                                                                                                                     
        has_autotune_silu = hasattr(silu_kernel, 'configs')                                                                               
                                                                                                                                        
        if has_autotune_silu:                                                                                                             
            grid_ew = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)                                                                  
            silu_kernel[grid_ew](x, y, n)                                                                                                 
            torch.cuda.synchronize()                                                                                                      
            ms = triton.testing.do_bench(lambda: silu_kernel[grid_ew](x, y, n))                                                           
        else:                                                                                                                             
            BLK = 1024                                                                                                                    
            grid_ew = (triton.cdiv(n, BLK),)                                                                                              
            ms = triton.testing.do_bench(lambda: silu_kernel[grid_ew](x, y, n, BLOCK_SIZE=BLK))                                           
        print(f"  silu_kernel:              {ms:.4f} ms")                                                                                 
                                                                                                                                        
        has_autotune_gelu = hasattr(gelu_kernel, 'configs')                                                                               
        if has_autotune_gelu:                                                                                                             
            grid_ew_g = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)                                                                
            gelu_kernel[grid_ew_g](x, y, n)                                                                                               
            torch.cuda.synchronize()                                                                                                      
            ms = triton.testing.do_bench(lambda: gelu_kernel[grid_ew_g](x, y, n))                                                         
        else:                                                                                                                             
            BLK = 1024                                                                                                                    
            grid_ew_g = (triton.cdiv(n, BLK),)                                                                                            
            ms = triton.testing.do_bench(lambda: gelu_kernel[grid_ew_g](x, y, n, BLOCK_SIZE=BLK))                                         
        print(f"  gelu_kernel:              {ms:.4f} ms")                                                                                 
                                                                                                                                        
        del x, y                                                                                                                          
        gc.collect()                                                                                                                      
        torch.cuda.empty_cache()                                                                                                          
                                                                                                                                        
        print("\n  --- Reduction Kernels ---")                                                                                            
                                                                                                                                        
        x_sm = torch.randn(B * N_heads_dec, S, device=device, dtype=torch.float32)                                                        
        y_sm = torch.empty_like(x_sm)                                                                                                   
        BLK_SM = triton.next_power_of_2(S)                                                                                                
        ms = triton.testing.do_bench(                                                                                                     
            lambda: softmax_kernel[(B * N_heads_dec,)](                                                                                   
                x_sm, y_sm, x_sm.stride(0), y_sm.stride(0), S, BLOCK_SIZE=BLK_SM                                                          
            )                                                                                                                             
        )                                                                                                                                 
        print(f"  softmax_kernel:           {ms:.4f} ms")                                                                                 
        del x_sm, y_sm                                                                                                                    
                                                                                                                                        
        x_rms = torch.randn(B * S, H_dec, device=device, dtype=torch.float32)                                                             
        w_rms = torch.ones(H_dec, device=device, dtype=torch.float32)                                                                     
        y_rms = torch.empty_like(x_rms)                                                                                                   
        BLK_D = triton.next_power_of_2(H_dec)                                                                                           
        ms = triton.testing.do_bench(                                                                                                     
            lambda: rmsnorm_kernel[(B * S,)](                                                                                             
                x_rms, w_rms, y_rms, x_rms.stride(0), y_rms.stride(0),                                                                    
                H_dec, 1e-6, BLOCK_SIZE=BLK_D,                                                                                            
            )                                                                                                                             
        )                                                                                                                                 
        print(f"  rmsnorm_kernel:           {ms:.4f} ms")                                                                                 
                                                                                                                                        
        x_ln = torch.randn(B * S, H_enc, device=device, dtype=torch.float32)                                                              
        w_ln = torch.ones(H_enc, device=device, dtype=torch.float32)                                                                      
        b_ln = torch.zeros(H_enc, device=device, dtype=torch.float32)                                                                     
        y_ln = torch.empty_like(x_ln)                                                                                                     
        BLK_E = triton.next_power_of_2(H_enc)                                                                                             
        ms = triton.testing.do_bench(                                                                                                     
            lambda: layernorm_kernel[(B * S,)](                                                                                           
                x_ln, w_ln, b_ln, y_ln, x_ln.stride(0), y_ln.stride(0),                                                                   
                H_enc, 1e-5, BLOCK_SIZE=BLK_E,                                                                                            
            )                                                                                                                             
        )                                                                                                                                 
        print(f"  layernorm_kernel:         {ms:.4f} ms")                                                                                 
                                                                                                                                        
        del x_rms, w_rms, y_rms, x_ln, w_ln, b_ln, y_ln                                                                                 
        gc.collect()                                                                                                                      
        torch.cuda.empty_cache()                                                                                                        
                                                                                                                                        
        print("\n  --- Tiled Matmul ---")                                                                                                 
                                                                                                                                        
        M, K, N = B * S, H_dec, I_dec                                                                                                     
        BM, BN, BK = 64, 64, 32                                                                                                         
        M_p = pad_to_multiple(M, BM)                                                                                                      
        K_p = pad_to_multiple(K, BK)                                                                                                      
        N_p = pad_to_multiple(N, BN)                                                                                                      
        a = torch.randn(M_p, K_p, device=device, dtype=torch.float32)                                                                     
        b_mat = torch.randn(K_p, N_p, device=device, dtype=torch.float32)                                                                 
        c = torch.zeros(M_p, N_p, device=device, dtype=torch.float32)                                                                     
        grid_mm = (triton.cdiv(M_p, BM), triton.cdiv(N_p, BN))                                                                            
        ms = triton.testing.do_bench(                                                                                                     
            lambda: linear_kernel_tf32[grid_mm](                                                                                          
                a, b_mat, c, M_p, N_p, K_p,                                                                                               
                a.stride(0), a.stride(1),                                                                                                 
                b_mat.stride(0), b_mat.stride(1),                                                                                         
                c.stride(0), c.stride(1),                                                                                                 
                BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,                                                                                       
            )                                                                                                                             
        )                                                                                                                                 
        print(f"  linear_kernel_tf32:       {ms:.4f} ms  (M={B*S}, K={H_dec}, N={I_dec})")                                                
                                                                                                                                        
        del a, b_mat, c                                                                                                                   
        gc.collect()                                                                                                                      
        torch.cuda.empty_cache()                                                                                                          
                                                                                                                                        
        print("\n  --- Fused MLP Kernels ---")                                                                                            
                                                                                                                                        
        try:                                                                                                                              
            from layers import swiglu_fused_kernel                                                                                      
            M, K, N = B * S, H_dec, I_dec                                                                                                 
            BM, BN, BK = 128, 128, 32                                                                                                     
            M_p = pad_to_multiple(M, BM)                                                                                                  
            K_p = pad_to_multiple(K, BK)                                                                                                  
            N_p = pad_to_multiple(N, BN)                                                                                                  
                                                                                                                                        
            a_sw = torch.randn(M_p, K_p, device=device, dtype=torch.float32)                                                              
            gate_w = torch.randn(K_p, N_p, device=device, dtype=torch.float32)                                                            
            up_w = torch.randn(K_p, N_p, device=device, dtype=torch.float32)                                                              
            c_sw = torch.zeros(M_p, N_p, device=device, dtype=torch.float32)                                                              
                                                                                                                                        
            has_autotune_sw = hasattr(swiglu_fused_kernel, 'configs')                                                                     
            if has_autotune_sw:                                                                                                           
                grid_sw = lambda meta: (                                                                                                  
                    triton.cdiv(M_p, meta["BLOCK_M"]),                                                                                    
                    triton.cdiv(N_p, meta["BLOCK_N"]),                                                                                    
                )                                                                                                                         
                swiglu_fused_kernel[grid_sw](                                                                                             
                    a_sw, gate_w, up_w, c_sw,                                                                                             
                    M_p, N_p, K_p,                                                                                                        
                    a_sw.stride(0), a_sw.stride(1),                                                                                       
                    gate_w.stride(0), gate_w.stride(1),                                                                                   
                    up_w.stride(0), up_w.stride(1),                                                                                       
                    c_sw.stride(0), c_sw.stride(1),                                                                                       
                )                                                                                                                         
                torch.cuda.synchronize()                                                                                                  
                ms = triton.testing.do_bench(                                                                                             
                    lambda: swiglu_fused_kernel[grid_sw](                                                                                 
                        a_sw, gate_w, up_w, c_sw,                                                                                         
                        M_p, N_p, K_p,                                                                                                    
                        a_sw.stride(0), a_sw.stride(1),                                                                                   
                        gate_w.stride(0), gate_w.stride(1),                                                                               
                        up_w.stride(0), up_w.stride(1),                                                                                   
                        c_sw.stride(0), c_sw.stride(1),                                                                                   
                    )                                                                                                                     
                )                                                                                                                         
            else:                                                                                                                         
                grid_sw = (triton.cdiv(M_p, BM), triton.cdiv(N_p, BN))                                                                    
                ms = triton.testing.do_bench(                                                                                             
                    lambda: swiglu_fused_kernel[grid_sw](                                                                                 
                        a_sw, gate_w, up_w, c_sw,                                                                                         
                        M_p, N_p, K_p,                                                                                                    
                        a_sw.stride(0), a_sw.stride(1),                                                                                   
                        gate_w.stride(0), gate_w.stride(1),                                                                               
                        up_w.stride(0), up_w.stride(1),                                                                                   
                        c_sw.stride(0), c_sw.stride(1),                                                                                   
                        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,                                                                               
                    )                                                                                                                     
                )                                                                                                                         
            print(f"  swiglu_fused_kernel:      {ms:.4f} ms  (M={B*S}, K={H_dec}, N={I_dec})")                                            
                                                                                                                                        
            del a_sw, gate_w, up_w, c_sw                                                                                                  
            gc.collect()                                                                                                                  
            torch.cuda.empty_cache()                                                                                                      
        except (ImportError, Exception) as e:                                                                                             
            print(f"  swiglu_fused_kernel:      N/A ({e})")                                                                               
                                                                                                                                        
        try:                                                                                                                              
            from layers import linear_gelu_kernel                                                                                         
            M, K, N = B * S, H_enc, I_enc                                                                                                 
            M_p = pad_to_multiple(M, 64)                                                                                                  
            K_p = pad_to_multiple(K, 32)                                                                                                  
            N_p = pad_to_multiple(N, 64)                                                                                                  
                                                                                                                                        
            a_lg = torch.randn(M_p, K_p, device=device, dtype=torch.float32)                                                              
            b_lg = torch.randn(K_p, N_p, device=device, dtype=torch.float32)                                                              
            c_lg = torch.zeros(M_p, N_p, device=device, dtype=torch.float32)                                                              
                                                                                                                                        
            grid_lg = (triton.cdiv(M_p, 64), triton.cdiv(N_p, 64))                                                                        
            ms = triton.testing.do_bench(                                                                                                 
                lambda: linear_gelu_kernel[grid_lg](                                                                                      
                    a_lg, b_lg, c_lg,                                                                                                     
                    M_p, N_p, K_p,                                                                                                        
                    a_lg.stride(0), a_lg.stride(1),                                                                                       
                    b_lg.stride(0), b_lg.stride(1),                                                                                       
                    c_lg.stride(0), c_lg.stride(1),                                                                                       
                    BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,                                                                                   
                )                                                                                                                         
            )                                                                                                                             
            print(f"  linear_gelu_kernel:       {ms:.4f} ms  (M={B*S}, K={H_enc}, N={I_enc})")                                            
                                                                                                                                        
            del a_lg, b_lg, c_lg                                                                                                          
            gc.collect()                                                                                                                  
            torch.cuda.empty_cache()                                                                                                      
        except (ImportError, Exception) as e:                                                                                             
            print(f"  linear_gelu_kernel:       N/A ({e})")
                                                                                                                                        
        print("\n  --- Attention Kernels ---")                                                                                            
                                                                                                                                        
        HD = HD_dec                                                                                                                       
        q = torch.randn(B * N_heads_dec, S, HD, device=device, dtype=torch.float32)                                                     
        k = torch.randn(B * N_heads_dec, S, HD, device=device, dtype=torch.float32)                                                       
        v = torch.randn(B * N_heads_dec, S, HD, device=device, dtype=torch.float32)                                                       
        scores = torch.zeros(B * N_heads_dec, S, S, device=device, dtype=torch.float32)                                                   
        out = torch.zeros_like(q)                                                                                                         
                                                                                                                                        
        BLK_K = triton.next_power_of_2(S)                                                                                                 
        BLK_HD = triton.next_power_of_2(HD)                                                                                               
        scale = 1.0 / (HD ** 0.5)                                                                                                         
                                                                                                                                        
        ms = triton.testing.do_bench(                                                                                                     
            lambda: attention_scores_kernel[(B * N_heads_dec, S)](                                                                        
                q, k, scores, scale, S, HD,                                                                                               
                q.stride(0), q.stride(1), q.stride(2),                                                                                    
                k.stride(0), k.stride(1), k.stride(2),                                                                                    
                scores.stride(0), scores.stride(1), scores.stride(2),                                                                     
                BLOCK_K=BLK_K, BLOCK_D=BLK_HD,                                                                                            
            )                                                                                                                             
        )                                                                                                                                 
        print(f"  attention_scores_kernel:  {ms:.4f} ms")                                                                                 
                                                                                                                                        
        ms = triton.testing.do_bench(                                                                                                     
            lambda: softmax_inplace_kernel[(B * N_heads_dec * S,)](                                                                       
                scores, scores.stride(1), S, BLOCK_SIZE=BLK_K,                                                                            
            )                                                                                                                             
        )                                                                                                                                 
        print(f"  softmax_inplace_kernel:   {ms:.4f} ms")                                                                                 
                                                                                                                                        
        ms = triton.testing.do_bench(                                                                                                     
            lambda: attention_output_kernel[(B * N_heads_dec, S)](                                                                        
                scores, v, out, S, HD,                                                                                                    
                scores.stride(0), scores.stride(1), scores.stride(2),                                                                     
                v.stride(0), v.stride(1), v.stride(2),                                                                                    
                out.stride(0), out.stride(1), out.stride(2),                                                                              
                BLOCK_K=BLK_K, BLOCK_D=BLK_HD,                                                                                            
            )                                                                                                                             
        )                                                                                                                                 
        print(f"  attention_output_kernel:  {ms:.4f} ms")                                                                                 
                                                                                                                                        
        del scores                                                                                                                        
        gc.collect()                                                                                                                      
        torch.cuda.empty_cache()                                                                                                          
                                                                                                                                        
        try:                                                                                                                              
            from attention import fused_attention_kernel                                                                                  
            S_pad = triton.next_power_of_2(S)                                                                                             
            q_f = torch.randn(B * N_heads_dec, S, HD, device=device, dtype=torch.float32)                                                 
            k_f = torch.zeros(B * N_heads_dec, S_pad, BLK_HD, device=device, dtype=torch.float32)                                         
            v_f = torch.zeros(B * N_heads_dec, S_pad, BLK_HD, device=device, dtype=torch.float32)                                         
            o_f = torch.zeros(B * N_heads_dec, S, BLK_HD, device=device, dtype=torch.float32)                                             
                                                                                                                                        
            k_f[:, :S, :HD] = k                                                                                                           
            v_f[:, :S, :HD] = v                                                                                                           
                                                                                                                                        
            grid_fa = lambda meta: (                                                                                                      
                triton.cdiv(S, meta["TILE_M"]),                                                                                           
                B * N_heads_dec,                                                                                                          
            )                                                                                                                             
            fused_attention_kernel[grid_fa](                                                                                              
                q_f, k_f, v_f, o_f,                                                                                                       
                float(scale), S, S, HD,                                                                                                   
                *q_f.stride(), *k_f.stride(), *v_f.stride(), *o_f.stride(),                                                               
                IS_CAUSAL=True, SEQ_K=S_pad, BLOCK_D=BLK_HD,                                                                              
            )                                                                                                                             
            torch.cuda.synchronize()                                                                                                      
            ms = triton.testing.do_bench(                                                                                                 
                lambda: fused_attention_kernel[grid_fa](                                                                                  
                    q_f, k_f, v_f, o_f,                                                                                                   
                    float(scale), S, S, HD,                                                                                               
                    *q_f.stride(), *k_f.stride(), *v_f.stride(), *o_f.stride(),                                                           
                    IS_CAUSAL=True, SEQ_K=S_pad, BLOCK_D=BLK_HD,                                                                          
                )                                                                                                                         
            )                                                                                                                             
            print(f"  flash_attention_kernel:   {ms:.4f} ms")                                                                             
                                                                                                                                        
            del q_f, k_f, v_f, o_f                                                                                                        
        except (ImportError, Exception) as e:                                                                                             
            print(f"  flash_attention_kernel:   N/A ({e})")                                                                               
                                                                                                                                        
        del q, k, v, out                                                                                                                  
        gc.collect()                                                                                                                      
        torch.cuda.empty_cache()                                                                                                          
                                                                                                                                        
        print("\n  --- RoPE Kernels ---")                                                                                                 
                                                                                                                                        
        half_dim = HD_dec // 2                                                                                                            
        positions = torch.arange(S, device=device, dtype=torch.float32)                                                                 
        inv_freq = torch.randn(half_dim, device=device, dtype=torch.float32)                                                              
        cos_cache = torch.empty(S, HD_dec, device=device, dtype=torch.float32)                                                            
        sin_cache = torch.empty(S, HD_dec, device=device, dtype=torch.float32)                                                            
        BLK_F = triton.next_power_of_2(half_dim)                                                                                          
        ms = triton.testing.do_bench(                                                                                                     
            lambda: compute_freqs_kernel[(S,)](                                                                                           
                positions, inv_freq, cos_cache, sin_cache,                                                                                
                S, half_dim,                                                                                                              
                positions.stride(0), inv_freq.stride(0),                                                                                  
                cos_cache.stride(0), cos_cache.stride(1),                                                                                 
                sin_cache.stride(0), sin_cache.stride(1),                                                                                 
                BLOCK=BLK_F, num_warps=2,                                                                                                 
            )                                                                                                                             
        )                                                                                                                                 
        print(f"  compute_freqs_kernel:     {ms:.4f} ms")                                                                                 
                                                                                                                                        
        try:                                                                                                                              
            from rope import apply_rope_kernel                                                                                            
            x_rope = torch.randn(B, N_heads_dec, S, HD_dec, device=device, dtype=torch.float32)                                           
            o_rope = torch.empty_like(x_rope)                                                                                             
            cos_s = cos_cache[:, :half_dim].contiguous()                                                                                  
            sin_s = sin_cache[:, :half_dim].contiguous()                                                                                  
            BLK_ROPE = triton.next_power_of_2(HD_dec)                                                                                     
            ms = triton.testing.do_bench(                                                                                                 
                lambda: apply_rope_kernel[B * N_heads_dec * S,](                                                                          
                    x_rope, cos_s, sin_s, o_rope,                                                                                         
                    N_heads_dec, S, half_dim, HD_dec,                                                                                     
                    x_rope.stride(0), x_rope.stride(1), x_rope.stride(2), x_rope.stride(3),                                               
                    cos_s.stride(0), cos_s.stride(1),                                                                                     
                    o_rope.stride(0), o_rope.stride(1), o_rope.stride(2), o_rope.stride(3),                                               
                    BLOCK_D=BLK_ROPE, num_warps=4,                                                                                        
                )                                                                                                                         
            )                                                                                                                             
            print(f"  apply_rope_kernel:        {ms:.4f} ms")                                                                             
        except (ImportError, Exception) as e:                                                                                             
            print(f"  apply_rope_kernel:        N/A ({e})")                                                                               
                                                                                                                                        
    print("\n" + "=" * 70)                                                                                                                
    print("DONE")                                                                                                                         
    print("=" * 70)                                                                                                                       
                                                                                                                                        
                                                                                                                                        
if __name__ == "__main__":                                                                                                                
    main()      