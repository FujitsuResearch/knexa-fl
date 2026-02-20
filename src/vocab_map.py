import torch

def sparse_to_dense(values, indices, teacher_tok, student_tok):
    """
    Convert sparse teacher logits to dense student vocab space.
    
    Args:
        values: Top-k logit values from teacher [batch, seq, k]
        indices: Top-k token indices from teacher [batch, seq, k] 
        teacher_tok: Teacher tokenizer
        student_tok: Student tokenizer
        
    Returns:
        Dense logits in student vocab space [batch, seq, student_vocab_size]
    """
    batch_size, seq_len, k = values.shape
    student_vocab_size = student_tok.vocab_size
    
    # Initialize dense tensor with half precision for memory efficiency
    # Will be converted to float32 later if needed
    dtype = torch.float16 if values.dtype == torch.float16 else torch.float32
    dense = torch.zeros(batch_size, seq_len, student_vocab_size, device=values.device, dtype=dtype)
    
    # Pre-compute token mappings for efficiency
    teacher_pad_id = teacher_tok.pad_token_id
    student_unk_id = student_tok.unk_token_id
    
    for b in range(batch_size):
        for s in range(seq_len):
            teacher_indices = indices[b, s]
            teacher_values = values[b, s]
            
            # Map teacher tokens to student vocab
            for i, (t_idx, t_val) in enumerate(zip(teacher_indices, teacher_values)):
                if t_idx == teacher_pad_id:
                    continue
                    
                # Convert teacher token ID to text then to student token ID
                try:
                    token_text = teacher_tok.convert_ids_to_tokens(t_idx.item())
                    if isinstance(token_text, list):
                        token_text = token_text[0] if token_text else teacher_tok.unk_token
                    
                    student_idx = student_tok.convert_tokens_to_ids(token_text)
                    if student_idx != student_unk_id:
                        dense[b, s, student_idx] = t_val
                except:
                    # Fallback: map to UNK token
                    if student_unk_id is not None:
                        dense[b, s, student_unk_id] = t_val
    
    return dense