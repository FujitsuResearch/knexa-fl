#!/usr/bin/env python3
"""
Robust token transfer function for teacher-student knowledge distillation.
"""

import torch
from torch.nn.utils.rnn import pad_sequence


def transfer_tokens_safely(teacher_tokens, teacher_tokenizer, student_tokenizer, target_device="cpu"):
    """
    Safely transfer tokens from teacher to student model.
    
    Args:
        teacher_tokens: Teacher model token output (dict with 'input_ids', 'attention_mask') or tensor
        teacher_tokenizer: Teacher tokenizer
        student_tokenizer: Student tokenizer  
        target_device: Target device for output tokens
        
    Returns:
        dict: Student-compatible tokens on target device
    """
    
    # Extract input_ids properly - handle BatchEncoding and dict
    if hasattr(teacher_tokens, 'input_ids'):  # BatchEncoding or similar
        input_ids = teacher_tokens.input_ids
        attention_mask = getattr(teacher_tokens, 'attention_mask', None)
    elif isinstance(teacher_tokens, dict):
        if 'input_ids' in teacher_tokens:
            input_ids = teacher_tokens['input_ids']
            attention_mask = teacher_tokens.get('attention_mask', None)
        else:
            raise ValueError("Dictionary must contain 'input_ids' key")
    elif torch.is_tensor(teacher_tokens):
        input_ids = teacher_tokens
        attention_mask = None
    else:
        raise ValueError(f"Unsupported teacher_tokens type: {type(teacher_tokens)}")
    
    # Ensure we have a tensor
    if not torch.is_tensor(input_ids):
        raise ValueError(f"input_ids must be a tensor, got {type(input_ids)}")
    
    # Move to CPU for processing
    input_ids_cpu = input_ids.cpu()
    if attention_mask is not None:
        attention_mask_cpu = attention_mask.cpu()
    
    # Approach 1: Re-tokenization (most robust)
    try:
        batch_size = input_ids_cpu.shape[0]
        student_tokens_list = []
        
        for i in range(batch_size):
            # Decode single sequence
            decoded_text = teacher_tokenizer.decode(
                input_ids_cpu[i], 
                skip_special_tokens=True
            )
            
            # Re-encode with student tokenizer
            student_encoded = student_tokenizer(
                decoded_text,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=input_ids_cpu.shape[1]  # Match original length
            )
            
            student_tokens_list.append(student_encoded['input_ids'])
        
        # Combine batch and pad
        padded_tokens = pad_sequence(
            [tokens.squeeze(0) for tokens in student_tokens_list],
            batch_first=True,
            padding_value=student_tokenizer.pad_token_id
        )
        
        # Create attention mask
        attention_mask_new = (padded_tokens != student_tokenizer.pad_token_id).long()
        
        result = {
            'input_ids': padded_tokens.to(target_device),
            'attention_mask': attention_mask_new.to(target_device)
        }
        
        return result
        
    except Exception as e:
        print(f"Re-tokenization failed: {e}, falling back to device transfer")
        
        # Approach 2: Direct transfer with bounds checking
        # Check bounds and clip if necessary
        out_of_bounds_mask = input_ids_cpu >= len(student_tokenizer)
        if out_of_bounds_mask.any():
            print(f"Warning: Found {out_of_bounds_mask.sum().item()} out-of-bounds tokens, clipping to UNK")
            input_ids_cpu = input_ids_cpu.clone()  # Don't modify original
            input_ids_cpu[out_of_bounds_mask] = student_tokenizer.unk_token_id
            
        result = {'input_ids': input_ids_cpu.to(target_device)}
        
        if attention_mask is not None:
            result['attention_mask'] = attention_mask_cpu.to(target_device)
        else:
            result['attention_mask'] = (input_ids_cpu != student_tokenizer.pad_token_id).long().to(target_device)
            
        return result


def simple_device_transfer(teacher_tokens, target_device="cpu"):
    """
    Simple device transfer without re-tokenization.
    Use this when you're sure tokens are compatible.
    """
    if isinstance(teacher_tokens, dict):
        return {k: v.to(target_device) for k, v in teacher_tokens.items()}
    elif torch.is_tensor(teacher_tokens):
        return teacher_tokens.to(target_device)
    else:
        raise ValueError(f"Unsupported type: {type(teacher_tokens)}")


def check_token_compatibility(teacher_tokens, student_vocab_size):
    """
    Check if teacher tokens are compatible with student vocabulary.
    
    Returns:
        bool: True if all tokens are valid for student
        list: List of invalid token IDs if any
    """
    if hasattr(teacher_tokens, 'input_ids'):  # BatchEncoding or similar
        input_ids = teacher_tokens.input_ids
    elif isinstance(teacher_tokens, dict):
        input_ids = teacher_tokens['input_ids']
    else:
        input_ids = teacher_tokens
        
    input_ids_cpu = input_ids.cpu()
    invalid_mask = input_ids_cpu >= student_vocab_size
    
    if invalid_mask.any():
        invalid_tokens = input_ids_cpu[invalid_mask].unique().tolist()
        return False, invalid_tokens
    else:
        return True, []
