# Diagnostic Output

[STAGE 1/4] Loading tokenizer...
  ✓ Tokenizer loaded

[STAGE 2/4] Loading dataset to detect channels...
  Loading EEG files and aligning with text (this may take 2-10 minutes)...
    Loading text sentences...
    ✓ Loaded 1195 text sentences
    Processing 116 EEG files across 12 subjects...
    ✓ Processed 116 files, extracted 116 aligned samples
  ✓ Detected 105 channels from ZuCo data
  ✓ Loaded 12 samples from ZuCo 1.0 (val split) in 75.0 seconds
  ⚠ Config has 64 channels, but data has 105
  ✓ Updating config to use 105 channels
  ⚠ Config vocab_size (10000) != tokenizer vocab_size (30522)
  ✓ Updating config to use tokenizer vocab_size (30522)

[STAGE 3/4] Creating model...
[STAGE 4/4] Loading checkpoint: checkpoints/best_model.pt
  Attempting flexible checkpoint loading (strict=False)...
  ⚠ 201 missing keys (architecture changed):
    - frozen_text_model.embeddings.word_embeddings.weight
    - frozen_text_model.embeddings.position_embeddings.weight
    - frozen_text_model.embeddings.token_type_embeddings.weight
    - frozen_text_model.embeddings.LayerNorm.weight
    - frozen_text_model.embeddings.LayerNorm.bias
    - frozen_text_model.encoder.layer.0.attention.self.query.weight
    - frozen_text_model.encoder.layer.0.attention.self.query.bias
    - frozen_text_model.encoder.layer.0.attention.self.key.weight
    - frozen_text_model.encoder.layer.0.attention.self.key.bias
    - frozen_text_model.encoder.layer.0.attention.self.value.weight
    ... and 191 more
  ⚠ 4 unexpected keys (old architecture):
    - text_encoder.0.weight
    - text_encoder.0.bias
    - text_encoder.3.weight
    - text_encoder.3.bias
  ✓ Checkpoint loaded with 201 missing and 4 unexpected keys
  ⚠ Model architecture changed since checkpoint was saved
  ⚠ Diagnostics may not reflect actual trained model behavior
  ✓ Model ready for diagnostics
======================================================================
DIAGNOSTIC CHECKS FOR ZERO BLEU/ROUGE
======================================================================
  [DataLoader] Processing sample 11/12 (preprocessing + frequency extraction)...
  [DataLoader] Processing sample 2/12 (preprocessing + frequency extraction) (ETA: 30s)...
  [DataLoader] Processing sample 1/12 (preprocessing + frequency extraction)...
  [DataLoader] Processing sample 5/12 (preprocessing + frequency extraction) (ETA: 20s)...
  [DataLoader] Processing sample 4/12 (preprocessing + frequency extraction) (ETA: 36s)...

[CHECK 0.1] Printing decoded predictions and references...
----------------------------------------------------------------------
  [WARNING] Truncating 1 sequences from 159384 to 20000 time steps to avoid memory issues

Sample 1:
  Reference IDs[:30]: [101, 4463, 6986, 7277, 1998, 4097, 5622, 14517, 2050, 2191, 2005, 2028, 21459, 2135, 3459, 3940, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]      
  Predicted IDs[:30]: [101, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010]
  Reference text: 'Jason Patric and Ray Liotta make for one splendidly cast pair.'    
  Reference decoded: 'jason patric and ray liotta make for one splendidly cast pair.' 
  Predicted text: ',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,'
  Ref length: 11 tokens
  Pred length: 1 tokens
  Ref ID length: 128 tokens
  Pred ID length: 128 tokens
  ⚠️  RED FLAG: Prediction is very short: 1 tokens

  [DEBUG] Why model predicts padding:
    - Model is untrained (checkpoint architecture mismatch)
    - Untrained models predict padding token (0) as most likely
    - Generation: [CLS] → argmax → padding (0) → padding (0) → ...
    - Solution: Retrain OR mask pad_token_id during generation
  [WARNING] Truncating 1 sequences from 279138 to 20000 time steps to avoid memory issues

Sample 2:
  Reference IDs[:30]: [101, 1037, 2613, 3453, 1011, 1011, 6047, 1010, 6057, 1010, 11259, 1010, 1998, 24501, 7856, 3372, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]      
  Predicted IDs[:30]: [101, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010]
  Reference text: 'A real winner -- smart, funny, subtle, and resonant.'
  Reference decoded: 'a real winner - - smart, funny, subtle, and resonant.'
  Predicted text: ',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,'
  Ref length: 9 tokens
  Pred length: 1 tokens
  Ref ID length: 128 tokens
  Pred ID length: 128 tokens
  ⚠️  RED FLAG: Prediction is very short: 1 tokens
  [WARNING] Truncating 1 sequences from 271788 to 20000 time steps to avoid memory issues

Sample 3:
  Reference IDs[:30]: [101, 4463, 6986, 7277, 1998, 4097, 5622, 14517, 2050, 2191, 2005, 2028, 21459, 2135, 3459, 3940, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]      
  Predicted IDs[:30]: [101, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010]
  Reference text: 'Jason Patric and Ray Liotta make for one splendidly cast pair.'    
  Reference decoded: 'jason patric and ray liotta make for one splendidly cast pair.' 
  Predicted text: ',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,'
  Ref length: 11 tokens
  Pred length: 1 tokens
  Ref ID length: 128 tokens
  Pred ID length: 128 tokens
  ⚠️  RED FLAG: Prediction is very short: 1 tokens
  [WARNING] Truncating 1 sequences from 124406 to 20000 time steps to avoid memory issues

Sample 4:
  Reference IDs[:30]: [101, 2023, 2003, 2062, 1037, 2553, 1997, 1036, 17266, 2890, 1038, 2571, 2226, 999, 1005, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  Predicted IDs[:30]: [101, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010]
  Reference text: 'This is more a case of `Sacre bleu!''
  Reference decoded: 'this is more a case of ` sacre bleu! ''
  Predicted text: ',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,'
  Ref length: 8 tokens
  Pred length: 1 tokens
  Ref ID length: 128 tokens
  Pred ID length: 128 tokens
  ⚠️  RED FLAG: Prediction is very short: 1 tokens
  [WARNING] Truncating 1 sequences from 246935 to 20000 time steps to avoid memory issues

Sample 5:
  Reference IDs[:30]: [101, 2023, 2003, 2062, 1037, 2553, 1997, 1036, 17266, 2890, 1038, 2571, 2226, 999, 1005, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  Predicted IDs[:30]: [101, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010]
  Reference text: 'This is more a case of `Sacre bleu!''
  Reference decoded: 'this is more a case of ` sacre bleu! ''
  Predicted text: ',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,'
  Ref length: 8 tokens
  Pred length: 1 tokens
  Ref ID length: 128 tokens
  Pred ID length: 128 tokens
  ⚠️  RED FLAG: Prediction is very short: 1 tokens

[CHECK 0.2] Verifying tokenizer consistency...
----------------------------------------------------------------------
  ✓ Tokenizer: BertTokenizerFast
  ✓ Vocab size: 30522
  ✓ Using tokenizer.decode() with skip_special_tokens=True

[CHECK 0.3] Checking teacher forcing/shifting...
----------------------------------------------------------------------
  ✓ Should use: targets = text_tokens[:, 1:]
  ✓ Should use: decoder_input = text_tokens[:, :-1]
  ⚠️  Please verify in train.py line 118: targets = text_tokens[:, 1:]

[CHECK 0.4] Checking loss ignore_index...
----------------------------------------------------------------------
  Current ignore_index: -100
  ⚠️  WARNING: ignore_index=-100 != pad_token_id=0
  ⚠️  Padding tokens (ID=0) will NOT be ignored in loss!

[CHECK 0.5] Checking EOS handling in generation...
----------------------------------------------------------------------
  EOS token ID: 102 ([SEP] for BERT)
  EOS found in 0/5 predictions
  ⚠️  WARNING: EOS never generated - sequences may be truncated

[CHECK 0.6] Testing BLEU/ROUGE code with identical strings...
----------------------------------------------------------------------
  Test metrics (identical strings):
    bleu_1: 100.0000
    bleu_2: 100.0000
    bleu_3: 100.0000
    bleu_4: 100.0000
    rouge1_precision: 100.0000
    rouge1_recall: 100.0000
    rouge1_F: 100.0000
    rouge2_precision: 100.0000
    rouge2_recall: 100.0000
    rouge2_F: 100.0000
    rougeL_precision: 100.0000
    rougeL_recall: 100.0000
    rougeL_F: 100.0000
  ✓ BLEU/ROUGE code works correctly

[CHECK 0.7] Checking lowercasing/cleaning consistency...
----------------------------------------------------------------------
  Reference processing: text.split()
  Candidate processing: tokenizer.decode(..., skip_special_tokens=True).split()       
  ✓ No wordpiece tokens found - decoding looks correct

[CHECK 0.8] Checking dataset split and references...
----------------------------------------------------------------------
  Dataset size: 12
  Split: val
  ✓ All references have content (checked 5 samples)

Most likely issues:
1. ✅ FIXED: Missing skip_special_tokens=True in decode()
2. ⚠️  Loss ignore_index may not match pad_token_id
3. ⚠️  Some predictions are empty or only special tokens

Next steps:
1. ✅ skip_special_tokens=True fix applied to train.py
2. ⚠️  Retrain model OR fix checkpoint loading
3. ⚠️  Consider masking pad_token_id during generation
4. ⚠️  Verify ignore_index matches pad_token_id in loss
