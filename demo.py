# demo.py (R-XEAD v4.2 - FINAL 3-Layered Defense)
import torch
import joblib
import numpy as np
from model_loader import model, tokenizer
from brain_scanner import BrainScanner
from transformers import AutoTokenizer as AutoTokenizerForSentry, AutoModelForSequenceClassification, TextClassificationPipeline

# --- 1. LOAD LAYER 2 (R-XEAD / CHAOS GUARD) ---
# This is your new 2-feature model, trained ONLY on jailbreaks
print("Loading Layer 2 (R-XEAD Chaos Guard)...")
try:
    guard_model_chaos = joblib.load('guard_model.pkl')
    scaler_chaos = joblib.load('scaler.pkl')
    print("R-XEAD models loaded.")
except FileNotFoundError:
    print("Error: 'guard_model.pkl' or 'scaler.pkl' not found.")
    print("Please run the 'Phase3_Training.ipynb' notebook first!")
    exit()

# --- 2. LOAD LAYER 1 (SENTRY / CONTENT GUARD) ---
print("Loading Layer 1 (Sentry Content Guard)...")
sentry_model_id = "Vrandan/Comment-Moderation" #
sentry_tokenizer = AutoTokenizerForSentry.from_pretrained(sentry_model_id)
sentry_model = AutoModelForSequenceClassification.from_pretrained(sentry_model_id)
sentry_pipeline = TextClassificationPipeline(model=sentry_model, tokenizer=sentry_tokenizer, top_k=None)
print("Sentry model loaded.")

# --- 3. DEFINE LAYER 0 (CONTEXTUAL KEYWORD FILTER) ---
# This catches "grey area" topics the Sentry AI might miss.
BANNED_KEYWORDS = { "weed", "marijuana", "cocaine", "heroin", "methamphetamine" }
# This is the "allow-list" you asked for, to provide context
CONTEXT_ALLOW_LIST = { "novel", "plot", "scene", "dialogue", "scientific", "research", "context" }
print("Layer 0 (Contextual Keyword Filter) is active.")

# --- 4. INITIALIZE BRAIN SCANNER (for Layer 2) ---
scanner = BrainScanner(model)

# --- 5. THE LIVE LOOP ---
print("\n--- Project R-XEAD v4.2 is Active (3-Layer Defense) ---")
print("Enter a prompt. Type 'exit' to quit.")

while True:
    prompt_text = input("\nUser > ")
    if prompt_text.lower() == 'exit':
        break

    prompt_lower = prompt_text.lower()

    # --- LAYER 0 CHECK (CONTEXTUAL KEYWORD FILTER) ---
    print("Bot > [Layer 0] Running keyword check...")
    found_banned_keyword = False
    found_context_keyword = False

    for keyword in BANNED_KEYWORDS:
        if keyword in prompt_lower:
            found_banned_keyword = True
            break
            
    if found_banned_keyword:
        for context_word in CONTEXT_ALLOW_LIST:
            if context_word in prompt_lower:
                found_context_keyword = True
                break
    
    if found_banned_keyword and not found_context_keyword:
        print("\n>>> PROJECT R-XEAD INTERVENTION (LAYER 0: KEYWORD) <<<")
        print("ACCESS DENIED. Harmful content detected.")
        print(f"EXPLANATION: Prompt contains banned keyword: '{keyword}' without a safe context.")
        continue # Stop and ask for a new prompt
    else:
        print("Bot > [Layer 0] Keyword check passed.")

    # --- LAYER 1 CHECK (SENTRY / CONTENT) ---
    print("Bot > [Layer 1] Running content check...")
    sentry_results = sentry_pipeline(prompt_text)[0]
    top_label = sentry_results[0]['label']
    top_score = sentry_results[0]['score']
    print(f"Bot > [Layer 1] Sentry Top Result: {top_label} (Score: {top_score:.4f})")
    
    if top_label != 'OK': # 'OK' is the safe label
        print("\n>>> PROJECT R-XEAD INTERVENTION (LAYER 1: SENTRY) <<<")
        print("ACCESS DENIED. Direct harmful content detected.")
        print(f"EXPLANATION: Content flagged as '{top_label}'. (Conf: {top_score * 100:.2f}%)")
        continue 
    else:
        print("Bot > [Layer 1] Content check passed.")

    # --- LAYER 2 CHECK (R-XEAD / CHAOS) ---
    print("Bot > [Layer 2] Running chaos check...")
    scanner.clear_states()
    try:
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        scores = scanner.get_chaos_scores_from_states()
    except Exception as e:
        print(f"Bot > Model Error: {e}")
        continue

    if not scores:
        print("Bot > Could not process prompt.")
        continue

    # --- THIS IS THE FIX ---
    # We are back to a 2-feature array
    scores_array = np.array([[ scores['magnitude'], scores['confusion'] ]])
    
    scores_scaled = scaler_chaos.transform(scores_array)
    prediction_chaos = guard_model_chaos.predict(scores_scaled)
    
    if prediction_chaos[0] == 1: # 1 = Hijacked
        print("\n>>> PROJECT R-XEAD INTERVENTION (LAYER 2: R-XEAD) <<<")
        print("ACCESS DENIED. Adversarial jailbreak pattern detected.")
        print("EXPLANATION: Internal Chaos Score spiked.")
        continue
    else:
        print("Bot > [Layer 2] Chaos check passed.")

    # --- ALLOW (ALL 3 CHEKS PASSED) ---
    print("Bot > (All layers clear. Prompt seems safe.) Generating response...")
    
    formatted_prompt = f"<s>[INST] {prompt_text} [/INST]"
    generation_inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**generation_inputs, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id, do_sample=False)
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:
        answer = full_response.split("[/INST]")[1].strip()
        print(f"Bot > {answer}")
    except IndexError:
        print(f"Bot > (Model generated an unusual response): {full_response}")

# Cleanup
scanner.remove_hooks()
print("--- Project R-XEAD Deactivated ---")