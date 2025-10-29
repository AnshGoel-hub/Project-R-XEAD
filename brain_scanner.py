# brain_scanner.py
import torch
import numpy as np

class BrainScanner:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.hidden_states = []
        # We pick a layer "deep" in the model, e.g., Layer 20
        # Note: Layer names can vary. 'model.layers.19' is common for Mistral/Llama.
        self.layer_name = 'model.layers.19' 
        self.register_hook()

    def _hook_fn(self, module, input, output):
        # 'output' is the hidden state tensor. (Fixed from output[0])
        # We detach it from the computation graph and move to CPU.
        self.hidden_states.append(output.detach().cpu().numpy())

    def register_hook(self):
        # Find the specific layer module to attach the hook to
        layer = None
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                layer = module
                break
        
        if layer is None:
            # Try to find a layer if the name is slightly different
            found_layers = [name for name, _ in self.model.named_modules() if 'model.layers.19' in name]
            if found_layers:
                self.layer_name = found_layers[0]
                for name_check, module_check in self.model.named_modules():
                    if name_check == self.layer_name:
                        layer = module_check
                        break
            if layer is None:
                print(f"Warning: Layer {self.layer_name} not found.")
                print("Available layers include:")
                for name, _ in self.model.named_modules():
                    if 'model.layers' in name:
                        print(name)
                raise Exception(f"Layer {self.layer_name} not found in model.")

        # Attach the forward hook
        hook = layer.register_forward_hook(self._hook_fn)
        self.hooks.append(hook)
        print(f"--- Hook attached to {self.layer_name} ---")

    def clear_states(self):
        # Clear states after processing one prompt
        self.hidden_states = []

    def remove_hooks(self):
        # Clean up hooks when done
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        print("--- Hooks removed ---")

    def get_chaos_scores_from_states(self):
        if not self.hidden_states:
            return None
        final_layer_state = self.hidden_states[-1]
        last_token_vector = final_layer_state[0, -1, :] 

        magnitude = np.linalg.norm(last_token_vector)
        confusion = np.var(last_token_vector) # Use variance or std, var is fine

        return {"magnitude": magnitude, "confusion": confusion}