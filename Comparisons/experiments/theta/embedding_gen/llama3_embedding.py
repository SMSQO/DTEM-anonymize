import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

class Llama3Embedder:
    '''
    Input a piece of text, 
    returns the embedding of it. 

    The text can be either NL or CL.
    '''

    def __init__(self, device='cuda:0'): 
        self.device = device

    def _ensure_model_loaded(self): 
        '''
        Technical details.
        Not important. 
        '''
        if hasattr(self, "model"): 
            return 
        
        model_id = "meta-llama/Meta-Llama-3-8B"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        config_kwargs = { "output_hidden_states": True }
        config = AutoConfig.from_pretrained(model_id, **config_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            config=config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            revision='main',
            device_map=self.device
        )

        Llama3Embedder.model = model
        Llama3Embedder.tokenizer = tokenizer

    def get_text_embedding(self, text: str) -> np.ndarray: 
        self._ensure_model_loaded()
        model, tokenizer = self.model, self.tokenizer

        with torch.no_grad(): 
            tokens = tokenizer.encode_plus(
                    text, 
                    truncation=True,
                    max_length=1024,
                    return_tensors='pt'
            )

            mask = tokens['attention_mask']
            outputs = model(input_ids=tokens['input_ids'].cuda(self.device), attention_mask=mask.cuda(self.device))
            hidden_states = list(outputs.hidden_states)
            first_hidden_state = hidden_states[ 0].cpu().numpy()
            last_hidden_state  = hidden_states[-1].cpu().numpy()
            # first_hidden_state: Tensor<1,5,4096>

            ret = np.squeeze(first_hidden_state) \
                + np.squeeze(last_hidden_state)
            ret: np.ndarray = np.mean(ret, axis=0)
            
            return ret
    
    def __call__(self, text: str) -> np.ndarray: 
        return self.get_text_embedding(text)

    