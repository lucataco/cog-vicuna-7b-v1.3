from cog import BasePredictor, Input
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        name = 'lmsys/vicuna-7b-v1.3'
        self.tokenizer = AutoTokenizer.from_pretrained(
            name,
            cache_dir="cache"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.bfloat16,
            cache_dir="cache"
        )

    def predict(self,
        prompt: str = Input(description="Instruction for the model"),
        max_new_tokens: int = Input(description="max tokens to generate", default=500)
    ) -> str:    
        pipe = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer, device='cuda:0')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = pipe(
                        prompt,
                        max_new_tokens=max_new_tokens,
                        use_cache=True
                    )
        return str(output)