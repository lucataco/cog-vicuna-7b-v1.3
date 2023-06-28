from cog import BasePredictor, Input
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda"
MODEL_ID = 'lmsys/vicuna-7b-v1.3'

class Predictor(BasePredictor):
    def setup(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            cache_dir="cache"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir="cache"
        )

    def predict(self,
        prompt: str = Input(description="Instruction for the model"),
        max_new_tokens: int = Input(description="max tokens to generate", default=128),
        temperature: float = Input(description="0.01 to 1.0 temperature", default=0.75),
    ) -> str:    
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
        outputs = self.model.generate(
            inputs, max_new_tokens=max_new_tokens, temperature=temperature
        )
        output = self.tokenizer.decode(outputs[0])
        print(output)

        return output