import torch
from transformers import (
    pipeline,
    GenerationConfig,
    AutoTokenizer,
    AutoModelForCausalLM
)

class Llama:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        model_id = self.kwargs['model']
        tokenizer_id = self.kwargs['tokenizer']

        free, total = torch.cuda.mem_get_info()
        print(f'Free memory: {free / 1e9} GB, Total memory: {total / 1e9} GB')

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'left'

        self.tokenizer = tokenizer
        self.model = model
        self.pipe = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            max_new_tokens=256,
            generation_config=GenerationConfig(
                do_sample=False,
                top_p=None,
                temperature=None,
            )
        )

    def __call__(self, dataset, batch_size: int = 100):
        self.pipe.tokenizer.padding_side = 'right'
        return self.pipe(dataset, batch_size=batch_size)
