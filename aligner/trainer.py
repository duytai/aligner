import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from trl import DPOConfig, DPOTrainer
from transformers import (
    GenerationConfig,
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

class Trainer:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        model_id = self.kwargs.get('model')
        tokenizer_id = self.kwargs.get('tokenizer')
        dataset = self.kwargs.get('dataset')
        output_dir = self.kwargs.get('output_dir')

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        lora_config = dict(
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM',
            target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
        )
        peft_config = LoraConfig(**lora_config)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            #  attn_implementation='flash_attention_2',
            device_map='auto',
        )
        model = get_peft_model(base_model, peft_config)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

        training_args = DPOConfig(
            output_dir=output_dir,
            logging_steps=10
        )

        trainer = DPOTrainer(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=Dataset.from_list(dataset)
        )

        trainer.train()

        self.pipe = pipeline(
            'text-generation',
            model=base_model,
            tokenizer=trainer.tokenizer,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            max_new_tokens=256,
            generation_config=GenerationConfig(
                do_sample=False,
                top_p=None,
                temperature=None,
            )
        )
        self.pipe.model = trainer.model

    def __call__(self, dataset, batch_size: int = 100):
        self.pipe.tokenizer.padding_side = 'right'
        return self.pipe(dataset, batch_size=batch_size)
