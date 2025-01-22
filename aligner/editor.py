import spacy
import torch
import yaml
from transformers import (
    GenerationConfig,
    pipeline
)
from pathlib import Path
from easyeditor import (
    LoRAHyperParams,
    EMMETHyperParams,
    FTHyperParams,
    GraceHyperParams,
    MEMITHyperParams,
    PMETHyperParams,
    QLoRAHyperParams,
    ROMEHyperParams,
    R_ROMEHyperParams,
    WISEHyperParams,
    BaseEditor,
)

class Editor:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        hparam_map = {
            'LoRA': LoRAHyperParams,
            'FT': FTHyperParams,
            'WISE': WISEHyperParams,
            'EMMET': EMMETHyperParams,
            'GRACE': GraceHyperParams,
            'MEMIT': MEMITHyperParams,
            'PMET': PMETHyperParams,
            'QLoRA': QLoRAHyperParams,
            'ROME': ROMEHyperParams,
            'R-ROME': R_ROMEHyperParams,
        }

        dataset = self.kwargs.get('dataset')
        hparam_file = self.kwargs.get('hparam')

        config = yaml.safe_load(Path(hparam_file).read_text())
        editing_hparams = hparam_map.get(config['alg_name'])
        assert editing_hparams is not None

        hparams = editing_hparams.from_hparams(hparam_file)
        editor = BaseEditor.from_hparams(hparams)
        nlp = spacy.load('en_core_web_sm')

        questions = []
        answers = []
        subjects = []

        for x in dataset:
            content = x['question'][-1]['content']
            subject = ([tok.text for tok in nlp(content) if tok.dep_ == 'nsubj'] + [tok.text for tok in nlp(content)])[0]

            full_prompt = editor.tok.apply_chat_template(
                x['question'] + x['answer'],
                tokenize=False,
            )
            question_prompt = editor.tok.apply_chat_template(
                x['question'],
                tokenize=False,
            )
            answer_prompt = full_prompt[len(question_prompt):]

            subjects.append(subject)
            questions.append(question_prompt)
            answers.append(answer_prompt)

        metrics, edited_model, _ = editor.edit(
            prompts=questions,
            target_new=answers,
            subject=subjects,
            loc_prompts=[''] * len(questions),
            sequential_edit=True,
            eval_metric='token em',
        )
        self.pipe = pipeline(
            'text-generation',
            model=editor.model,
            tokenizer=editor.tok,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            max_new_tokens=256,
            generation_config=GenerationConfig(
                do_sample=False,
                top_p=None,
                temperature=None,
            )
        )
        self.pipe.model = edited_model

    def __call__(self, dataset, batch_size: int = 100):
        self.pipe.tokenizer.padding_side = 'right'
        return self.pipe(dataset, batch_size=batch_size)
