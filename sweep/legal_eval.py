import yaml
import copy
from pathlib import Path
from glob import glob

LLAMA = """
_component_: aligner.llama.Llama
model: /common/home/users/d/dtnguyen.2019/scratchDirectory/Meta-Llama-3.1-8B-Instruct
tokenizer: /common/home/users/d/dtnguyen.2019/scratchDirectory/Meta-Llama-3.1-8B-Instruct
"""

EDITOR = """
_component_: aligner.editor.Editor
hparam: null
dataset: null
"""

TRAINER = """
_component_: aligner.trainer.Trainer
model: /common/home/users/d/dtnguyen.2019/scratchDirectory/Meta-Llama-3.1-8B-Instruct
tokenizer: /common/home/users/d/dtnguyen.2019/scratchDirectory/Meta-Llama-3.1-8B-Instruct
output_dir: /common/home/users/d/dtnguyen.2019/scratchDirectory/checkpoint/
dataset: null
"""

DATASET = """
_component_: data.regulation.Regulation
split: null
regulation: null
question_prompt: |
  I am going to describe a scenario.
  Does the scenario involve a violation of law?
  Tell me 'Yes' or 'No' and give me a short reason.
  Here is the scenario: {scenario}

  Provide your response by filling out the following forms (no markdown).
  Answer: string
  Reason: string
answer_prompt: |
  Answer: {answer}
  Reason: {reason}
"""

REGULATION = 'Singapore Rapid Transit Systems Regulations'

class LegalEval:
    def __init__(self):
        pass

    def __call__(self):
        kwargs = {
            'dataset': yaml.safe_load(DATASET),
            'llama': yaml.safe_load(LLAMA),
            'editor': yaml.safe_load(EDITOR),
            'trainer': yaml.safe_load(TRAINER)
        }
        output = Path.cwd() / 'output'

        dataset = {}
        for split in ['train', 'test', 'loc']:
            template = copy.deepcopy(kwargs['dataset'])
            template['split'] = split
            #  template['max_len'] = 1
            template['regulation'] = REGULATION
            dataset[split] = template

        llama = kwargs['llama']

        trainer = kwargs['trainer']
        trainer['dataset'] = dataset['train']

        runs = [
            {
                'dataset': dataset,
                'llm': llama,
                'output_dir': str(output / 'raw')
            },
            {
                'dataset': dataset,
                'llm': trainer,
                'output_dir': str(output / 'trainer')
            }
        ]

        editor = kwargs['editor']
        editor['dataset'] = dataset['train']

        for hparam in glob('hparam/*.yaml'):
            tmp = copy.deepcopy(editor)
            tmp['hparam'] = hparam
            runs.append({
                'dataset': dataset,
                'llm': tmp,
                'output_dir': str(output / Path(hparam).stem)
            })

        return runs
