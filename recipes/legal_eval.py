from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from aligner.utils import trim_parse_form
import numpy as np

class LegalEval:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        llm = self.kwargs.get('llm')
        dataset = self.kwargs.get('dataset')
        dataset = KeyDataset(dataset, 'question')
        result = np.zeros(len(dataset))
        pos = 0

        for outputs in tqdm(llm(dataset, 100)):
            for output in outputs:
                content = output['generated_text'][-1]['content']
                parsed = trim_parse_form(content)
                if 'answer' in parsed and 'reason' in parsed:
                    if parsed['answer'] == 'Yes':
                        result[pos] = 1
                    elif parsed['answer'] == 'No':
                        result[pos] = 0
                    else:
                        result[pos] = 2
                else:
                    result[pos] = 2
                pos += 1

        print(f'Agree: {result[result==1].size}')
        print(f'Disagree: {result[result==0].size}')
        print(f'Errors: {result[result==2].size}')
