from tqdm import tqdm
from pathlib import Path
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
        output_dir = Path(self.kwargs.get('output_dir'))

        output_dir.mkdir(parents=True, exist_ok=True)

        result = {}
        for split_name, ds in dataset.items():
            ds = KeyDataset(ds, 'question')
            data = np.zeros(len(ds))
            pos = 0
            

            for outputs in tqdm(llm(ds, 100)):
                for output in outputs:
                    content = output['generated_text'][-1]['content']
                    parsed = trim_parse_form(content)
                    if 'answer' in parsed and 'reason' in parsed:
                        if parsed['answer'] == 'Yes':
                            data[pos] = 1
                        elif parsed['answer'] == 'No':
                            data[pos] = 0
                        else:
                            data[pos] = 2
                    else:
                        data[pos] = 2
                    pos += 1
            
            result[split_name] = data

        np.savez(output_dir / 'result.npz', **result)
        #  np.save(output_dir / f'{split_name}.npz', )
        #  print(f'Split: {split_name}')
        #  print(f'Agree: {data[data==1].size}')
        #  print(f'Disagree: {data[data==0].size}')
        #  print(f'Errors: {data[data==2].size}')
