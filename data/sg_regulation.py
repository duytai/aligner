from torch.utils.data import Dataset
from datasets import load_dataset
import pandas as pd

class SgRegulation(Dataset):
    def __init__(
        self,
        split: str,
        question_prompt: str,
        answer_prompt: str,
        ratio: float = 0.8,
        max_len: int = None,
        regulation: str = None,
        misconduct: str = None,
    ) -> None:
        dataset = load_dataset('taidnguyen/SingaporeLaw', split='train')
        df = dataset.to_pandas()

        regulation_list = list(df.groupby('regulation').groups.keys())
        misconduct_list = list(df.groupby('misconduct').groups.keys())

        if regulation is not None:
            assert regulation in regulation_list
            df = df[df['regulation'] == regulation]

        if misconduct is not None:
            assert misconduct in misconduct_list

        trains, tests = [], []
        keys = df.groupby('misconduct').groups.keys()
        for key in keys:
            misconducts = df[df['misconduct'] == key]
            split_at = int(ratio * len(misconducts))
            trains.append(misconducts[:split_at])
            tests.append(misconducts[split_at:])

        train = pd.concat(trains, ignore_index=True, axis=0)
        test = pd.concat(tests, ignore_index=True, axis=0)

        assert split in ['train', 'test']
        data = train if split == 'train' else test
        self.data = list(
            zip(
                data['scenario'].to_list(),
                data['justification'].to_list(),
                data['reason'].to_list(),
            )
        )
        if max_len is not None and max_len < len(self.data):
            self.data = self.data[:max_len]
        self.question_prompt = question_prompt
        self.answer_prompt = answer_prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scenario, justification, reason = self.data[idx]
        return {
            'chosen': [
                {
                    'role': 'user',
                    'content': self.question_prompt.format(scenario=scenario)
                },
                {
                    'role': 'assistant',
                    'content': self.answer_prompt.format(answer='Yes', reason=justification)
                }
            ],
            'rejected': [
                {
                    'role': 'user',
                    'content': self.question_prompt.format(scenario=scenario)
                },
                {
                    'role': 'assistant',
                    'content': self.answer_prompt.format(answer='No', reason=reason)
                }
            ],
            'question': [
                {
                    'role': 'user',
                    'content': self.question_prompt.format(scenario=scenario)
                }
            ],
            'answer': [
                {
                    'role': 'assistant',
                    'content': self.answer_prompt.format(answer='Yes', reason=justification)
                }
            ],
        }
