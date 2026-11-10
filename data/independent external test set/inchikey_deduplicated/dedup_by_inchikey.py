import os
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.dirname(HERE)
DATA = os.path.dirname(EXT)

SUBSETS = [
    ('pathway_subset.csv', 'smiles'),
    ('superclass_subset.csv', 'smiles'),
    ('class_subset.csv', 'smiles'),
]


def inchikey(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol) if mol else None


def main():
    train = pd.read_csv(os.path.join(DATA, 'train_data.csv'))
    val = pd.read_csv(os.path.join(DATA, 'validation_data.csv'))
    ref = {
        k for k in (inchikey(s) for s in pd.concat([train['index'], val['index']]))
        if k
    }
    total = 0
    for name, col in SUBSETS:
        df = pd.read_csv(os.path.join(EXT, name))
        keep = [inchikey(s) not in ref for s in df[col]]
        out = df.loc[keep].reset_index(drop=True)
        out.to_csv(os.path.join(HERE, name), index=False)
        total += len(out)
        print(f'{name}: {len(df)} -> {len(out)}')
    print(f'total: {total}')


if __name__ == '__main__':
    main()
