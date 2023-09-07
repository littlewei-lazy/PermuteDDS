import torch
import os
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdMolDescriptors, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
from map4 import MAP4Calculator


def get_data(dataset):
    if dataset == 'ONEIL':
        drug_smiles_file = 'Data/ONEIL-COSMIC/drug_smiles.csv'
        cline_feature_file = 'Data/ONEIL-COSMIC/cell line_gene_expression.csv'
        drug_synergy_file = 'Data/ONEIL-COSMIC/drug_synergy.csv'
    else:
        drug_smiles_file = 'Data/ALMANAC-COSMIC/drug_smiles.csv'
        cline_feature_file = 'Data/ALMANAC-COSMIC/cell line_gene_expression.csv'
        drug_synergy_file = 'Data/ALMANAC-COSMIC/drug_synergy.csv'

    # cosmic_file = 'Data/cell_line/cosmic.csv'
    # gene_file = 'Data/cell_line/biogps_ccle_gdsc_normal.csv'
    gene_file = 'Data/cell_line/gene_expr_sparse.csv'
    mutations_file = 'Data/cell_line/mutations.csv'

    drug = pd.read_csv(drug_smiles_file, sep=',', header=0, index_col=[0])
    drug2smile = dict(zip(drug['pubchemid'], drug['isosmiles']))

    drug2hastt = {}
    drug2map4 = {}
    drug2maccs = {}
    for smile in tqdm(drug['isosmiles'].values):
        drug2hastt[smile], drug2map4[smile], drug2maccs[smile] = get_fp(smile)

    gene = pd.read_csv(cline_feature_file, sep=',', header=0, index_col=[0])
    gene_data = pd.read_csv(gene_file, sep=',', header=0, index_col=[0])
    mutation_data = pd.read_csv(mutations_file, sep=',', header=0, index_col=[0])

    cline_required = list(set(gene.index))
    cline_num = len(cline_required)

    cline2id = dict(zip(cline_required, range(cline_num))) ##给每个细胞系编号

    cline2gene = {}
    cline2mutation = {}
    for cline, cline_id in cline2id.items():
        cline2gene[cline_id] = np.array(gene_data.loc[cline].values, dtype='float32')
        cline2mutation[cline_id] = np.array(mutation_data.loc[cline].values, dtype='float32')
    gene_dim = gene_data.shape[1]
    mutation_dim = mutation_data.shape[1]
    # id2sparse = {key: cline2sparse[cline] for (key, cline) in id2cline.items()}
    # id2mutation = {key: cline2mutation[cline] for (key, cline) in id2cline.items()}

    synergy_load = pd.read_csv(drug_synergy_file, sep=',', header=0)
    synergy = [[row[0], row[1], cline2id[row[2]], float(row[3])] for _, row in
               synergy_load.iterrows()]
    
    return synergy, drug2smile, drug2hastt, drug2map4, drug2maccs, cline2gene, cline2mutation, gene_dim, mutation_dim


def data_split(synergy, test_size, rd_seed=42):
    synergy = np.array(synergy)
    train_data, test_data = train_test_split(synergy, test_size=test_size, random_state=rd_seed)

    return train_data, test_data


def process_data(synergy, drug2smile, cline2gene, cline2mutation, task_name='regression'):
    processed_synergy = []
    # drug2smile
    for row in synergy:
        processed_synergy.append([drug2smile[row[0]], drug2smile[row[1]],
                                cline2gene[row[2]], cline2mutation[row[2]], float(row[3])])

    if task_name == 'classification':
        threshold = 30
        for row in processed_synergy:
            row[3] = 1 if row[3] >= threshold else 0

    return np.array(processed_synergy, dtype=object)


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def get_fp(smile):
    # RDKit descriptors -->
    nbits = 512
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    MAP4 = MAP4Calculator(dimensions=nbits)

    fpFunc_dict = {}
    fpFunc_dict['ecfp0'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 0, nBits=nbits)
    fpFunc_dict['ecfp2'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, nBits=nbits)
    fpFunc_dict['ecfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits)
    fpFunc_dict['ecfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=nbits)
    fpFunc_dict['fcfp2'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, useFeatures=True, nBits=nbits)
    fpFunc_dict['fcfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True, nBits=nbits)
    fpFunc_dict['fcfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True, nBits=nbits)
    fpFunc_dict['lecfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits)
    fpFunc_dict['lecfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=nbits)
    fpFunc_dict['lfcfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True, nBits=nbits)
    fpFunc_dict['lfcfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True, nBits=nbits)
    #fpFunc_dict['maccs'] = lambda m: MACCSkeys.GenMACCSKeys(m)
    fpFunc_dict['hashap'] = lambda m: rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=nbits)
    fpFunc_dict['hashtt'] = lambda m: rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m, nBits=nbits)
    #fpFunc_dict['avalon'] = lambda m: fpAvalon.GetAvalonFP(m, nbits)
    #fpFunc_dict['laval'] = lambda m: fpAvalon.GetAvalonFP(m, nbits)
    fpFunc_dict['rdk5'] = lambda m: Chem.RDKFingerprint(m, maxPath=5, fpSize=nbits, nBitsPerHash=2)
    fpFunc_dict['rdk6'] = lambda m: Chem.RDKFingerprint(m, maxPath=6, fpSize=nbits, nBitsPerHash=2)
    fpFunc_dict['rdk7'] = lambda m: Chem.RDKFingerprint(m, maxPath=7, fpSize=nbits, nBitsPerHash=2)
    fpFunc_dict['rdkDes'] = lambda m: calc.CalcDescriptors(m)
    fpFunc_dict['maccs'] = lambda m: MACCSkeys.GenMACCSKeys(m)
    fpFunc_dict['map4'] = lambda m: MAP4.calculate(m)

    mol = Chem.MolFromSmiles(smile)
    hashtt = np.array(fpFunc_dict['hashtt'](mol)).flatten().astype(np.float32)
    map4 = np.array(fpFunc_dict['map4'](mol)).flatten().astype(np.float32)
    maccs = np.array(fpFunc_dict['maccs'](mol)).flatten().astype(np.float32)  # length is 167

    return hashtt, map4, maccs


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('Folder created: ', path)
    else:
        print('Folder existed: ', path)


def ptable_to_csv(table, filename, headers=True):
    """Save PrettyTable results to a CSV file.

    Adapted from @AdamSmith https://stackoverflow.com/questions/32128226

    :param PrettyTable table: Table object to get data from.
    :param str filename: Filepath for the output CSV.
    :param bool headers: Whether to include the header row in the CSV.
    :return: None
    """
    raw = table.get_string()
    data = [tuple(filter(None, map(str.strip, splitline)))
            for line in raw.splitlines()
            for splitline in [line.split('|')] if len(splitline) > 1]
    if table.title is not None:
        data = data[1:]
    if not headers:
        data = data[1:]
    with open(filename, 'w') as f:
        for d in data:
            f.write('{}\n'.format(','.join(d)))
