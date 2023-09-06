from torch.utils.data import Dataset
import torch 


class FPDataset(Dataset):
    def __init__(self, smiles1, smiles2, gene, mutation, labels, drug2hashtt, drug2map4, drug2maccs):
        self.labels =  labels 
        self.length = len(self.labels)
        self.smiles1 = smiles1
        self.smiles2 = smiles2

        self.drug2hashtt = drug2hashtt
        self.drug2map4 = drug2map4
        self.drug2maccs = drug2maccs

        self.cline_gene = gene
        self.cline_mutation = mutation

    def __getitem__(self, idx):
        label = self.labels[idx]

        cline_gene = torch.FloatTensor(self.cline_gene[idx])
        cline_mutation = torch.FloatTensor(self.cline_mutation[idx])
        
        drug1_hashtt = torch.FloatTensor(self.drug2hashtt[self.smiles1[idx]])
        drug2_hashtt = torch.FloatTensor(self.drug2hashtt[self.smiles2[idx]])

        drug1_map4 = torch.FloatTensor(self.drug2map4[self.smiles1[idx]])
        drug2_map4 = torch.FloatTensor(self.drug2map4[self.smiles2[idx]])

        drug1_maccs = torch.FloatTensor(self.drug2maccs[self.smiles1[idx]])
        drug2_maccs = torch.FloatTensor(self.drug2maccs[self.smiles2[idx]])

        return drug1_hashtt, drug2_hashtt, drug1_map4, drug2_map4, drug1_maccs, drug2_maccs, \
            cline_gene, cline_mutation, torch.FloatTensor([label])

    def __len__(self):
        return self.length
