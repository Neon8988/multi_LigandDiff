import argparse
from pathlib import Path
import numpy as np
from torch_geometric.loader import DataLoader
import os
import random 
import pandas as pd
import ast
from rdkit import Chem
from generate import get_ligand_size
from src.molecule_builder import BasicLigandMetrics, build_mol,sanitycheck,write_xyz_file,ligand_slice
import torch
import tempfile
from src import utils
from src.lightning import DDPM
from torch_geometric.data import Data
from src import const
from torch_scatter import scatter_add
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Classes.ligand import ligand_breakdown




parser = argparse.ArgumentParser()
parser.add_argument('--model', type=Path)
parser.add_argument('--outdir', type=Path)
parser.add_argument('--dataset', type=Path)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--ligand_size', type=str, default='random')

num_atom_types=const.NUMBER_OF_ATOM_TYPES




def reform_data(dataset,device,ligand_size='random'):
    new_data=[]
    for i in dataset:
        #context
        x=i['pos'][i['context']==1]
        one_hot=i['one_hot'][i['context']==1]
        ligand_group=i['ligand_group'][i['context']==1]
        nuclear_charges=i['nuclear_charges'][i['context']==1]
        c_coord_site=i['coord_site'][i['context']==1]

        index=torch.all(ligand_group == 0, dim=0).nonzero(as_tuple=True)[0]
        cn_c=torch.sum(c_coord_site).item()
        ligand_slices=ligand_slice(ligand_group[1:])# remove metal in the context
        LD_c=[]
        for item in ligand_slices:
            item=[i+1 for i in item]
            LD_c.append(torch.sum(i['coord_site'][i['context'].squeeze()==1][item]).item())
        assert sum(LD_c)==cn_c 
        if 0<=cn_c<5:
            for LD_g in const.cn_nonoct[cn_c]:
                ligand_index=index[:len(LD_g)]
                gen_ligand_groups=[]
                gen_ligand_coord_sites=[]
                for k,num_coord_site in zip(ligand_index,LD_g):
                    if num_coord_site<3:
                        g_ligand_size=np.random.randint(num_coord_site,10)
                    else:
                        g_ligand_size=get_ligand_size(ligand_size,startnum=10,endnum=30)
                    assert g_ligand_size>= num_coord_site,"The assigned ligand size is smaller than the denticity of the generated ligand. Please assign a larger ligand size."
                    gen_ligand_group=torch.zeros(g_ligand_size,6)
                    gen_ligand_group[:,k]=1
                    gen_ligand_groups.append(gen_ligand_group)
                    gen_coord_site=torch.zeros(g_ligand_size)
                    gen_coord_site[:num_coord_site]=1
                    gen_ligand_coord_sites.append(gen_coord_site)
                gen_ligand_group=torch.cat(gen_ligand_groups,dim=0)
                gen_ligand_size=gen_ligand_group.shape[0]
                gen_ligand_x=torch.zeros(gen_ligand_size,3)
                gen_ligand_coord_site=torch.cat(gen_ligand_coord_sites,dim=0)
                gen_ligand_onehot=torch.zeros(gen_ligand_size,num_atom_types)
                new_x=torch.cat([x,gen_ligand_x],dim=0)
                new_context=torch.cat([torch.ones(x.shape[0]),torch.zeros(gen_ligand_size)],dim=0)
                new_ligand_diff=torch.cat([torch.zeros(x.shape[0]),torch.ones(gen_ligand_size)],dim=0)
                new_nuclear_charges=torch.cat([nuclear_charges,torch.zeros(gen_ligand_size)],dim=0)
                new_coord_site=torch.cat([c_coord_site,gen_ligand_coord_site],dim=0)
                assert new_x.shape[0]==new_nuclear_charges.shape[0]
                new_ligand_group=torch.cat([ligand_group,gen_ligand_group],dim=0)
                new_onehot=torch.cat([one_hot,gen_ligand_onehot],dim=0)
                natoms=new_x.shape[0]
                data = Data(pos=new_x.to(device),label=f"{i['label']}_{LD_c}_{LD_g}",coord_site=new_coord_site.to(device),nuclear_charges=new_nuclear_charges.to(device), context=new_context.to(device), ligand_diff=new_ligand_diff.to(device), ligand_group=new_ligand_group.to(device), one_hot=new_onehot.to(device), num_atoms=natoms)
                new_data.append(data)
    #new_data=[item for item in new_data for _ in range(2)]
    return new_data

def main(model,outdir,dataset,batch_size=64,ligand_size='random'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(outdir, exist_ok=True)
    ligand_metrics=BasicLigandMetrics()
    ddpm = DDPM.load_from_checkpoint(model, map_location=device).eval().to(device)
    dataset=torch.load(dataset)
    new_data=reform_data(dataset,device,ligand_size=ligand_size)
    batch_size=batch_size if batch_size is not None else ddpm.batch_size


    num=0
    duplicate=0
    total_ligands=0
    valid_comp=0
    validity=0
    connectivity=0
    connected_mols=[]

    dataloader = DataLoader(new_data, batch_size=batch_size, shuffle=False)
    for b, data in enumerate(dataloader):
        pos_orginal=data['pos']
        batch_seg=data.batch
        batch_size=torch.max(batch_seg)+1
        context = data['context'].view(-1,1)
        metals=[data['nuclear_charges'][batch_seg==i][0] for i in range(batch_size)]
        fixed_mean = scatter_add(pos_orginal*context, batch_seg, dim=0)/scatter_add(context, batch_seg, dim=0).view(-1,1)
        natoms=data['num_atoms']
        labels=data['label']
        
        try:
            chain_batch = ddpm.sample_chain(data, keep_frames=100)
        except utils.FoundNaNException as e:
            continue
        
        x = chain_batch[0][ :, :3]
        x=x+fixed_mean[batch_seg]
        one_hot = chain_batch[0][ :, 3:]
        unique_indices = torch.unique(batch_seg)
        for i in unique_indices:
            n_fragment=int(torch.sum(context[batch_seg==i].squeeze()).item())
            positions=x[batch_seg==i]
            atom_types=one_hot[batch_seg==i].argmax(dim=1)
            metal=metals[i]
            overlapping,liglist=sanitycheck(positions, atom_types,metal,BondedOct=False)
            total_atoms=sum(len(lig) for lig in liglist)+1
            if not overlapping and total_atoms==natoms[i].item():
                rdmols=[build_mol(positions[lig],atom_types[lig]) for lig in liglist if any(item >= n_fragment for item in lig)] 
                total_ligands+=len(rdmols)
                valid= ligand_metrics.compute_validity(rdmols)
                validity+=len(valid)
                connected=ligand_metrics.compute_connectivity(valid)
                connectivity+=len(connected)
                if len(connected)==len(rdmols):
                    valid_comp+=1
                    connected_mols.append([Chem.MolToSmiles(ligand)for ligand in rdmols])
                    # write_xyz_file(positions, atom_types,f'{outdir}/{b}_{i}_{labels[i]}',metal,n_fragment)
                    with tempfile.NamedTemporaryFile() as tmp:
                        tmp_file = tmp.name
                        write_xyz_file(positions, atom_types, tmp_file,metal,n_fragment='nan')
                        mol=mol3D()
                        mol.readfromxyz(f'{tmp_file}.xyz')
                        liglist,ligdent,ligcon=ligand_breakdown(mol,silent=True,BondedOct=False)
                    
                    LD_c=ast.literal_eval(labels[i].split('_')[-2])
                    LD_g=ast.literal_eval(labels[i].split('_')[-1])
                    LD_g.extend(LD_c)
                    if sorted(ligdent)==sorted(LD_g):
                        num+=1
                        #write_xyz_file(positions, atom_types,f'{outdir}/{b}_{i}_{labels[i]}',metal,n_fragment)

        
    connected_mols=[tuple(i) for i in connected_mols]   
    train_smiles=pd.read_json('data/train_smiles_diff_ligands.json')
    train_smiles=train_smiles['smiles'].tolist()
    for item in connected_mols:
        if item in train_smiles:
            duplicate+=1
    
    
    metrics={
             'valid_ligand':validity/total_ligands,
             'connected_ligand':connectivity/validity,
             'valid_complex':valid_comp/len(new_data),
             'uniqueness':len(set(connected_mols))/len(connected_mols),
             'novelty':1-duplicate/valid_comp,
             'correct LD':num/valid_comp
                }
    print(f'Finish sampling on  {len(new_data)} samples and {valid_comp} valid complexes have been successfully generated.')
    print('Metrics for sampling:')
    print(metrics)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.model,args.outdir,args.dataset,args.batch_size,args.ligand_size)




