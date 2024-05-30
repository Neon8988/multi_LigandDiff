import argparse
import os
import numpy as np
import tempfile
import ast
from itertools import combinations
import torch
from src import const
from src import utils
from src.lightning import DDPM
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_scatter import scatter_add
from src.molecule_builder import BasicLigandMetrics, build_mol,sanitycheck,write_xyz_file,ligand_slice,reset_dative_bonds
from openbabel import openbabel
from rdkit import Chem
import random

from molSimplify.Classes.mol3D import mol3D
from molSimplify.Classes.ligand import ligand_breakdown

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--complex', type=str)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_samples', type=int, default=1)
parser.add_argument('--ligand_sizes', type=str, default='random')
parser.add_argument('--add_Hs', type=eval, default=False)


atom2idx=const.ATOM2IDX 
idx2atom=const.IDX2ATOM 
charges=const.CHARGES
num_atom_types=const.NUMBER_OF_ATOM_TYPES
metal_list=const.metals


def sort_pos(xyz_file):
    metal_index = None
    with open(xyz_file, 'r') as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(tuple(metal_list)):
            metal_index = i
            break
    if metal_index is not None :
        lines.insert(2, lines.pop(metal_index))
        with open(f'{xyz_file[:-4]}_re.xyz', 'w') as new_file:
            new_file.writelines(lines)
        return True
    else:
        return False
    

def parse_complex(filename):
    label=filename[:-4]
    data_list=[]
    ele=[]
    pos=[]
    nuclear_charges=[]
    H_list=[]# store H atoms, maybe add them back later
    noH_list=[]
    with open(filename, 'r') as f:
        lines=f.readlines()
    
    for i in lines[3:]:
        if i.split()[0] =='H':
            H_list.append(i)
        else:
            noH_list.append(i)
            ele.append(atom2idx[i.split()[0]])
            nuclear_charges.append(charges[i.split()[0]])
            pos.append([float(j) for j in i.split()[1:]])
    noH_list.insert(0,lines[2])
    pos.insert(0,[float(j) for j in lines[2].split()[1:]]) # add  the position of metal
    nuclear_charges.insert(0,charges[lines[2].split()[0]]) # add  the nuclear charge of metal
    one_hot=torch.zeros(len(ele),8)
    one_hot[range(len(ele)),ele]=1
    one_hot=torch.cat([torch.zeros(8).view(1,-1),one_hot],dim=0)
    num_atoms=len(pos)
    pos=torch.tensor(pos)
    nuclear_charges=torch.tensor(nuclear_charges)

    with tempfile.NamedTemporaryFile() as tmp:
        tmp_file = tmp.name
        with open(f'{tmp_file}.xyz', 'w') as file:
            file.write(f"{num_atoms}\n\n")
            file.writelines(noH_list)

    mol=mol3D()
    mol.readfromxyz(f'{tmp_file}.xyz')
    liglist,ligdents,ligcon=ligand_breakdown(mol,silent=True,BondedOct=True)
    f_group=torch.zeros(num_atoms)
    for i in range(len(liglist)):
        f_group[liglist[i]]=i+1   

    ligand_group=torch.zeros((num_atoms,7) )
    ligand_group[range(len(f_group.long())),f_group.long()]=1
    
    anchor_group=torch.zeros(num_atoms)
    for i in range(len(ligcon)):
        anchor_group[ligcon[i]]=i+1
    anchors_group=torch.zeros((num_atoms,7) )
    anchors_group[range(len(anchor_group.long())),anchor_group.long()]=1
    coord_site=anchors_group[:,1:].any(dim=1).to(torch.int)
    # list all combinations of ligands
    all_lig=[]
    for i in range(len(liglist)): 
        all_lig.extend(list(list(combinations(liglist, i+1)))) 
    all_anchor=[]
    for i in range(len(ligcon)): 
        all_anchor.extend(list(list(combinations(ligcon, i+1))))
    for k in range(len(all_lig)):
        anchors=torch.zeros(num_atoms)
        ligand=torch.zeros(num_atoms)
        for i in all_anchor[k]:
            anchors[i]=1
        for i in all_lig[k]:
            ligand[i]=1
        
        context = 1-ligand

        data = Data(pos=pos,label=label,  context=context,  nuclear_charges=nuclear_charges,ligand_diff=ligand, num_atoms=num_atoms, one_hot=one_hot,ligand_group=ligand_group[:,1:],coord_site=coord_site)
        data_list.append(data)
    print('The coordination type of the given complex is:',ligdents)
    print('The number of combinations by masking the ligands from partially to totally is:',len(data_list))
    return data_list






def read_molecule(filename):
    if not filename.endswith('.xyz'):
        raise Exception('Unknown file extension, only .xyz file is supported')
    
    with open(filename, 'r') as file:
        metal = file.readlines()[2]
        if metal.split()[0] not in metal_list:
            if sort_pos(filename):
                print(f'Metal is not located at the begining of the coordinates.The {filename} is rearranged and saved to {filename[:-4]}_re.xyz')
                return parse_complex(f'{filename[:-4]}_re.xyz')
            else:
                print('Metal is not found in the domain of top 20 metals in CSD, please add the metal to the list of metals in const.py')
        else:
            return parse_complex(filename)


def get_ligand_size(ligand_size='random',startnum=1,endnum=10):
    if ligand_size == 'random':
        ligand_size=np.random.randint(startnum,endnum)
    else:
        ligand_size=int(ligand_size)
    return ligand_size


def reform_data(dataset,device,ligand_size='random'):
    new_data=[]
    for i in dataset:
        #context
        x=i['pos'][i['context']==1]
        one_hot=i['one_hot'][i['context']==1]
        ligand_group=i['ligand_group'][i['context']==1]
        nuclear_charges=i['nuclear_charges'][i['context']==1]
        c_coord_site=i['coord_site'][i['context']==1]

        #all possible ligand index to generate under given context
        index=torch.all(ligand_group == 0, dim=0).nonzero(as_tuple=True)[0]
        #coordination number of context
        cn_c=torch.sum(c_coord_site).item()
        ##Ligand denticity of context
        ligand_slices=ligand_slice(ligand_group[1:])# remove metal in the context
        LD_c=[]
        for item in ligand_slices:
            item=[i+1 for i in item]
            LD_c.append(torch.sum(i['coord_site'][i['context'].squeeze()==1][item]).item())
        assert sum(LD_c)==cn_c 
        #coordination type of generated ligands,i.e,ligand denticity(LD_g)
        #LD_g=const.cn_oct[cn_c][0]# max ligand denticity
        # LD_g= random.choice(const.coordination_type[cn_c])
        for LD_g in const.cn_oct[cn_c]:
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
            assert torch.sum(new_coord_site).item()==6
            new_ligand_group=torch.cat([ligand_group,gen_ligand_group],dim=0)
            new_onehot=torch.cat([one_hot,gen_ligand_onehot],dim=0)
            natoms=new_x.shape[0]
            data = Data(pos=new_x.to(device),label=f"{i['label']}_{LD_c}_{LD_g}",coord_site=new_coord_site.to(device),nuclear_charges=new_nuclear_charges.to(device), context=new_context.to(device), ligand_diff=new_ligand_diff.to(device), ligand_group=new_ligand_group.to(device), one_hot=new_onehot.to(device), num_atoms=natoms)
            new_data.append(data)
    #new_data=[item for item in new_data for _ in range(2)]
    return new_data

def generate_ligand(data,model,device,batch_size=64,outdir='generated_complexes'):
    os.makedirs(f'{outdir}/noH', exist_ok=True)
    ddpm = DDPM.load_from_checkpoint(model, map_location=device).eval().to(device)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    ligand_metrics=BasicLigandMetrics()
    num=0
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
            overlapping,liglist=sanitycheck(positions, atom_types,metal)
            total_atoms=sum(len(lig) for lig in liglist)+1
            if not overlapping and total_atoms==natoms[i].item():
                rdmols=[build_mol(positions[lig],atom_types[lig]) for lig in liglist if any(item >= n_fragment for item in lig)] 
                valid= ligand_metrics.compute_validity(rdmols)
                connected=ligand_metrics.compute_connectivity(valid)
                if len(connected)==len(rdmols):
                    num+=1
                    write_xyz_file(positions, atom_types,f'{outdir}/noH/{b}_{i}_{labels[i]}',metal,n_fragment)

                    # with tempfile.NamedTemporaryFile() as tmp:
                    #     tmp_file = tmp.name
                    #     write_xyz_file(positions, atom_types, tmp_file,metal,n_fragment='nan')
                    #     mol=mol3D()
                    #     mol.readfromxyz(f'{tmp_file}.xyz')
                    #     liglist,ligdent,ligcon=ligand_breakdown(mol,silent=True,BondedOct=False)
                    
                    # LD_c=ast.literal_eval(labels[i].split('_')[-2])
                    # LD_g=ast.literal_eval(labels[i].split('_')[-1])
                    # LD_g.extend(LD_c)
                    # if sorted(ligdent)==sorted(LD_g):
                    #     num+=1
                    #     write_xyz_file(positions, atom_types,f'{outdir}/noH/{b}_{i}_{labels[i]}',metal,n_fragment)
    print(f'Totally {num} valid complexes are generated and saved in {outdir}/noH') 



def add_H(org_xyz,gen_dir):
    """
    Add H from the original complex to the generated complex. 
    For ligands in context, H atoms are copied from the original complex.
    For generated ligands, H atoms are generated by RDKit.
    Args:
        org_xyz: orginial complex xyz file
        gen_dir: directory of generated complexes
    """
    #If using RDKit to automatically add H atoms for generated ligands, manual check after protonation is highly recommended.
    #one of possible issues:https://github.com/rdkit/rdkit/issues/4667
    #Alternative: ChimeraX --addh
    
    os.makedirs(f'{gen_dir}/add_H', exist_ok=True)
    my_mol=mol3D()
    my_mol.readfromxyz(f'{org_xyz}')
    liglist,ligdents,ligcon=ligand_breakdown(my_mol,silent=True,BondedOct=True)

    with open(f'{org_xyz}','r+') as f:
        lines=f.readlines()
        atom_hs=[]
        atom_nohs=[]
        for i in liglist:
            ligand_h=[lines[k+2] for k in i]
            atom_h=[]
            atom_noh=[]
            for atom in ligand_h:
                if atom.split()[0]=='H':
                    atom_h.append(atom)
                else:
                    atom_noh.append(atom)
            round_noh = []
            for item in atom_noh:
                elements = item.split('\t')
                new_elements = []
                for elem in elements:
                    try:
                        num = float(elem)
                        new_elements.append(f"{num:.3f}")
                    except ValueError:
                        new_elements.append(elem)
                round_noh.append('\t'.join(new_elements).strip())
            atom_hs.append(atom_h)
            atom_nohs.append(round_noh)  
            
    
    for gen_xyz in os.listdir(f'{gen_dir}/noH'):
        # add H atoms to heavy atoms in context
        my_mol=mol3D()
        my_mol.readfromxyz(f'{gen_dir}/noH/{gen_xyz}')
        liglist,ligdents,ligcon=ligand_breakdown(my_mol,silent=True,BondedOct=True)
        h_atoms=[]
        with open(f'{gen_dir}/noH/{gen_xyz}','r+') as f:
            lines=f.readlines()
        for i in liglist:
            ligand=[lines[k+2] for k in i]
            round_noh_ligand = []
            for item in ligand:
                elements = item.split('\t')  
                new_elements = []
                for elem in elements:
                    try:
                        num = float(elem)
                        new_elements.append(f"{num:.3f}")
                    except ValueError:
                        new_elements.append(elem)
                round_noh_ligand.append('\t'.join(new_elements).strip())
            if round_noh_ligand in atom_nohs:
                h_atoms.extend(atom_hs[atom_nohs.index(round_noh_ligand)])
                    
        #generate H atoms for the generated ligands
        context=int(lines[1])
        gen_ligands=lines[context+2:]
        gen_ligands.insert(0,lines[2])
        with tempfile.NamedTemporaryFile() as tmp:
            tmp_file = tmp.name
        with open(f'{tmp_file}.xyz','w') as f:
            f.write(f'{len(gen_ligands)}\n')
            f.write('ligand\n')
            f.write(''.join(gen_ligands))

        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("xyz", "sdf")     
        ob_mol = openbabel.OBMol()
        obConversion.ReadFile(ob_mol, f'{tmp_file}.xyz')
        obConversion.WriteFile(ob_mol, f'{tmp_file}.sdf')
        tmp_mol = Chem.SDMolSupplier(f'{tmp_file}.sdf', sanitize=False)[0]
        mol = Chem.RWMol()
        for atom in tmp_mol.GetAtoms():
            mol.AddAtom(Chem.Atom(atom.GetSymbol()))     
        mol.AddConformer(tmp_mol.GetConformer(0))
        for bond in tmp_mol.GetBonds():
            mol.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                bond.GetBondType())     
        m2=reset_dative_bonds(mol)
        try:
            Chem.SanitizeMol(m2,sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
            mh=Chem.AddHs(m2, addCoords=(len(m2.GetConformers()) > 0))
            Chem.SanitizeMol(mh,sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
            coord=Chem.MolToXYZBlock(mh)
            mol_H=[i for i in coord.split('\n') if i.startswith('H')]
            mol_H=['\t'.join(item.split())+'\n' for item in mol_H]
            lines.extend(h_atoms)
            lines.extend(mol_H)
            lines[0]=f'{len(lines)-2}\n'
            with open(f'{gen_dir}/add_H/{gen_xyz}','w+') as g:
                g.writelines(lines)
        except ValueError:
            continue





def main(outdir,model,complex,batch_size=64,n_samples=1,ligand_size='random',add_Hs=False):
    """
    Generate multiple new structures for each variation in a given complex
    Args:
        outdir: path to save generated complexes
        model:path to the pretrained model
        complex: path to the reference complex
        ligand_size: number of ligand atoms to generate, default is random
        add_Hs: add H atoms to the generated complexes
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset=read_molecule(complex)*n_samples
    print(f'{len(dataset)} samples will be generated')
    data=reform_data(dataset,device,ligand_size=ligand_size)
    batch_size=min(batch_size,len(data))
    generate_ligand(data,model,device,batch_size,outdir=outdir)
    if add_Hs:
        add_H(complex,outdir)
    print('Done!')
    

        


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.outdir,args.model,args.complex,args.batch_size,args.n_samples,args.ligand_sizes,args.add_Hs)

    

    