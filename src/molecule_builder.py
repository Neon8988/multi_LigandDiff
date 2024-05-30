
import torch
import numpy as np
import tempfile
from rdkit import Chem, Geometry
from openbabel import openbabel
from src import const
import warnings
import sys
sys.path.append('/mnt/gs21/scratch/jinhongn/pyg/molSimplify')
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Classes.ligand import ligand_breakdown

import warnings
warnings.filterwarnings("ignore")

def get_bond_order(atom1, atom2, distance, check_exists=True, margins=const.MARGINS_EDM):
    distance = 100 * distance  # We change the metric

    # Check exists for large molecules where some atom pairs do not have a
    # typical bond length.
    if check_exists:
        if atom1 not in const.BONDS_1:
            return 0
        if atom2 not in const.BONDS_1[atom1]:
            return 0

    # margin1, margin2 and margin3 have been tuned to maximize the stability of the QM9 true samples
    if distance < const.BONDS_1[atom1][atom2] + margins[0]:

        # Check if atoms in bonds2 dictionary.
        if atom1 in const.BONDS_2 and atom2 in const.BONDS_2[atom1]:
            thr_bond2 = const.BONDS_2[atom1][atom2] + margins[1]
            if distance < thr_bond2:
                if atom1 in const.BONDS_3 and atom2 in const.BONDS_3[atom1]:
                    thr_bond3 = const.BONDS_3[atom1][atom2] + margins[2]
                    if distance < thr_bond3:
                        return 3  # Triple
                return 2  # Double
        return 1  # Single
    return 0  # No bond

def create_conformer(coords):
    conformer = Chem.Conformer()
    for i, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(i, Geometry.Point3D(x, y, z))
    return conformer

def ligand_slice(ligand_group):
    column_indices = torch.argmax((ligand_group != 0).int(), dim=1)
    groups = {}
    for row_idx, col_idx in enumerate(column_indices):
        if col_idx.item() in groups:
            groups[col_idx.item()].append(row_idx)
        else:
            groups[col_idx.item()] = [row_idx]

    ligand_slices=list(groups.values())
    return ligand_slices




def extract_ligand(x,onehot,ligand_diff_mask,batch_seg,ligand_group):
    unique_indices = torch.unique(batch_seg)
    ligands=[]
    for idx in unique_indices:
        ligand_diff_masks = ligand_diff_mask[batch_seg == idx]
        indices = (ligand_diff_masks == 1).nonzero(as_tuple=True)[0]
        pos = x[batch_seg == idx][indices]
        hs = onehot[batch_seg == idx][indices]
        atoms = torch.argmax(hs, dim=1)
        ligand_diff_group=ligand_group[batch_seg == idx][ligand_diff_masks.squeeze()==1]
        column_indices = torch.argmax((ligand_diff_group != 0).int(), dim=1)
        groups = {}
        for row_idx, col_idx in enumerate(column_indices):
            if col_idx.item() in groups:
                groups[col_idx.item()].append(row_idx)
            else:
                groups[col_idx.item()] = [row_idx]

        ligand_diff_indices=list(groups.values())
        single_ligand_diffs=[]
        for i in ligand_diff_indices:
            single_pos=pos[i]
            single_atoms=atoms[i]
            single_ligand_diffs.append(list((single_pos,single_atoms)))
        ligands.append(single_ligand_diffs)
    return ligands




def extract_ligand_index(ligand_diff_mask,ligand_group):
    ligand_diff_indices=[]
    ligand_diff_group=ligand_group[ligand_diff_mask.squeeze()==1]
    column_indices = torch.argmax((ligand_diff_group != 0).int(), dim=1).unique()
    ligand_diff_indices=[]
    for i in column_indices:
        single_ligand_diff=torch.where(ligand_group[:, i] == 1)[0]
        ligand_diff_indices.append(single_ligand_diff.tolist())
    return ligand_diff_indices


def write_xyz_file(coords, atom_types,filename,metal,n_fragment):
    idx2atom = const.IDX2ATOM
    idx2metals=const.idx2metals
    f=open(f'{filename}.xyz','w')
    assert len(coords) == len(atom_types)
    f.write(f'{len(coords)}\n{n_fragment}\n')
    if metal==None:
        for i in range(len(coords)):
            atom=idx2atom[atom_types[i].item()]
            f.write(f"{atom} \t{coords[i, 0]:.6f}\t{coords[i, 1]:.6f}\t{coords[i, 2]:.6f}\n")
        f.close()

    else:
        for i in range(len(coords)):
            if i ==0:
                atom=idx2metals[metal.item()]
            else:
                atom=idx2atom[atom_types[i].item()]
            f.write(f"{atom} \t{coords[i, 0]:.6f}\t{coords[i, 1]:.6f}\t{coords[i, 2]:.6f}\n")
        f.close()

def build_mol(positions, atom_types,use_openbabel=True):
                   
    """
    Build RDKit molecule
    Args:
        positions: N x 3
        atom_types: N
        dataset_info: dict
        add_coords: Add conformer to mol (always added if use_openbabel=True)
        use_openbabel: use OpenBabel to create bonds
    Returns:
        RDKit molecule
    """
    if use_openbabel:
        mol = make_mol_openbabel(positions, atom_types)
                                
    else:
        raise NotImplementedError

    return mol



def make_mol_openbabel(positions, atom_types):
    """
    Build an RDKit molecule using openbabel for creating bonds
    Args:
        positions: N x 3
        atom_types: N
        atom_decoder: maps indices to atom types
    Returns:
        rdkit molecule
    """
    openbabel.obErrorLog.StopLogging()
    with tempfile.NamedTemporaryFile() as tmp:
        tmp_file = tmp.name
        # Write xyz file
        write_xyz_file(positions, atom_types, tmp_file,metal=None,n_fragment='nan')

        # Convert to sdf file with openbabel
        # openbabel will add bonds
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("xyz", "sdf")     
        ob_mol = openbabel.OBMol()
        obConversion.ReadFile(ob_mol, f'{tmp_file}.xyz')
        obConversion.WriteFile(ob_mol, f'{tmp_file}.sdf')
        # Read sdf file with RDKit
        tmp_mol = Chem.SDMolSupplier(f'{tmp_file}.sdf', sanitize=False)[0]
    # Build new molecule. This is a workaround to remove radicals.
    mol = Chem.RWMol()
    for atom in tmp_mol.GetAtoms():
        mol.AddAtom(Chem.Atom(atom.GetSymbol()))
    mol.AddConformer(tmp_mol.GetConformer(0))

    for bond in tmp_mol.GetBonds():
        mol.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                bond.GetBondType())

    return mol
   


class BasicLigandMetrics(object):
    def __init__(self,connectivity_thresh=1.0):
                 
        self.connectivity_thresh = connectivity_thresh

    def compute_validity(self, generated):
        """ generated: list of couples (positions, atom_types)"""
        valid=[]
        for i in generated:
            try:
                Chem.SanitizeMol(i)
                valid.append(i)
            except ValueError:
                continue

        return valid

    def compute_connectivity(self, generated):
        """ Consider molecule connected if its largest fragment contains at
        least x% of all atoms, where x is determined by
        self.connectivity_thresh (defaults to 100%). """
        connected=[]
        for i in generated:
            mol_frags = Chem.rdmolops.GetMolFrags(i, asMols=True)
            if len(mol_frags) == 1:
                connected.append(i)
        return connected


def sanitycheck(positions, atom_types,metal,BondedOct=True):
    """
    Using molsimplify to compute metrics of generated molecules
    Args:
        positions: N x 3
        atom_types: N
    Returns:
        validity, connectivity, uniqueness, novelty
    """
    # idx2atom = const.IDX2ATOM
    # atom_types = [idx2atom[x] for x in atom_types]

    with tempfile.NamedTemporaryFile() as tmp:
        tmp_file = tmp.name
        write_xyz_file(positions, atom_types, tmp_file,metal,n_fragment='nan')
    
    mol=mol3D()
    mol.readfromxyz(f'{tmp_file}.xyz')
    overlapping=mol.sanitycheck(silence=True)[0]
    liglist,ligdents,ligcon=ligand_breakdown(mol,silent=True,BondedOct=BondedOct)
    return overlapping,liglist
    

def sublist_overlap(list1,list2):
    tuples1 = {tuple(sublist) for sublist in list1}
    tuples2 = {tuple(sublist) for sublist in list2}
    tuples = tuples1-tuples2
    lists = [list(sublist) for sublist in tuples]
    return lists


def is_transition_metal(atom):
    """define transition metals by their atomic number

    For the purpose of a motif in the template library of ligands, the
    dummy atom `*` equally should be processed as if it were a transition
    metal.  By convention, its atomic number is 0."""
    n = atom.GetAtomicNum()
    return (
        (n >= 22 and n <= 29)
        or (n >= 40 and n <= 47)
        or (n >= 72 and n <= 79)
        or (n == 0)
    )

def reset_dative_bonds(mol, fromAtoms=(6, 7, 8, 15, 16)):  # i.e., C, N, O, P, S
    """edit some "dative bonds"

    Bonds between atoms of transition metals typical donor atoms will be marked
    as single bonds.  Initially inspired by the RDKit Cookbook[1] depicting an
    example with pointy arrows, a subsequent discussion in RDKit's user forum[2]
    convinced nbehrnd to drop this approach in favor of plain bonds.

    [1] http://rdkit.org/docs/Cookbook.html#organometallics-with-dative-bonds
    [2] https://github.com/rdkit/rdkit/discussions/6995

    Returns the modified molecule.
    """
    pt = Chem.GetPeriodicTable()
    rwmol = Chem.RWMol(mol)
    rwmol.UpdatePropertyCache(strict=False)
    metals = [at for at in rwmol.GetAtoms() if is_transition_metal(at)]
    for metal in metals:
        for nbr in metal.GetNeighbors():
            if (
                nbr.GetAtomicNum() in fromAtoms
                and rwmol.GetBondBetweenAtoms(
                    nbr.GetIdx(), metal.GetIdx()
                ).GetBondType()
                == Chem.BondType.SINGLE
            ):
                rwmol.RemoveBond(nbr.GetIdx(), metal.GetIdx())
                rwmol.AddBond(nbr.GetIdx(), metal.GetIdx(), Chem.BondType.DATIVE)
    return rwmol