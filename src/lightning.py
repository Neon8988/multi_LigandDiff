import numpy as np
import pandas as pd
import os
import pytorch_lightning as pl
import torch
import wandb
from rdkit import Chem
from src import const
from src import utils
from src.egnn import Dynamics
from src.edm import EDM
from src.visualizer import visualize_chain
from src.molecule_builder import BasicLigandMetrics, build_mol,extract_ligand_index,sanitycheck,write_xyz_file
from typing import Dict, List, Optional
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_add
import sys
sys.path.append('/mnt/gs21/scratch/jinhongn/pyg/molSimplify')
from molSimplify.Classes.mol3D import mol3D
from molSimplify.Classes.ligand import ligand_breakdown
        

class DDPM(pl.LightningModule):
    train_dataset = None
    val_dataset = None
    test_dataset = None
    starting_epoch = None
    metrics: Dict[str, List[float]] = {}

    FRAMES = 100

    def __init__(self,
        data_path, train_data, val_data,
        in_node_nf, n_dims, ligand_node_nf,
        hidden_nf, activation,n_layers,attention,
        normalization_factor,
        diffusion_steps, diffusion_noise_schedule, diffusion_noise_precision, diffusion_loss_type,
        normalize_factors,
        lr,batch_size,torch_device, model,test_epochs, n_stability_samples,
        samples_dir=None,
        center_of_mass='context',clip_grad=True,drop_rate=0.0
    ):
        super(DDPM, self).__init__()

        self.save_hyperparameters()
        self.data_path = data_path
        self.train_data = train_data
        self.val_data = val_data
        
        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        
        self.batch_size = batch_size
        self.lr = lr
        self.torch_device = torch_device
        self.test_epochs = test_epochs
        self.n_stability_samples = n_stability_samples
        self.samples_dir = samples_dir
        
        self.center_of_mass = center_of_mass
        self.loss_type = diffusion_loss_type
        self.T=diffusion_steps
        self.clip_grad=clip_grad
        if self.clip_grad:
            self.gradnorm_queue = utils.Queue()
            self.gradnorm_queue.add(3000)
        
        self.ligand_metrics=BasicLigandMetrics()
        
        # save targets in each batch to compute metric over one epoch
        self.training_step_outputs = []   
        self.validation_step_outputs = []
        self.test_step_outputs = []

        dynamics = Dynamics(
            in_node_nf=in_node_nf,
            n_dims=n_dims,
            ligand_node_nf=ligand_node_nf,
            hidden_nf=hidden_nf,
            activation=activation,
            n_layers=n_layers,
            attention=attention,
            normalization_factor=normalization_factor,
            device=torch_device,
            model=model,
            drop_rate=drop_rate
        )

        self.edm = EDM(
            dynamics=dynamics,
            in_node_nf=in_node_nf,
            n_dims=n_dims,
            timesteps=diffusion_steps,
            noise_schedule=diffusion_noise_schedule,
            noise_precision=diffusion_noise_precision,
            loss_type=diffusion_loss_type,
            norm_values=normalize_factors,
        )

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            self.train_dataset=torch.load(f'{self.data_path}/{self.train_data}.pt',map_location=self.torch_device)
            self.val_dataset=torch.load(f'{self.data_path}/{self.val_data}.pt',map_location=self.torch_device)
        elif stage == 'val':   
            self.val_dataset = torch.load(f'{self.data_path}/{self.val_data}.pt',map_location=self.torch_device)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=0, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size)

    
    def forward(self, data):
        x = data['pos']
        h = data['one_hot']
        coord_site = data['coord_site'].view(-1,1)
        context = data['context'].view(-1,1)
        ligand_diff = data['ligand_diff'].view(-1,1)
        ligand_group=data['ligand_group']
        batch_seg=data.batch
        batch_size=int(torch.max(batch_seg))+1
        ligand_site=torch.cat([ligand_group,coord_site],dim=-1)
        
        #Removing COM of fragment from the atom coordinates
        if self.center_of_mass == 'context':
            x = utils.remove_partial_mean_with_mask(x,context,batch_seg)
        elif self.center_of_mass == 'ligand_diff':
            x = utils.remove_partial_mean_with_mask(x,ligand_diff,batch_seg)
        else:
            raise ValueError(f'Unknown center_of_mass: {self.center_of_mass}')
        delta_log_px, error_t, SNR_weight,loss_0_x, loss_0_h, neg_log_const_0,\
        kl_prior= self.edm.forward(
            x=x,
            h=h,
            context=context,
            ligand_diff=ligand_diff,
            batch_seg=batch_seg,
            batch_size=batch_size,
            ligand_site=ligand_site
        )
        if self.loss_type == 'l2' and self.training:
            #normalize loss_t
            normalization=(self.n_dims + self.in_node_nf)*EDM.inflate_batch_array(ligand_diff, batch_seg)
            error_t=error_t/normalization
            loss_t=error_t
            #normaliza loss_0
            loss_0_x=loss_0_x/self.n_dims*EDM.inflate_batch_array(ligand_diff, batch_seg)
            loss_0=loss_0_x+loss_0_h
        
        else:
            loss_t = self.T * 0.5 * SNR_weight * error_t
            loss_0 = loss_0_x + loss_0_h
            loss_0 = loss_0 + neg_log_const_0

        nll=loss_t + loss_0 + kl_prior

        if not (self.loss_type == 'l2' and self.training):
            nll=nll-delta_log_px
        
        metrics={'error_t':error_t.mean(0),
        'SNR_weight': SNR_weight.mean(0),
        'loss_0':loss_0.mean(0),
        'kl_prior':kl_prior.mean(0),
        'delta_log_px':delta_log_px.mean(0),
        'neg_log_const_0': neg_log_const_0.mean(0)}
        return nll, metrics
            
    def log_metrics(self, metrics_dict, split, batch_size=None, **kwargs):
        for m, value in metrics_dict.items():
            self.log(f'{m}/{split}', value, batch_size=batch_size, **kwargs)

    def training_step(self, data, *args):
        try:
            nll, metrics = self.forward(data)
        except RuntimeError as e:
            # this is not supported for multi-GPU
            if self.trainer.num_devices < 2 and 'out of memory' in str(e):
                print('WARNING: ran out of memory, skipping to the next batch')
                return None
            else:
                raise e

        loss = nll.mean(0)
        metrics['loss']=loss
        self.log_metrics(metrics, 'train', batch_size=int(torch.max(data.batch))+1)
        self.training_step_outputs.append(metrics)
        return metrics

    def _shared_eval_step(self, data, prefix, *args):
        nll, metrics = self.forward(data)
        loss = nll.mean(0)
        metrics['loss'] = loss
        self.log_metrics(metrics, prefix, batch_size=torch.max(data.batch)+1,sync_dist=True)
        if prefix == 'val':
            self.validation_step_outputs.append(metrics)
        else:
            self.test_step_outputs.append(metrics)
        return metrics
    
    def validation_step(self, data, *args):
        self._shared_eval_step(data, 'val', *args)

    def test_step(self, data, *args):
        self._shared_eval_step(data, 'test', *args)

    def on_train_epoch_end(self):
        for metric in self.training_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(self.training_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/train', []).append(avg_metric)
            self.log(f'{metric}/train', avg_metric, prog_bar=True)
        self.training_step_outputs.clear()
    def on_validation_epoch_end(self):
        for metric in self.validation_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(self.validation_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/val', []).append(avg_metric)
            self.log(f'{metric}/val', avg_metric, prog_bar=True)
        print(f"=======current epoch on validation:{self.current_epoch} and current val_loss:{self.metrics['loss/val'][-1]}")
        if (self.current_epoch + 1) % self.test_epochs == 0:
            sampling_results = self.sample_and_analyze(self.val_dataset,animation=True)
            for metric_name, metric_value in sampling_results.items():
                self.log(f'{metric_name}/val', metric_value, prog_bar=True)
                self.metrics.setdefault(f'{metric_name}/val', []).append(metric_value)

            # Logging the results corresponding to the best validation_and_connectivity
            best_metrics, best_epoch = self.compute_best_validation_metrics()
            self.log('best_epoch', int(best_epoch), prog_bar=True, batch_size=self.batch_size)
            for metric, value in best_metrics.items():
                self.log(f'best_{metric}', value, prog_bar=True, batch_size=self.batch_size)
        self.validation_step_outputs.clear()
    def test_epoch_end(self):
        for metric in self.test_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(self.test_step_outputs, metric)
            self.log(f'{metric}/test', avg_metric, prog_bar=True)
        self.test_step_outputs.clear()
    
    
    def generate_animation(self, chain_batch, batch_i,batch_seg,metals):
        
        idx2atom = const.IDX2ATOM
        idx2metals=const.idx2metals
        # only visualize the first molcule in the batch
        pos = chain_batch[:,batch_seg==0,:3]
        onehot = chain_batch[:,batch_seg==0,3:]
        n_atoms =pos.shape[1]
        name = f'mol_{batch_i}'
        chain_output = os.path.join(self.samples_dir, f'epoch_{self.current_epoch}', name)
        os.makedirs(chain_output, exist_ok=True)
        for j in range(self.FRAMES):
            f = open(os.path.join(chain_output, f'{name}_{j}.xyz'), "w")
            f.write("%d\n\n" % n_atoms)
            atoms = torch.argmax(onehot[j], dim=1)
            for atom_i in range(n_atoms):
                if atom_i==0:
                    atom=idx2metals[metals[0].item()]
                else:
                    atom = idx2atom[atoms[atom_i].item()]
                f.write("%s %.5f %.5f %.5f\n" % (
                    atom, pos[j][atom_i, 0], pos[j][atom_i, 1], pos[j][atom_i, 2]
                ))
            f.close()
        visualize_chain(chain_output, wandb=wandb, mode=name)
    
    
    @torch.no_grad()
    def sample_and_analyze(self, dataset,batch_size=None, outdir=None,animation=False):
        batch_size=self.batch_size if batch_size is None else batch_size
        valid_comp=0
        total_ligands=0
        connectivity=0
        validity=0
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
        for b, data in enumerate(dataloader):
            pos_orginal=data['pos']
            h_orginal=data['one_hot']
            batch_seg=data.batch
            batch_size=torch.max(batch_seg)+1
            ligand_diffs = data['ligand_diff'].view(-1,1)
            contexts = data['context'].view(-1,1)
            ligand_groups=data['ligand_group']
            metals=[data['nuclear_charges'][batch_seg==i][0] for i in range(batch_size)]
            labels=data['label']
            natoms=data['num_atoms']
            fixed_mean = scatter_add(pos_orginal*contexts, batch_seg, dim=0)/scatter_add(contexts, batch_seg, dim=0).view(-1,1)
            for sample_idx in range(self.n_stability_samples):
                try:
                    chain_batch = self.sample_chain(data, keep_frames=self.FRAMES)
                except utils.FoundNaNException as e:
                    continue
                
                if animation and self.samples_dir is not None and sample_idx == 0 and b in [0, 1]:
                    self.generate_animation(chain_batch=chain_batch, batch_i=b,batch_seg=batch_seg,metals=metals)
                # Get final molecules from chains – for computing metrics
                x = chain_batch[0][ :, :3]
                x=x+fixed_mean[batch_seg]
                one_hot = chain_batch[0][ :, 3:]
                assert one_hot.shape[1]==self.in_node_nf
                unique_indices = torch.unique(batch_seg)
                for i in unique_indices:
                    positions=x[batch_seg==i]
                    atom_types=one_hot[batch_seg==i].argmax(dim=1)
                    metal=metals[i]
                    overlapping,liglist=sanitycheck(positions, atom_types,metal)
                    total_atoms=sum(len(lig) for lig in liglist)+1
                    if not overlapping and total_atoms==natoms[i].item(): 
                        ligand_diff=ligand_diffs[batch_seg==i].squeeze()
                        ligand_group=ligand_groups[batch_seg==i]
                        ligand_diff_group=extract_ligand_index(ligand_diff,ligand_group)
                        ligand_diff_indexs = [item for sublist in ligand_diff_group for item in sublist]
                        ligand_generated=[lig for lig in liglist if any(x in lig for x in ligand_diff_indexs)]
                        rdmols=[build_mol(positions[lig],atom_types[lig]) for lig in ligand_generated]
                        total_ligands+=len(rdmols)
                        valid= self.ligand_metrics.compute_validity(rdmols)
                        validity+=len(valid)
                        connected=self.ligand_metrics.compute_connectivity(valid)
                        connectivity+=len(connected)
                        if len(connected)==len(rdmols):
                            valid_comp+=1
                    
                            #write_xyz_file(positions, atom_types,f'{outdir}/{b}_{i}',metal,'N/A')
        
        
        metrics={'valid_ligand':validity/total_ligands,
                 'connected_ligand':connectivity/validity,
                 'valid_complex':valid_comp/len(dataset)}
        print(f'Finished sampling on  {len(dataset)} samples')
        print('Metrics for sampling:')
        print(metrics)
        return metrics               

    def sample_chain(self, data,  keep_frames=None):

        x = data['pos']
        h = data['one_hot']
        coord_site = data['coord_site'].view(-1,1)
        context = data['context'].view(-1,1)
        ligand_diff = data['ligand_diff'].view(-1,1)
        batch_seg=data.batch
        batch_size=int(torch.max(batch_seg))+1
        ligand_group=data['ligand_group']
        ligand_site = torch.cat([ligand_group,coord_site],dim=-1)

        if self.center_of_mass == 'context':
            x= utils.remove_partial_mean_with_mask(x,context,batch_seg)
        
        chain = self.edm.sample_chain(
            x=x,
            h=h,
            context=context,
            ligand_diff=ligand_diff,
            batch_seg=batch_seg,
            batch_size=batch_size,
            ligand_site=ligand_site,
            keep_frames=keep_frames)
        
        return chain 

    def configure_optimizers(self):
        return torch.optim.AdamW(self.edm.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-6)


    def configure_gradient_clipping(self, optimizer,gradient_clip_val, gradient_clip_algorithm):
                                
        if not self.clip_grad:
            return

        # Allow gradient norm to be 150% + 2 * stdev of the recent history.
        max_grad_norm = 1.5 * self.gradnorm_queue.mean() + \
                        2 * self.gradnorm_queue.std()

        # Get current grad_norm
        params = [p for g in optimizer.param_groups for p in g['params']]
        grad_norm = utils.get_grad_norm(params)

        # Lightning will handle the gradient clipping
        self.clip_gradients(optimizer, gradient_clip_val=max_grad_norm,
                            gradient_clip_algorithm='norm')

        if float(grad_norm) > max_grad_norm:
            self.gradnorm_queue.add(float(max_grad_norm))
        else:
            self.gradnorm_queue.add(float(grad_norm))

        if float(grad_norm) > max_grad_norm:
            print(f'Clipped gradient with value {grad_norm:.1f} '
                  f'while allowed {max_grad_norm:.1f}')
            

    def compute_best_validation_metrics(self):
        loss = self.metrics[f'valid_complex/val']
        best_epoch = np.argmax(loss)
        best_metrics = {
            metric_name: metric_values[best_epoch]
            for metric_name, metric_values in self.metrics.items()
            if metric_name.endswith('/val')
        }
        return best_metrics, best_epoch

    @staticmethod
    def aggregate_metric(step_outputs, metric):
        return torch.tensor([out[metric] for out in step_outputs]).mean()
    


   

