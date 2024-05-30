# multi-LigandDiff
This repository is an extension of [LigandDiff](https://github.com/Neon8988/LigandDiff).
<img src="https://github.com/Neon8988/multi_LigandDiff/blob/master/image/toc.png" width="480">

# Dataset
Download all datasets from this [link](https://zenodo.org/records/11397730)
# Generation
To generate ligands, run generate.py with your own complex in .xyz format.
```python
python generate.py --model model/pre_trained.ckpt --outdir generate --complex PEQNAB01_comp_1.xyz --add_Hs False
