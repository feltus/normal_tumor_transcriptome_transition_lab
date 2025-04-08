# normal_tumor_transcriptome_transition_lab

# Background.  
The GEMDiff repository contains the code for the diffusion model and a neural network model for a breast cancer study case.  The results of this study can be found in [X Ai et al](https://academic.oup.com/bib/article/26/2/bbaf093/8069412?utm_source=advanceaccess&utm_campaign=bib&utm_medium=email&login=true)] and our [website(https://xai990.github.io/)].

# Useful genrative AI prompts
What is a diffusion model?
What is RNAseq?
What is the GTEX project?
What is the TCGA project?

# Gene Expression Matrix Preprocessing
Any normalized gene expression matrix (GEM) that contains RNAseq data for two groups should work.  Here is a repository to obtain and preprocess several co-normalized GTEX_NORMAL, TCGA_NORMAL, and TCGA_TUMOR GEMs [gembuld(https://github.com/feltus/gembuild)]. This Workflow will prepare a series of normal (GTEX) and tunor (TCGA) co-normalized gene expression matrices (GEMs) from Wang et al (https://pubmed.ncbi.nlm.nih.gov/29664468/).  GEMs for comparable groups (e.g. NORMAL_GTEX_BREAST,TCGA_TUMOR_BRCA) will be mixed and separated into train and test GEMs for AI/ML applications.  The steps in this process are as follows:

```
Download GEMs
Merge GEMs from Wang et al
Transpose the GEMs
Split GEMs into test and train sets
Convert GEMS to tab-delimited format
Make  group label files (e.g. BRCAT, BREAN)
Draw a histogram of all GEMs
```

Here are the available GEMs with a potential group label:
```
bladder-rsem-fpkm-gtex.txt	BLADN
blca-rsem-fpkm-tcga-t.txt	BLCAT
blca-rsem-fpkm-tcga.txt	BLCAN
brca-rsem-fpkm-tcga-t.txt	BRCAT
brca-rsem-fpkm-tcga.txt	BRCAN
breast-rsem-fpkm-gtex.txt	BREAN
cervix-rsem-fpkm-gtex.txt	CERVN
cesc-rsem-fpkm-tcga-t.txt	CESCT
cesc-rsem-fpkm-tcga.txt	CESCN
chol-rsem-fpkm-tcga-t.txt	CHOLT
chol-rsem-fpkm-tcga.txt	CHOLN
coad-rsem-fpkm-tcga-t.txt	COADT
coad-rsem-fpkm-tcga.txt	COADN
colon-rsem-fpkm-gtex.txt	COLON
esca-rsem-fpkm-tcga-t.txt	ESCAT
esca-rsem-fpkm-tcga.txt	ESCAN
esophagus_gas-rsem-fpkm-gtex.txt	ESOGN
esophagus_muc-rsem-fpkm-gtex.txt	ESOCN
esophagus_mus-rsem-fpkm-gtex.txt	ESOSN
hnsc-rsem-fpkm-tcga-t.txt	HNSCT
hnsc-rsem-fpkm-tcga.txt	HNSCN
kich-rsem-fpkm-tcga-t.txt	KICHT
kich-rsem-fpkm-tcga.txt	KICHN
kidney-rsem-fpkm-gtex.txt	KIDNN
kirc-rsem-fpkm-tcga-t.txt	KIRCT
kirc-rsem-fpkm-tcga.txt	KIRCN
kirp-rsem-fpkm-tcga-t.txt	KIRPT
kirp-rsem-fpkm-tcga.txt	KIRPN
lihc-rsem-fpkm-tcga-t.txt	LIHCT
lihc-rsem-fpkm-tcga.txt	LIHCN
liver-rsem-fpkm-gtex.txt	LIVEN
luad-rsem-fpkm-tcga-t.txt	LUADT
luad-rsem-fpkm-tcga.txt	LUADN
lung-rsem-fpkm-gtex.txt	LUNGN
lusc-rsem-fpkm-tcga-t.txt	LUSCT
lusc-rsem-fpkm-tcga.txt	LUSCN
prad-rsem-fpkm-tcga-t.txt	PRADT
prad-rsem-fpkm-tcga.txt	PRADN
prostate-rsem-fpkm-gtex.txt	PROSN
read-rsem-fpkm-tcga-t.txt	READT
read-rsem-fpkm-tcga.txt	READN
salivary-rsem-fpkm-gtex.txt	SALIN
stad-rsem-fpkm-tcga-t.txt	STADT
stad-rsem-fpkm-tcga.txt	STADN
stomach-rsem-fpkm-gtex.txt	STOMN
thca-rsem-fpkm-tcga-t.txt	THCAT
thca-rsem-fpkm-tcga.txt	THCAN
thyroid-rsem-fpkm-gtex.txt	THYRN
ucec-rsem-fpkm-tcga-t.txt	UCECT
ucec-rsem-fpkm-tcga.txt	UCECN
ucs-rsem-fpkm-tcga-t.txt	UCST
uterus-rsem-fpkm-gtex.txt	UTERN
```

# Selecting a gene set
As you begin to train AI models with more than more than 64 genes, you begin to see random genes clustering samples.  We call this background classification potential which is described here: 

Targonski C, Shearer CA, Shealy BT, Smith MC and Feltus FA. 2019. Uncovering biomarker genes with enriched classification potential from Hallmark gene sets. Sci Rep 9: 9747. doi: 10.1038/s41598-019-46059-1.

Thus it is important to pre-select a gene set that is less than 64 genes for testing the hypotesis that GENESET X is discriminator of the two groups and likely to be enriched for causla genetic factors that differentiate the two sample states.  Always compare the results of the target genes with an equal number of random genes.  

Here is an exampl gene set (THCATOP20MUTATE) that are the most mutated genes in thyroid cancer:

```
THCATOP20MUTATE	BRAF	NRAS	HRAS	MUC16	ZFHX3	EIF1AX	KMT2A	AKT1	ATM	KMT2C	KRAS	CSMD3	WNK2	PPM1D	CHEK2	ARID2	CUX1	TNC	DICER1	DNMT3A
```

# Training the GEMDiff model
Once you have preprocessed GEMs, can install and run [GEMDiff (https://github.com/xai990/GEMDiff)].  You will need train and test GEMS, a gene list, and a config file.  Make sure to put all these datasets into a directory path that will be visible to GEMDiff.

Here is an example of a config file:
```
data:
  data_dir: "datasets/"
  dir_out: "results"
  train_path: "datasets/BLADN_BLCAT.train.log2" 
  train_label_path: "datasets/BLADN_BLCAT.train.label"
  test_path: "datasets/BLADN_BLCAT.test.log2"
  test_label_path: "datasets/BLADN_BLCAT.test.label" 
  filter: null
  corerate: 1 
  
model:
  class_cond: True
  dropout: 0.0
  n_layer: 4
  n_head: 2
  feature_size: 20
  
diffusion:
  noise_schedule: "cosine"
  linear_start: 0.0001 
  linear_end: 0.0195
  diffusion_steps: 1000
  log_every_t: 10
  learn_sigma: False
  
train:
  lr: 0.00003
  # num_epoch: 1
  batch_size: 16
  schedule_plot: False
  # log_interval: 100
```

Here is a SLURM script written for Palmetto2 to obtain an A100 GPU node for rapid training.  Note that the SLURM script clones GEMDiff into a conda environment called GEMDiff.  You can create the GEMDiff environment

for you and installs it.
```
#!/bin/bash

#SBATCH --job-name=TRAIN     # Set the job name
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --gpus a100:1
#SBATCH --mem 32gb
#SBATCH --time 24:00:00

#load modules
module load anaconda3/2023.09-0

#Go to working directory
cd /scratch/ffeltus/emt/GEMDIFF/BLCA

# activate the created conda environment
source activate GEMDiff  #create before running script

# clone the package
rm GEMDiff -rf
git clone https://github.com/xai990/GEMDiff.git
cd GEMDiff
pip install -e . 

#Move datasets to GEMDiff
cp ../datasets . -r

#train the model
python scripts/train.py --config datasets/THCA.yaml --dir log --gene_set datasets/THCATOP20MUTATE.txt
```
Here are excerpts example training log file.  Note the trained model path is located towrds the end of the file on the line with 'Saved checkpoint':

```
Cloning into 'GEMDiff'...
.
.
.
The args gene set is: datasets/THCATOP20MUTATE.txt
reading input data from BLADN_BLCAT.train.log2
loaded input data has 19427 genes, 261 samples
The input gene set is THCATOP20MUTATE, contains ['BRAF', 'NRAS', 'HRAS', 'MUC16', 'ZFHX3', 'EIF1AX', 'KMT2A', 'AKT1', 'ATM', 'KMT2C', 'KRAS', 'CSMD3', 'WNK2', 'PPM1D', 'CHEK2', 'ARID2', 'CUX1', 'TNC', 'DICER1', 'DNMT3A'] genes
The intersection geneset is: Index(['AKT1', 'ARID2', 'ATM', 'BRAF', 'CHEK2', 'CSMD3', 'CUX1', 'DICER1',
       'DNMT3A', 'EIF1AX', 'HRAS', 'KMT2A', 'KMT2C', 'KRAS', 'MUC16', 'NRAS',
       'PPM1D', 'TNC', 'WNK2', 'ZFHX3'],
      dtype='object') -- dataset
loaded selected data has 20 genes, 261 samples
loaded input data has 2 classes
The selected genes are: Index(['BRAF', 'KRAS', 'ATM', 'PPM1D', 'MUC16', 'CSMD3', 'CHEK2', 'AKT1',
       'DNMT3A', 'WNK2', 'TNC', 'KMT2A', 'HRAS', 'EIF1AX', 'DICER1', 'ARID2',
       'ZFHX3', 'KMT2C', 'CUX1', 'NRAS'],
      dtype='object') -- dataset
After data pre-processing, the dataset contains 20 gene.
The size of train dataset: (261, 20)

train the model:
The loss is : 1.1385488510131836 at 0 epoch
The loss is : 0.061073221266269684 at 500 epoch
The loss is : 0.05706595256924629 at 1000 epoch
The loss is : 0.05757423862814903 at 1500 epoch
The loss is : 0.03685588017106056 at 2000 epoch
The loss is : 0.07838549464941025 at 2500 epoch
The loss is : 0.21263229846954346 at 3000 epoch
The loss is : 0.16671980917453766 at 3500 epoch
The loss is : 0.024001337587833405 at 4000 epoch
The loss is : 0.14863793551921844 at 4500 epoch
The loss is : 0.05309063196182251 at 5000 epoch
The loss is : 0.03520705923438072 at 5500 epoch
The loss is : 0.1990010291337967 at 6000 epoch
The loss is : 0.016131063923239708 at 6500 epoch
The loss is : 0.018425656482577324 at 7000 epoch
The loss is : 0.019640332087874413 at 7500 epoch
.
.
.
The loss is : 0.11051638424396515 at 8000 epoch
The loss is : 0.028280073776841164 at 8500 epoch
The loss is : 0.034478265792131424 at 9000 epoch
The loss is : 0.02980395220220089 at 9500 epoch
The loss is : 0.041882146149873734 at 10000 epoch
Saved checkpoint to log/2025-04-02-11-06/model10000.pt
training process completed
```

# Perturbing from Tumor to Normal Gene Expression State
Once the model is trained from the training GEMs that cotains both labeled groups, one can use GEMDiff to simulate the transition between groups.  The results of this simulation are visualized with a UMAP plot and the most perturbed genes can be identified in the log file. 
```
#!/bin/bash

#SBATCH --job-name=PERTURB      # Set the job name
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --gpus a100:1
#SBATCH --mem 32gb
#SBATCH --time 04:00:00

#load modules
module load anaconda3/2023.09-0

#Go to working directory
cd /scratch/ffeltus/emt/GEMDIFF/THCA/GEMDiff

# activate the created conda environment
source activate GEMDiff  #create before running script

#perturb samples
python scripts/perturb.py --config datasets/THCA.yaml --dir log  --model_path log/2025-04-02-11-06/model10000.pt --valid --gene_set datasets/THCATOP20MUTATE.txt ###Make sure to change the model path which is found in the train log file.

# Visulaize the results.

The log file will contain the most perturbed genes and the lod directory will contain a UMAP plot of the input groups and perturbed samples.  Here is an example of the perturbation log file:


```
/home/ffeltus/.conda/envs/GEMDiff/lib/python3.11/site-packages/numba/np/ufunc/dufunc.py:343: NumbaWarning: Compilation requested for previously compiled argument types ((uint32,)). This has no effect and perhaps indicates a bug in the calling code (compiling a ufunc more than once for the same signature
  warnings.warn(msg, errors.NumbaWarning)
/home/ffeltus/.conda/envs/GEMDiff/lib/python3.11/site-packages/numba/np/ufunc/dufunc.py:343: NumbaWarning: Compilation requested for previously compiled argument types ((uint32,)). This has no effect and perhaps indicates a bug in the calling code (compiling a ufunc more than once for the same signature
  warnings.warn(msg, errors.NumbaWarning)
/home/ffeltus/.conda/envs/GEMDiff/lib/python3.11/site-packages/numba/np/ufunc/dufunc.py:343: NumbaWarning: Compilation requested for previously compiled argument types ((uint32,)). This has no effect and perhaps indicates a bug in the calling code (compiling a ufunc more than once for the same signature
  warnings.warn(msg, errors.NumbaWarning)
Logging to log/2025-04-02-14-24
**********************************
log configure
device information:cuda:0
Load the config...
reading input data from BLADN_BLCAT.train.log2
loaded input data has 19427 genes, 261 samples
The input gene set is THCATOP20MUTATE, contains ['BRAF', 'NRAS', 'HRAS', 'MUC16', 'ZFHX3', 'EIF1AX', 'KMT2A', 'AKT1', 'ATM', 'KMT2C', 'KRAS', 'CSMD3', 'WNK2', 'PPM1D', 'CHEK2', 'ARID2', 'CUX1', 'TNC', 'DICER1', 'DNMT3A'] genes
The intersection geneset is: Index(['AKT1', 'ARID2', 'ATM', 'BRAF', 'CHEK2', 'CSMD3', 'CUX1', 'DICER1',
       'DNMT3A', 'EIF1AX', 'HRAS', 'KMT2A', 'KMT2C', 'KRAS', 'MUC16', 'NRAS',
       'PPM1D', 'TNC', 'WNK2', 'ZFHX3'],
      dtype='object') -- dataset
loaded selected data has 20 genes, 261 samples
loaded input data has 2 classes
The selected genes are: Index(['BRAF', 'KRAS', 'ATM', 'PPM1D', 'MUC16', 'CSMD3', 'CHEK2', 'AKT1',
       'DNMT3A', 'WNK2', 'TNC', 'KMT2A', 'HRAS', 'EIF1AX', 'DICER1', 'ARID2',
       'ZFHX3', 'KMT2C', 'CUX1', 'NRAS'],
      dtype='object') -- dataset
After data pre-processing, the dataset contains 20 gene.
reading input data from BLADN_BLCAT.test.log2
loaded input data has 19427 genes, 112 samples
The input gene set is THCATOP20MUTATE, contains ['BRAF', 'NRAS', 'HRAS', 'MUC16', 'ZFHX3', 'EIF1AX', 'KMT2A', 'AKT1', 'ATM', 'KMT2C', 'KRAS', 'CSMD3', 'WNK2', 'PPM1D', 'CHEK2', 'ARID2', 'CUX1', 'TNC', 'DICER1', 'DNMT3A'] genes
The intersection geneset is: Index(['AKT1', 'ARID2', 'ATM', 'BRAF', 'CHEK2', 'CSMD3', 'CUX1', 'DICER1',
       'DNMT3A', 'EIF1AX', 'HRAS', 'KMT2A', 'KMT2C', 'KRAS', 'MUC16', 'NRAS',
       'PPM1D', 'TNC', 'WNK2', 'ZFHX3'],
      dtype='object') -- dataset
loaded selected data has 20 genes, 112 samples
loaded input data has 2 classes
The selected genes are: Index(['BRAF', 'KRAS', 'ATM', 'PPM1D', 'MUC16', 'CSMD3', 'CHEK2', 'AKT1',
       'DNMT3A', 'WNK2', 'TNC', 'KMT2A', 'HRAS', 'EIF1AX', 'DICER1', 'ARID2',
       'ZFHX3', 'KMT2C', 'CUX1', 'NRAS'],
      dtype='object') -- dataset
creating model and diffusion ... 
The model feature size is : 20
{'data': {'data_dir': 'datasets/', 'dir_out': 'results', 'gene_selection': None, 'drop_fraction': 0, 'train_path': 'datasets/BLADN_BLCAT.train.log2', 'train_label_path': 'datasets/BLADN_BLCAT.train.label', 'test_path': 'datasets/BLADN_BLCAT.test.log2', 'test_label_path': 'datasets/BLADN_BLCAT.test.label', 'filter': None, 'corerate': 1}, 'train': {'microbatch': 16, 'log_interval': 1000, 'save_interval': 20000, 'schedule_plot': False, 'resume_checkpoint': '', 'ema_rate': 0.9999, 'num_epoch': 40001, 'schedule_sampler': 'uniform', 'lr': 3e-05, 'batch_size': 16}, 'perturb': {}, 'umap': {'n_neighbors': 90, 'min_dist': 0.3}, 'model': {'feature_size': 20, 'patch_size': 20, 'dropout': 0.0, 'class_cond': True, 'n_embd': 160, 'n_head': 2, 'n_layer': 4}, 'diffusion': {'diffusion_steps': 1000, 'noise_schedule': 'cosine', 'linear_start': 0.0001, 'linear_end': 0.0195, 'log_every_t': 10, 'schedule_sampler': 'uniform', 'learn_sigma': False, 'rescale_timesteps': True, 'timestep_respacing': ''}}
number of parameters: 1.45M
/scratch/ffeltus/emt/GEMDIFF/BLCA/GEMDiff/scripts/perturb.py:84: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  state_dict = th.load(f"{args.model_path}")
pertubing the source to target

Sampling latent:   0%|                                                                                                                            | 0/1000 [00:00<?, ?it/s]
Sampling latent:   0%|                                                                                                                    | 1/1000 [00:02<36:46,  2.21s/it]
Sampling latent:   4%|████▎                                                                                                              | 38/1000 [00:02<00:42, 22.87it/s]
Sampling latent:   8%|████████▋                                                                                                          | 75/1000 [00:02<00:18, 50.52it/s]
Sampling latent:  11%|████████████▊                                                                                                     | 112/1000 [00:02<00:10, 83.09it/s]
Sampling latent:  15%|████████████████▊                                                                                                | 149/1000 [00:02<00:07, 119.51it/s]
Sampling latent:  19%|█████████████████████                                                                                            | 186/1000 [00:02<00:05, 157.92it/s]
Sampling latent:  22%|█████████████████████████▏                                                                                       | 223/1000 [00:02<00:03, 195.96it/s]
Sampling latent:  26%|█████████████████████████████▍                                                                                   | 260/1000 [00:02<00:03, 231.33it/s]
Sampling latent:  30%|█████████████████████████████████▌                                                                               | 297/1000 [00:03<00:02, 262.47it/s]
Sampling latent:  33%|█████████████████████████████████████▋                                                                           | 334/1000 [00:03<00:02, 288.39it/s]
Sampling latent:  37%|█████████████████████████████████████████▉                                                                       | 371/1000 [00:03<00:02, 309.14it/s]
Sampling latent:  41%|██████████████████████████████████████████████                                                                   | 408/1000 [00:03<00:01, 324.55it/s]
Sampling latent:  44%|██████████████████████████████████████████████████▎                                                              | 445/1000 [00:03<00:01, 336.81it/s]
Sampling latent:  48%|██████████████████████████████████████████████████████▍                                                          | 482/1000 [00:03<00:01, 345.84it/s]
Sampling latent:  52%|██████████████████████████████████████████████████████████▋                                                      | 519/1000 [00:03<00:01, 351.94it/s]
Sampling latent:  56%|██████████████████████████████████████████████████████████████▊                                                  | 556/1000 [00:03<00:01, 356.78it/s]
Sampling latent:  59%|███████████████████████████████████████████████████████████████████                                              | 593/1000 [00:03<00:01, 360.22it/s]
Sampling latent:  63%|███████████████████████████████████████████████████████████████████████▏                                         | 630/1000 [00:03<00:01, 362.69it/s]
Sampling latent:  67%|███████████████████████████████████████████████████████████████████████████▎                                     | 667/1000 [00:04<00:00, 364.53it/s]
Sampling latent:  70%|███████████████████████████████████████████████████████████████████████████████▌                                 | 704/1000 [00:04<00:00, 365.71it/s]
Sampling latent:  74%|███████████████████████████████████████████████████████████████████████████████████▋                             | 741/1000 [00:04<00:00, 366.50it/s]
Sampling latent:  78%|███████████████████████████████████████████████████████████████████████████████████████▉                         | 778/1000 [00:04<00:00, 367.20it/s]
Sampling latent:  82%|████████████████████████████████████████████████████████████████████████████████████████████                     | 815/1000 [00:04<00:00, 367.57it/s]
Sampling latent:  85%|████████████████████████████████████████████████████████████████████████████████████████████████▎                | 852/1000 [00:04<00:00, 367.79it/s]
Sampling latent:  89%|████████████████████████████████████████████████████████████████████████████████████████████████████▍            | 889/1000 [00:04<00:00, 367.89it/s]
Sampling latent:  93%|████████████████████████████████████████████████████████████████████████████████████████████████████████▋        | 926/1000 [00:04<00:00, 368.02it/s]
Sampling latent:  96%|████████████████████████████████████████████████████████████████████████████████████████████████████████████▊    | 963/1000 [00:04<00:00, 367.97it/s]
Sampling latent: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:04<00:00, 367.94it/s]
Sampling latent: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:04<00:00, 203.06it/s]

Sampling t:   0%|                                                                                                                                 | 0/1000 [00:00<?, ?it/s]
Sampling t:   0%|                                                                                                                         | 1/1000 [00:00<07:09,  2.32it/s]
Sampling t:   3%|████                                                                                                                    | 34/1000 [00:00<00:11, 83.84it/s]
Sampling t:   7%|████████                                                                                                               | 68/1000 [00:00<00:06, 150.54it/s]
Sampling t:  10%|████████████                                                                                                          | 102/1000 [00:00<00:04, 201.36it/s]
Sampling t:  14%|████████████████                                                                                                      | 136/1000 [00:00<00:03, 239.06it/s]
Sampling t:  17%|████████████████████                                                                                                  | 170/1000 [00:00<00:03, 266.74it/s]
Sampling t:  20%|████████████████████████                                                                                              | 204/1000 [00:01<00:02, 286.51it/s]
Sampling t:  24%|████████████████████████████                                                                                          | 238/1000 [00:01<00:02, 300.84it/s]
Sampling t:  27%|████████████████████████████████                                                                                      | 272/1000 [00:01<00:02, 311.25it/s]
Sampling t:  31%|████████████████████████████████████                                                                                  | 306/1000 [00:01<00:02, 318.77it/s]
Sampling t:  34%|████████████████████████████████████████                                                                              | 340/1000 [00:01<00:02, 324.01it/s]
Sampling t:  37%|████████████████████████████████████████████▏                                                                         | 374/1000 [00:01<00:01, 327.55it/s]
Sampling t:  41%|████████████████████████████████████████████████▏                                                                     | 408/1000 [00:01<00:01, 330.00it/s]
Sampling t:  44%|████████████████████████████████████████████████████▏                                                                 | 442/1000 [00:01<00:01, 332.65it/s]
Sampling t:  48%|████████████████████████████████████████████████████████▏                                                             | 476/1000 [00:01<00:01, 333.89it/s]
Sampling t:  51%|████████████████████████████████████████████████████████████▏                                                         | 510/1000 [00:01<00:01, 334.48it/s]
Sampling t:  54%|████████████████████████████████████████████████████████████████▏                                                     | 544/1000 [00:02<00:01, 334.94it/s]
Sampling t:  58%|████████████████████████████████████████████████████████████████████▏                                                 | 578/1000 [00:02<00:01, 335.03it/s]
Sampling t:  61%|████████████████████████████████████████████████████████████████████████▏                                             | 612/1000 [00:02<00:01, 333.55it/s]
Sampling t:  65%|████████████████████████████████████████████████████████████████████████████▏                                         | 646/1000 [00:02<00:01, 334.30it/s]
Sampling t:  68%|████████████████████████████████████████████████████████████████████████████████▏                                     | 680/1000 [00:02<00:00, 334.68it/s]
Sampling t:  71%|████████████████████████████████████████████████████████████████████████████████████▎                                 | 714/1000 [00:02<00:00, 334.98it/s]
Sampling t:  75%|████████████████████████████████████████████████████████████████████████████████████████▎                             | 748/1000 [00:02<00:00, 335.25it/s]
Sampling t:  78%|████████████████████████████████████████████████████████████████████████████████████████████▎                         | 782/1000 [00:02<00:00, 335.48it/s]
Sampling t:  82%|████████████████████████████████████████████████████████████████████████████████████████████████▎                     | 816/1000 [00:02<00:00, 335.61it/s]
Sampling t:  85%|████████████████████████████████████████████████████████████████████████████████████████████████████▎                 | 850/1000 [00:02<00:00, 335.71it/s]
Sampling t:  88%|████████████████████████████████████████████████████████████████████████████████████████████████████████▎             | 884/1000 [00:03<00:00, 335.78it/s]
Sampling t:  92%|████████████████████████████████████████████████████████████████████████████████████████████████████████████▎         | 918/1000 [00:03<00:00, 335.72it/s]
Sampling t:  95%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎     | 952/1000 [00:03<00:00, 335.64it/s]
Sampling t:  99%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎ | 986/1000 [00:03<00:00, 335.54it/s]
Sampling t: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:03<00:00, 293.36it/s]
saving perturb data array to log/2025-04-02-14-24/reverse_sample_108x20.npz
visulize the perturbed data and real data
The mmd score is:0.014780950938991788
/home/ffeltus/.conda/envs/GEMDiff/lib/python3.11/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
filter the perturbed gene -- 1 std
The real data mean [ 0.44266433  0.46999094  0.37065968  0.3497042  -0.0493876  -0.66582197
  0.50274587  0.7668456   0.48156074  0.30529064  0.4954468   0.34648234
  0.7990788   0.6825261   0.4341889   0.3764099   0.30780873  0.4563495
  0.6531624   0.5506654 ]-- script_util
The real data std [0.06524871 0.06676315 0.07775255 0.07166351 0.4756117  0.40093306
 0.06165875 0.04257644 0.07795445 0.23032239 0.22685818 0.06350208
 0.07346484 0.05895608 0.06329691 0.05610733 0.09744514 0.05676592
 0.0647126  0.05566232]-- script_util
The perturb data mean [ 0.45425132  0.48374942  0.43468422  0.38148817 -0.56696486 -0.55387276
  0.40122715  0.8001777   0.404018    0.44526327  0.66704106  0.4316616
  0.7146631   0.69587016  0.4816805   0.3624624   0.39476258  0.5092408
  0.71500194  0.47730255]-- script_util
The perturb data std [0.06902174 0.06504394 0.06409916 0.06573853 0.3434523  0.30143946
 0.06893926 0.05020548 0.06920076 0.16775039 0.15756726 0.07257611
 0.07038266 0.05927245 0.05955416 0.04511983 0.06772096 0.05148546
 0.06166882 0.05568291]-- script_util
The differences between real and perturb data [0.05361823 0.04067468 0.07080145 0.05196977 0.5175771  0.16832158
 0.1020387  0.04196308 0.0844323  0.15038116 0.18406875 0.08673385
 0.08688916 0.04121995 0.05417978 0.0381176  0.09383969 0.05439159
 0.06835628 0.07490841] -- script_util
The standard deviation between real and perturb data data 0.10352832823991776 -- script_util
The mean between real and perturb data data 0.10322415828704834 -- script_util
The real data [-0.0493876]-- script_util
The perturb data [-0.56696486]-- script_util
The perturbation percentages between real and perturb data data [-10.4799]-- script_util
The indentified genes are: Index(['MUC16'], dtype='object') -- 1 standard deviation of the perturbation among all 20 gene
pertubing complete
[rank0]:[W402 14:24:36.814764033 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())
```
