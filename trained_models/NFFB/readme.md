Most of the NFFB networks training were made with 
some minor changes to parameters so the pytorch loader
may not be able to load them 

For cases 114,110,122 Consider defining Stymodulation and Style attention block (despite the fact that it is not used...) 

Also consider changing the following parameters in the CustomEmbeddingNetwork.py
HashGrid Parameters

base_sigma = 8.0
exp_sigma = 1.26
per_level_scale = 2.0

on nffb3d.py
sinw0 = 30
sinw0_high = 30

If all those adjustments are made the networks can be loaded properly

Despite that fact I provide the evaluation results of the networks on the eval folder.These evaluations were made with the above parameters which were minor adjusted later(thus you may nod be able to load properly the models).