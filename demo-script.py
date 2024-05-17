# %% [markdown]
# # Inference using the MolFormer Model

# %% [markdown]
# In this notebook, we show how to perform inference using GT4SD and finetuned variants of the MolFormer model. The current existing models have been trained based on the datasets provided by the [official MolFormer repository]https://github.com/IBM/molformer).

# %% [markdown]
# ### Models for regression
# 
# This method can be used for any regression task.
import time
start = time.time()
# %%
# from gt4sd.properties.molecules import MOLECULE_PROPERTY_PREDICTOR_FACTORY
from molformer_inference.common.properties.molecules import MOLECULE_PROPERTY_PREDICTOR_FACTORY

property_class, parameters_class = MOLECULE_PROPERTY_PREDICTOR_FACTORY["molformer_regression"]
var_a = parameters_class(algorithm_version="molformer_alpha_public_test")
model = property_class(var_a)

model(input=["OC12COC3=NCC1C23"])

# %% [markdown]
# ### Models for classification
# 
# This method can be used for any binary classification task.

# %%
# from gt4sd.properties.molecules import MOLECULE_PROPERTY_PREDICTOR_FACTORY
from molformer_inference.common.properties.molecules import MOLECULE_PROPERTY_PREDICTOR_FACTORY

property_class, parameters_class = MOLECULE_PROPERTY_PREDICTOR_FACTORY["molformer_classification"]
model = property_class(parameters_class(algorithm_version="molformer_bace_public_test"))

model(input=["OC12COC3=NCC1C23"])

# %% [markdown]
# ### Molformer for multiclass classification
# 
# This method can be used for any multiclass classification task.

# %%
property_class, parameters_class = MOLECULE_PROPERTY_PREDICTOR_FACTORY["molformer_multitask_classification"]
model = property_class(parameters_class(algorithm_version="molformer_clintox_test"))

print(model(["Ic1cc(ccc1)C[NH2+]C[C@@H](O)[C@@H](NC(=O)c1cc(cc(c1)C)C(=O)N(CCC)CCC)Cc1cc(F)cc(F)c1"]))

print(f"time taken: {time.time()-start}")
