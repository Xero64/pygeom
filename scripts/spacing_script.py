#%%
# Import Dependencies
from pygeom.tools.spacing import full_cosine_spacing, semi_cosine_spacing

#%%
# Create a semi-cosine spacing array
scs = semi_cosine_spacing(10)
print(f'scs = {scs}')

#%%
# Create a full-cosine spacing array
fcs = full_cosine_spacing(10)
print(f'fcs = {fcs}')
