from .utility.torch_compat import install as _install_torch_compat

# No-op on GPUs the installed torch was built for; see utility/torch_compat.py.
_install_torch_compat()

# import macarons.networks as networks
# import macarons.test as test
# import macarons.trainers as trainers
# import macarons.utility as utility
