### ADD TO THIS TO REGISTER NEW KERNELS
sources = {
    'st_attn': {
        'source_files': {
            'h100': 'st_attn/st_attn_h100.cu'  # define these source files for each GPU target desired.
        }
    },
    'block_sparse': {
        'source_files': {
            'h100': 'st_attn/block_sparse_h100.cu'
        }
    },
    'mha': {
        'source_files': {
            'h100': 'st_attn/full_attn.cu'
        }
    }
}

### WHICH KERNELS DO WE WANT TO BUILD?
# (oftentimes during development work you don't need to redefine them all.)
kernels = ['st_attn']

### WHICH GPU TARGET DO WE WANT TO BUILD FOR?
target = 'h100'
