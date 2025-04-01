### ADD TO THIS TO REGISTER NEW KERNELS
sources = {
    'attn': {
        'source_files': {
            'h100': 'st_attn/st_attn_h100.cu'  # define these source files for each GPU target desired.
        }
    }
}

### WHICH KERNELS DO WE WANT TO BUILD?
# (oftentimes during development work you don't need to redefine them all.)
kernels = ['attn']

### WHICH GPU TARGET DO WE WANT TO BUILD FOR?
target = 'h100'
