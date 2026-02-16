Compare VSA different sparsity and flash attention. 
- No torch compile enabled. 

Results: 
- Does not see much difference in E2E performance (2s?). 

TODO: 
- Need to check the detailed breakdown. 
- Need to check perf on other machines (e.g. NVL)

NOTE: 
This is using the DGX Machine. 
It is reported that it is slower than other b200 machines on brev. 
