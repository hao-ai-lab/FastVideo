```
python scripts/download_hf.py --repo_id=FastVideo/Distill-30K-Data-Src --local_dir=data/Distill-30K-Data-Src --repo_type=dataset
cd data/Distill-30K-Data-Src
cat archive_name.tar.gz.* > Distill-30K-Data-Src.tar.gz
rm archive_name.tar.gz.*
tar --use-compress-program="pigz --processes 64" -xvf Distill-30K-Data-Src.tar.gz
mv ephemeral/hao.zhang/codefolder/FastVideo-OSP/data/Distill-30K-Data-Src/* .
rm -r ephemeral
rm Distill-30K-Data-Src.tar.gz
cd ../..
# Now the src data is stored in folder data/Distill-30K-Data-Src, adjust the path in `data/Distill-30K-Src/merge.txt` correspondingly
# Adjust the `SHARD_IDX=0` in `./scripts/preprocess_hunyuan_data.sh` for propressing data on multiple nodes


bash ./scripts/preprocess_hunyuan_data.sh

# The preprocessed data will be stored in `data/HD-Hunyuan-30K-Distill-Data_Shard*`

```