from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from fastvideo.v1.pipelines.wan.wan_latent_pipeline import WanLatentPipeline


def main():
    print("Starting data preprocessor")
    pipeline = WanLatentPipeline.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

    train_dataset = getdataset(args)
    sampler = DistributedSampler(train_dataset,
                                 rank=local_rank,
                                 num_replicas=world_size,
                                 shuffle=True)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    for batch in train_dataloader:
        pipeline(batch)


if __name__ == "__main__":
    main()
