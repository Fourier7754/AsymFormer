try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
    HAS_DALI = True
except ImportError:
    HAS_DALI = False
    print("NVIDIA DALI not found. Falling back to PyTorch DataLoader.")
    class types:
        FLOAT = "FLOAT"
        FLOAT16 = "FLOAT16"
        RGB = "RGB"
        INTERP_TRIANGULAR = "INTERP_TRIANGULAR"

import torch

class DALILoader:
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop=224, size=256,
                 dali_cpu=False, local_rank=0, world_size=1, training=True, dtype=types.FLOAT, prefetch_queue_depth=2):
        self.training = training
        self.batch_size = batch_size
        
        # Define pipeline
        # prefetch_queue_depth: Number of batches to prefetch. Increasing this can smooth out IO jitters.
        # Reduced to 2 to save memory with large batch sizes.
        self.pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, prefetch_queue_depth=prefetch_queue_depth)
        
        with self.pipe:
            reader = ops.readers.File(file_root=data_dir,
                                      random_shuffle=training,
                                      pad_last_batch=True,
                                      read_ahead=True,
                                      shard_id=local_rank,
                                      num_shards=world_size,
                                      name="Reader")
            images, labels = reader()
            
            # Decode and Augment
            # We use mixed backend: CPU load -> GPU decode/resize
            dali_device = "cpu" if dali_cpu else "gpu"
            decoder_device = "cpu" if dali_cpu else "mixed"
            
            # Ask for larger output to allow cropping
            if training:
                # RandomResizedCrop
                # DALI's RandomResizedCrop is efficient.
                # Standard ImageNet scale: 0.08 to 1.0. Ratio: 3/4 to 4/3.
                
                # Decode + RandomResizedCrop
                # This is faster than Decode -> Resize -> Crop
                images = ops.decoders.ImageRandomCrop(device=decoder_device, output_type=types.RGB,
                                                      random_aspect_ratio=[0.75, 4.0/3.0],
                                                      random_area=[0.08, 1.0],
                                                      num_attempts=100)(images)
                                                      
                # Resize to target crop size
                images = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop,
                                    interp_type=types.INTERP_TRIANGULAR)(images)
                
                # Flip
                # fn.random.coin_flip replacement for ops.CoinFlip
                mirror = fn.random.coin_flip(probability=0.5)
                
                cmn = ops.CropMirrorNormalize(device=dali_device,
                                                 dtype=dtype,
                                                 output_layout="CHW",
                                                 mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                                 std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
                                                 
                images = cmn(images, mirror=mirror)
            else:
                # Validation: Resize (256) -> CenterCrop (224)
                images = ops.decoders.Image(device=decoder_device, output_type=types.RGB)(images)
                images = ops.Resize(device=dali_device, resize_shorter=size,
                                    interp_type=types.INTERP_TRIANGULAR)(images)
                
                images = ops.CropMirrorNormalize(device=dali_device,
                                                 dtype=dtype,
                                                 output_layout="CHW",
                                                 crop=(crop, crop),
                                                 mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                                 std=[0.229 * 255, 0.224 * 255, 0.225 * 255])(images)
            
            # Labels (ensure Int64/Long for PyTorch)
            # labels are usually int32 from reader.
            
            self.pipe.set_outputs(images, labels)
            
        self.pipe.build()
        self.epoch_size = self.pipe.epoch_size("Reader")
        
        # last_batch_policy:
        # - DROP: Drops the last batch if incomplete. Good for training (avoids partial batch issues).
        # - FILL: Pads the last batch with the last sample. Good for validation (if we want fixed batch size).
        # - PARTIAL: Returns the partial batch.
        
        policy = LastBatchPolicy.DROP if training else LastBatchPolicy.PARTIAL
        
        self.iterator = DALIClassificationIterator(self.pipe, reader_name="Reader", auto_reset=True,
                                                   last_batch_policy=policy)

    def __len__(self):
        return (self.epoch_size + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            # self.iterator.reset() # Handled by auto_reset=True
            raise StopIteration
        
        # Parse data
        # data is list of dicts if output_map is used, or list of lists
        # DALIClassificationIterator returns list of dicts: [{'data': tensor, 'label': tensor}]
        # We need to extract them.
        
        # Note: DALIClassificationIterator returns a LIST of outputs (one per GPU).
        # Since we use device_id=0 and num_gpus=1 implicitly (local), it's list of length 1.
        
        out = data[0]
        images = out['data']
        labels = out['label']
        
        # Squeeze labels if needed (DALI returns [Batch, 1] sometimes?)
        # PyTorch expects [Batch] for CrossEntropy
        labels = labels.squeeze().long()
        
        return images, labels

def get_dali_loader(data_dir, batch_size, num_threads=4, device_id=0, training=True, local_rank=0, world_size=1, fp16=False, prefetch=2):
    if not HAS_DALI:
        return None
        
    dtype = types.FLOAT16 if fp16 else types.FLOAT
        
    return DALILoader(batch_size=batch_size, num_threads=num_threads, device_id=device_id,
                      data_dir=data_dir, training=training, local_rank=local_rank, world_size=world_size,
                      dtype=dtype, prefetch_queue_depth=prefetch)
