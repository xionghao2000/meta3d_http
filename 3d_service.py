class meta3d_model():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_diffusion = None
        self.unsampler_diffusion = None
        self.base_model_path = 'base_model.pt'
        self.unsample_model_path = 'upsample_model.pt'
        self.base_name = 'base40M-textvec'
        self.base_model_loaded = None
        self.unsampler_diffusion_loaded = None
    
    def create_model(self):
        """
        create the model
        """

        base_model = model_from_config(MODEL_CONFIGS[self.base_name], self.device)
        base_model.eval()

        upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], self.device)
        upsampler_model.eval()

        base_model.load_state_dict(load_checkpoint(self.base_name, self.device))

        upsampler_model.load_state_dict(load_checkpoint('upsample', self.device))

        torch.save(base_model.state_dict(), self.base_model_path)
        torch.save(upsampler_model.state_dict(), self.unsample_model_path)
    
    # create a function to load the model
    def load_model(self):
        self.base_model_loaded = torch.load(self.base_model_path)
        self.unsampler_diffusion_loaded = torch.load(self.unsample_model_path)
    
    # create a function to get the difffusion
    def get_diffusion(self):
        self.model_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[self.base_name])
        self.unsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

    # create a function to create the point cloud sampler
    def create_point_cloud_sampler(self):
        sampler = PointCloudSampler(
            device=self.device,
            models=[self.base_model_loaded, self.unsampler_diffusion_loaded],
            diffusions=[self.model_diffusion, self.unsampler_diffusion],
            num_points=[1024, 4096 - 1024],
            aux_channels=['R', 'G', 'B'],
            guidance_scale=[3.0, 0.0],
            model_kwargs_key_filter=('texts', ''), # Do not condition the upsampler at all
        )
        return sampler
    
    # create a function to do the generation
    def generate(self, prompt: str = 'a red motorcycle'):
        # Produce a sample from the model.
        samples = None
        for x in tqdm(self.create_point_cloud_sampler().sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]))):
            samples = x
        pc = self.create_point_cloud_sampler().output_to_point_clouds(samples)[0]
        return pc   
    
    def save_model2ply(self, filename: str):
        '''
        save the model to ply file
        '''
        with open(filename, 'wb') as f:
            self.generate().weite_ply(f)