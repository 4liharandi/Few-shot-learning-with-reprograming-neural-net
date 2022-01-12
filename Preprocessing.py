

model = resnet18(pretrained=True)
image_size = 224
pimg_size = (224,224)
img_size = (105,105)
mask_size = (224,224)
num_channels = 3
l_pad = int((pimg_size[0]-img_size[0]+1)/2)
r_pad = int((pimg_size[0]-img_size[0])/2)

train_set = Omniglot(
    root="./data",
    background=True,
    transform=transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Pad(padding=(l_pad, l_pad, r_pad, r_pad)),
            transforms.ToTensor(),
        ]
    ),
    download=True,
)
test_set = Omniglot(
    root="./data",
    background=False,
    #train = False,
    transform=transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            #transforms.Resize([(image_size), (image_size)]),
            transforms.Pad(padding=(l_pad, l_pad, r_pad, r_pad)),
            transforms.ToTensor(),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

        ]
    ),
    download=True,
        
)

N_WAY = 5 # Number of classes in a task
N_SHOT = 5 # Number of images per class in the support set
N_QUERY = 10 # Number of images per class in the query set
N_EVALUATION_TASKS = 100

test_set.labels = [
    instance[1] for instance in test_set._flat_character_images
]

test_sampler = TaskSampler(
    test_set, 
    n_way=N_WAY, 
    n_shot=N_SHOT, 
    n_query=N_QUERY, 
    n_tasks=N_EVALUATION_TASKS,
)

test_loader = DataLoader(
    test_set,
    batch_sampler=test_sampler,
    num_workers=2,
    pin_memory=True,
    collate_fn=test_sampler.episodic_collate_fn,
)


def gaussian_kernel(size=5, sigma=0.7):

    x = np.linspace(-1,1,size)
    x /= np.sqrt(2)*sigma
    x2 = x**2
    kernel = np.exp(- x2[:, None] - x2[None, :])
    kernel = kernel / kernel.sum()
    return np.expand_dims(kernel, 0)


device = torch.device('cuda')
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
def blur(weight, sigma, size=5, padding=2):
    blur_kernel = torch.Tensor(gaussian_kernel(size=size, sigma=sigma)).to(device).repeat((1,1,1,1))
#     print(blur_kernel.requires_grad)
    weight[0] = F.conv2d(weight[0].unsqueeze(0).unsqueeze(0), blur_kernel, padding=padding)
    weight[1] = F.conv2d(weight[1].unsqueeze(0).unsqueeze(0), blur_kernel, padding=padding)
    weight[2] = F.conv2d(weight[2].unsqueeze(0).unsqueeze(0), blur_kernel, padding=padding)