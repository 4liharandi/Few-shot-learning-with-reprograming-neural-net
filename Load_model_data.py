from torchvision import transforms
from skimage.transform import resize
from torch.utils.data import DataLoader
from torchvision.datasets import Omniglot, CIFAR10
from torchvision.models import resnet18
from easyfsl.data_tools import TaskSampler
from torchvision.transforms import Normalize
import numpy as np


#load pretreined model and data
#for improve performance you can use other pretrained models
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



#define gaussian blurring for regularize weight and gradient 

def gaussian_kernel(size=5, sigma=0.7):

    x = np.linspace(-1,1,size)
    x /= np.sqrt(2)*sigma
    x2 = x**2
    kernel = np.exp(- x2[:, None] - x2[None, :])
    kernel = kernel / kernel.sum()
    return np.expand_dims(kernel, 0)


def blur(weight, sigma, size=5, padding=2):
    blur_kernel = torch.Tensor(gaussian_kernel(size=size, sigma=sigma)).to(device).repeat((1,1,1,1))

    weight[0] = F.conv2d(weight[0].unsqueeze(0).unsqueeze(0), blur_kernel, padding=padding)
    weight[1] = F.conv2d(weight[1].unsqueeze(0).unsqueeze(0), blur_kernel, padding=padding)
    weight[2] = F.conv2d(weight[2].unsqueeze(0).unsqueeze(0), blur_kernel, padding=padding)

    
    
def flags():

    parser = argparse.ArgumentParser()
     
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        help='number of epochs to train for')
    
    parser.add_argument(
        '--lr',
        type=float,
        default= 0.01,
        help='learning rate')
    
    parser.add_argument(
        '--num_ensemble',
        type=int,
        default= 10,
        help='number of ensemble')
   
    parser.add_argument(
        '--num_way',
        type=int,
        default= 10,
        help='number of class')
    
    parser.add_argument(
        '--num_shot',
        type=int,
        default= 10,
        help='number of train')
            
    parser.add_argument(
        '--num_query',
        type=int,
        default= 10,
        help='number of test')
    
     parser.add_argument(
        '--num_tasks',
        type=int,
        default= 100,
        help='number of episodes')
    
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed
