from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread
import matplotlib.pyplot as plt
import pickle
import torch
import numpy as np
from PIL import Image
import os
import lpips
import torch.nn as nn
import torchvision.transforms as transforms
from pytorch_msssim import ms_ssim
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms as transforms
from stylegan2_models.e4e.model_utils import load_e4e_standalone
from stylegan2_models.image_aligner.face_alignment import image_align
from stylegan2_models.image_aligner.landmarks_detector import LandmarksDetector
from stylegan2_models.arcface_model import get_model

def align_images(raw_images_dir, aligned_images_dir):

    landmarks_model_path = "/content/stylegan2-ada-pytorch/pretrained_models/shape_predictor_68_face_landmarks.dat"
    landmarks_detector = LandmarksDetector(landmarks_model_path)
    for celeb in os.listdir(raw_images_dir):
      celeb_folder = os.path.join(raw_images_dir, celeb)
      save_path = os.path.join(aligned_images_dir, celeb)
      os.makedirs(save_path, exist_ok=True)
      for i, img_name in enumerate(os.listdir(celeb_folder)):
          if img_name == ".ipynb_checkpoints":
            continue
          raw_img_path = os.path.join(celeb_folder, img_name)
          for face_landmarks in landmarks_detector.get_landmarks(raw_img_path):
              aligned_face_path = os.path.join(save_path, f"{i}.jpg")
              image_align(raw_img_path, aligned_face_path, face_landmarks)
              break


def showImages(images_paths):
    fig = figure(figsize=(20, 20))
    if ".ipynb_checkpoints" in images_paths:
      images_paths.remove(".ipynb_checkpoints")
    image_folder_len = len(images_paths)
    for i, image_path in enumerate(images_paths, start=1):
        fig.add_subplot(1,image_folder_len,i)
        image = imread(image_path)
        imshow(image)
        axis('off')

def broadcast_w_sg(w_batch, cast_n=18):
    input_ws = []
    for w in w_batch:
        w_broadcast = torch.broadcast_to(w, (cast_n, 512))
        input_ws.append(w_broadcast)
    return torch.stack(input_ws)


seed = 2345645
noise_mode = 'const' # шум
label = 0 # для разных моделей
device = "cuda:0"
model_path = "/content/stylegan2-ada-pytorch/pretrained_models/ffhq.pkl"

with open(model_path, 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()

# https://pypi.org/project/pytorch-msssim/
class Rec_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_criterion = torch.nn.L1Loss(reduction='mean')

    def forward(self, target, synth):
        target = torch.add(target, 1.0)
        target = torch.mul(target, 127.5)
        target = target / 255

        synth = torch.add(synth, 1.0)
        synth = torch.mul(synth, 127.5)
        synth = synth / 255

        loss = torch.mean(1 - ms_ssim(synth, target, data_range=1, size_average=True))
        return loss


# https://pypi.org/project/lpips/
class Lpips_loss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.lpips_loss = lpips.LPIPS(net='vgg')
        self.lpips_loss.to(device)
        self.lpips_loss.eval()

    def forward(self, target, synth):
        return torch.mean(self.lpips_loss(target, synth))


# https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/projector.py#L104
class Reg_loss(nn.Module):
    def __init__(self, noise_bufs):
        super().__init__()
        self.noise_bufs = noise_bufs

    def forward(self,):
        reg_loss = 0.0
        for v in self.noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        return reg_loss

class Arcface_Loss(nn.Module):
    def __init__(self, weights_path, device):
        super().__init__()

        self.arcnet = get_model("r50", fp16=False)
        self.arcnet.load_state_dict(torch.load(weights_path))
        self.arcnet.eval()
        self.arcnet.to(device)

        self.cosin_loss = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, source, synth):

        source = F.interpolate(source,size=(112,112), mode='bicubic')
        synth = F.interpolate(synth,size=(112,112), mode='bicubic')

        emb1 = self.arcnet(source)
        emb2 = self.arcnet(synth)
        loss = (1 - self.cosin_loss(emb1, emb2))[0]
        return loss

# image = (image - mean) / std
def image2tensor_norm(image):
    transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    tensor = transform(image)
    return tensor

def create_loss_projections(images_paths, save_dir, encoder_optim=False, init_latent_vectors=None, **optim_params):

  #  loss weights
  regularize_noise_weight = optim_params.get("regularize_noise_weight", 5e5)
  rec_weight = optim_params.get("rec_weight", 0.5)
  lpips_weight = optim_params.get("lpips_weoght", 1)

  # Параметры для оптимизации
  num_steps = optim_params.get("nums_sterps", 150)
  seed =  optim_params.get("seed", 13)
  initial_learning_rate = optim_params.get("initial_learning_rate", 0.05)
  w_avg_samples = optim_params.get("w_avg_samples", 10000)

  for i, img_path in enumerate(images_paths):

    print(f"Start processing image № {i}")

    # загружаем изображение
    target_pil = Image.open(img_path).convert('RGB')
    target_tensor = image2tensor_norm(target_pil).to(device).unsqueeze(0)


    # инициализируем функции потерь
    lpips_loss = Lpips_loss(device)
    rec_loss = Rec_loss()

    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }
    reg_loss = Reg_loss(noise_bufs)

    if encoder_optim:
      if init_latent_vectors is None:
        raise ValueError("Need initial latent vectors")
      w_opt = nn.Parameter(torch.clone(init_latent_vectors[i]), requires_grad=True)

    else:

      # Получаем средний вектор латентного пространства
      z_samples = torch.from_numpy(np.random.RandomState(seed).randn(w_avg_samples, G.z_dim)).to(device)
      w_samples = G.mapping(z_samples, None)
      w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)
      w_avg = np.mean(w_samples, axis=0, keepdims=True)   # [1, 1, C]
      w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

      # w or w_plus
      w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=False) # pylint: disable=not-callable
      w_opt = broadcast_w_sg(w_opt).requires_grad_(True)

    optimizer = torch.optim.Adam([w_opt], lr=initial_learning_rate)
    generated_tensors = []
    for step in tqdm(range(num_steps)):
      synth_tensor = G.synthesis(broadcast_w_sg(w_opt), noise_mode='const')

      lpips_value = lpips_loss(synth_tensor, target_tensor)
      rec_value = rec_loss(synth_tensor, target_tensor)
      reg_value = reg_loss()

      loss = lpips_value*lpips_weight + rec_value*rec_weight + reg_value*regularize_noise_weight

      optimizer.step()
      optimizer.zero_grad(set_to_none=True)
      loss.backward()

      generated_tensors.append(synth_tensor)
    generated_tensor = G.synthesis(broadcast_w_sg(w_opt), noise_mode='const', force_fp32=True)
    generated_tensor = (generated_tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = Image.fromarray(generated_tensor[0].cpu().numpy(), 'RGB')
    img.save(os.path.join(save_dir, f"{i}.jpg"))

arcface_path = "/content/stylegan2-ada-pytorch/ms1mv3_arcface_r50_fp16.pth"

def arcface_train(from_img, to_img, save_path, encoder_optim=False, init_latent_vector=None, **optim_params):

    #  loss weights
    regularize_noise_weight = optim_params.get("regularize_noise_weight", 5e5)
    rec_weight = optim_params.get("rec_weight", 0.5)
    lpips_weight = optim_params.get("lpips_weoght", 5)
    arcface_weight = optim_params.get("arcface_weight", 0.5)

    # Параметры для оптимизации
    num_steps = optim_params.get("nums_sterps", 150)
    seed =  optim_params.get("seed", 13)
    initial_learning_rate = optim_params.get("initial_learning_rate", 0.05)
    w_avg_samples = optim_params.get("w_avg_samples", 10000)

    # загружаем исходное изображение
    target_pil_from = Image.open(from_img).convert('RGB')
    target_tensor_from = image2tensor_norm(target_pil_from).to(device).unsqueeze(0)

    # загружаем натягиваемое изображение
    target_pil_to = Image.open(to_img).convert('RGB')
    target_tensor_to = image2tensor_norm(target_pil_to).to(device).unsqueeze(0)

    # инициализируем функции потерь
    lpips_loss = Lpips_loss(device)
    arcface_loss = Arcface_Loss(weights_path=arcface_path, device=device)
    rec_loss = Rec_loss()

    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }
    reg_loss = Reg_loss(noise_bufs)

    if encoder_optim:
      if init_latent_vector is None:
        raise ValueError("Need initial latent vector")
      w_opt = nn.Parameter(torch.clone(init_latent_vector), requires_grad=True)

    else:

      # Получаем средний вектор латентного пространства
      z_samples = torch.from_numpy(np.random.RandomState(seed).randn(w_avg_samples, G.z_dim)).to(device)
      w_samples = G.mapping(z_samples, None)
      w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)
      w_avg = np.mean(w_samples, axis=0, keepdims=True)   # [1, 1, C]
      w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

      # w or w_plus
      w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=False) # pylint: disable=not-callable
      w_opt = broadcast_w_sg(w_opt).requires_grad_(True)

    optimizer = torch.optim.Adam([w_opt], lr=initial_learning_rate)
    generated_tensors = []
    for step in tqdm(range(num_steps)):
      synth_tensor = G.synthesis(broadcast_w_sg(w_opt), noise_mode='const')

      lpips_value = lpips_loss(synth_tensor, target_tensor_from)
      rec_value = rec_loss(synth_tensor, target_tensor_from)
      reg_value = reg_loss()
      arcface_value = arcface_loss(target_tensor_to, synth_tensor)

      loss = lpips_value*lpips_weight + rec_value*rec_weight + reg_value*regularize_noise_weight + arcface_value*arcface_weight

      optimizer.step()
      optimizer.zero_grad(set_to_none=True)
      loss.backward()

      generated_tensors.append(synth_tensor)
    generated_tensor = G.synthesis(broadcast_w_sg(w_opt), noise_mode='const', force_fp32=True)
    generated_tensor = (generated_tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = Image.fromarray(generated_tensor[0].cpu().numpy(), 'RGB')
    img.save(save_path)



image2e4etensor = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((256, 256)),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


def create_encoder_projections(images_paths, save_dir):
    e4e_model, _ = load_e4e_standalone("/content/stylegan2-ada-pytorch/pretrained_models/e4e_ffhq_encode.pt")
    initial_latent_vectors = []
    for i, img_path in enumerate(images_paths, start=1):

      print(f"Start processing image № {i}")
      target_pil = Image.open(img_path).convert('RGB')
      target_uint8 = np.array(target_pil, dtype=np.uint8)
      e4e_tensor = image2e4etensor(target_uint8).to(device).unsqueeze(0)
      initial_latent_vector = e4e_model(e4e_tensor)
      initial_latent_vectors.append(initial_latent_vector)
      generated_tensor = G.synthesis(initial_latent_vector, noise_mode='const', force_fp32=True)
      generated_tensor = (generated_tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
      img = Image.fromarray(generated_tensor[0].detach().cpu().numpy(), 'RGB')
      img.save(os.path.join(save_dir, f"{i}.jpg"))

    return initial_latent_vectors
