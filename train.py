import argparse
import numpy as np
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import os.path as osp
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
import random
from diffusion.datasets import get_target_dataset
from diffusion.models import get_sd_model, get_scheduler_config
from diffusion.utils import LOG_DIR, get_formatstr
import torchvision.transforms as torch_transforms
from torchvision.transforms.functional import InterpolationMode

from coop import PromptLearner, load_clip_to_cpu, TextEncoder
from datasets import build_dataset
from diffusion.utils import DATASET_ROOT
from datasets.utils import build_data_loader
import clip
from utils.air_get_tree_target_2 import *

device = torch.device("cuda:2") if torch.cuda.is_available() else "cpu"

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose([
        torch_transforms.Resize(size, interpolation=interpolation),
        torch_transforms.CenterCrop(size),
        _convert_image_to_rgb,
        torch_transforms.ToTensor(),
        torch_transforms.Normalize([0.5], [0.5])
    ])
    return transform


def center_crop_resize(img, interpolation=InterpolationMode.BILINEAR):
    transform = get_transform(interpolation=interpolation)
    return transform(img)


def eval_prob_adaptive(unet, latent, text_embeds, scheduler, args, latent_size=64, all_noise=None):
    scheduler_config = get_scheduler_config(args)
    T = scheduler_config['num_train_timesteps']
    max_n_samples = max(args.n_samples)
    bsz = latent.size(0)

    if all_noise is None:
        all_noise = torch.randn((max_n_samples * args.n_trials, 4, latent_size, latent_size), device=latent.device)
    if args.dtype == 'float16':
        all_noise = all_noise.half()
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.half()

    data = dict()
    t_evaluated = set()
    remaining_prmpt_idxs = list(range(len(text_embeds)))
    start = T // max_n_samples // 2
    t_to_eval = list(range(start, T, T // max_n_samples))[:max_n_samples] 

    for n_samples, n_to_keep in zip(args.n_samples, args.to_keep):
        ts = []
        noise_idxs = []
        text_embed_idxs = []
        curr_t_to_eval = t_to_eval[len(t_to_eval) // n_samples // 2::len(t_to_eval) // n_samples][:n_samples]
        curr_t_to_eval = [t for t in curr_t_to_eval if t not in t_evaluated]

        # Randomly sample t values from curr_t_to_eval
        num_samples = 1
        sampled_t = random.sample(curr_t_to_eval, num_samples)
        curr_t_to_eval = [t for t in sampled_t if t not in t_evaluated]

        # sequential select
        # if start_idx < len(t_to_eval):
        #     end_idx = start_idx + min(5, len(t_to_eval) - start_idx)
        #     curr_t_to_eval = t_to_eval[start_idx:end_idx]
        #     start_idx = end_idx % len(t_to_eval)

        for prompt_i in remaining_prmpt_idxs:
            for t_idx, t in enumerate(curr_t_to_eval, start=len(t_evaluated)):
                ts.extend([t] * args.n_trials)
                noise_idxs.extend(list(range(args.n_trials * t_idx, args.n_trials * (t_idx + 1))))
                text_embed_idxs.extend([prompt_i] * args.n_trials)
        t_evaluated.update(curr_t_to_eval)
        pred_errors = eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs,
                                 text_embeds, text_embed_idxs, args.batch_size, args.dtype, args.loss)                              
        # match up computed errors to the data
        for prompt_i in remaining_prmpt_idxs:
            mask = torch.tensor(text_embed_idxs) == prompt_i
            prompt_ts = torch.tensor(ts)[mask]
            # prompt_pred_errors = pred_errors[mask]
            prompt_pred_errors = torch.masked_select(pred_errors, mask.unsqueeze(0)).reshape(bsz,-1)
            if prompt_i not in data:
                data[prompt_i] = dict(t=prompt_ts, pred_errors=prompt_pred_errors)
            else:
                data[prompt_i]['t'] = torch.cat([data[prompt_i]['t'], prompt_ts])
                data[prompt_i]['pred_errors'] = torch.cat([data[prompt_i]['pred_errors'], prompt_pred_errors])

        # compute the next remaining idxs
        errors = torch.stack([-data[prompt_i]['pred_errors'].mean(dim=1) for prompt_i in remaining_prmpt_idxs], dim=0)
        # output = F.softmax(errors/0.001, dim=0)
        errors = errors/0.0005
        best_idxs = torch.topk(errors, k=n_to_keep, dim=0).indices.tolist()
        best_idx = [idx[0] for idx in best_idxs]  
        remaining_prmpt_idxs = [remaining_prmpt_idxs[i] for i in best_idx]

    # organize the output
    assert len(remaining_prmpt_idxs) == 1
    pred_idx = remaining_prmpt_idxs[0]
    num_classes = errors.size(0)
    error = errors.view(bsz,num_classes)

    # return pred_idx, data, error
    return best_idxs, data, error


def eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs,
               text_embeds, text_embed_idxs, batch_size=32, dtype='float32', loss='l2'):
    assert len(ts) == len(noise_idxs) == len(text_embed_idxs)
    num_samples = latent.size(0)  
    pred_errors = torch.zeros((num_samples, len(ts)), device='cpu')
    idx = 0
    # with torch.inference_mode():
    for _ in tqdm.trange(len(ts) // batch_size + int(len(ts) % batch_size != 0), leave=False):
        batch_ts = torch.tensor(ts[idx: idx + batch_size]) 
        noise = all_noise[noise_idxs[idx: idx + batch_size]]
        latent = latent.to(device)
        noised_latent = latent * ((scheduler.alphas_cumprod[batch_ts]) ** 0.5).view(-1, 1, 1, 1, 1).to(device) + \
                        noise.unsqueeze(1) * ((1 - scheduler.alphas_cumprod[batch_ts]) ** 0.5).view(-1, 1, 1, 1, 1).to(device)
        noised_latent_each = torch.split(noised_latent, 1, dim=1)
        bsz, _, _, _, _ = noised_latent.shape
        for i, noised_latent in enumerate(noised_latent_each):
            noised_latent = noised_latent.view(bsz, 4, 64, 64)
            t_input = batch_ts.to(device).half() if dtype == 'float16' else batch_ts.to(device)
            text_input = text_embeds[text_embed_idxs[idx: idx + batch_size]]
            noise_pred = unet(noised_latent, t_input, encoder_hidden_states=text_input).sample
            if loss == 'l2':
                error = F.mse_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
            elif loss == 'l1':
                error = F.l1_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
            elif loss == 'huber':
                error = F.huber_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
            else:
                raise NotImplementedError
            # Apply coefficient exp(-7t) to error
            # exp_coeff = torch.exp(-7 * batch_ts.to(device))
            # error = exp_coeff * error
            # pred_errors[idx: idx + len(batch_ts)] = error.detach().cpu()
            pred_errors[i][idx: idx + len(batch_ts)] = error #.detach().cpu()
        idx += len(batch_ts)
    return pred_errors

def main():
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument('--dataset', type=str, default='oxford_pets',
                        choices=['oxford_pets', 'flowers', 'stl10', 'mnist', 'cifar10', 'food', 'caltech101', 'imagenet',
                                 'objectnet', 'aircraft', 'stanford_cars','cub'], help='Dataset to use')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'], help='Name of split')
    parser.add_argument('--shots', type=int, default=16, choices=(1,2,4,8,16), help='number of shots')

    # run args
    parser.add_argument('--version', type=str, default='1-5', help='Stable Diffusion model version')
    parser.add_argument('--img_size', type=int, default=512, choices=(256, 512), help='Number of trials per timestep')
    parser.add_argument('--batch_size', '-b', type=int, default=37)
    parser.add_argument('--n_trials', type=int, default=1, help='Number of trials per timestep')
    parser.add_argument('--prompt_path', type=str, required=True, help='Path to csv file with prompts to use')
    parser.add_argument('--noise_path', type=str, default=None, help='Path to shared noise to use')
    parser.add_argument('--subset_path', type=str, default=None, help='Path to subset of images to evaluate')
    parser.add_argument('--dtype', type=str, default='float16', choices=('float16', 'float32'),
                        help='Model data type to use')
    parser.add_argument('--interpolation', type=str, default='bicubic', help='Resize interpolation type')
    parser.add_argument('--extra', type=str, default=None, help='To append to the run folder name')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of workers to split the dataset across')
    parser.add_argument('--worker_idx', type=int, default=0, help='Index of worker to use')
    parser.add_argument('--load_stats', action='store_true', help='Load saved stats to compute acc')
    parser.add_argument('--loss', type=str, default='l2', choices=('l1', 'l2', 'huber'), help='Type of loss to use')

    # args for adaptively choosing which classes to continue trying
    parser.add_argument('--to_keep', nargs='+', type=int, required=True)
    parser.add_argument('--n_samples', nargs='+', type=int, required=True)

    args = parser.parse_args()
    assert len(args.to_keep) == len(args.n_samples)

    # make run output folder
    name = f"v{args.version}_{args.n_trials}trials_"
    name += '_'.join(map(str, args.to_keep)) + 'keep_'
    name += '_'.join(map(str, args.n_samples)) + 'samples'
    name += f"_{args.shots}shots"
    if args.interpolation != 'bicubic':
        name += f'_{args.interpolation}'
    if args.loss == 'l1':
        name += '_l1'
    elif args.loss == 'huber':
        name += '_huber'
    if args.img_size != 512:
        name += f'_{args.img_size}'
    if args.extra is not None:
        run_folder = osp.join(LOG_DIR, args.dataset + '_' + args.extra, name)
    else:
        run_folder = osp.join(LOG_DIR, args.dataset, name + '_order_csc') 
    os.makedirs(run_folder, exist_ok=True)
    print(f'Run folder: {run_folder}')

    # load pretrained models
    vae, tokenizer, _, unet, scheduler = get_sd_model(args)
    vae = vae.to(device)
    clip_model = load_clip_to_cpu("ViT-L/14")  
    # clip_model.float()
    text_encoder = TextEncoder(clip_model).to(device)
    unet = unet.to(device)
    
    for param in unet.parameters():
        param.requires_grad = False
    for param in vae.parameters():
        param.requires_grad = False
    for param in text_encoder.parameters():
        param.requires_grad = False         
    torch.backends.cudnn.benchmark = True

    # set up dataset
    interpolation = INTERPOLATIONS[args.interpolation]
    transform = get_transform(interpolation, args.img_size)
    latent_size = args.img_size // 8

    target_dataset = build_dataset(args.dataset, DATASET_ROOT, args.shots)  
    train_loader = build_data_loader(data_source=target_dataset.train_x, batch_size=1, is_train=True, tfm=transform, shuffle=True)

    # load noise
    if args.noise_path is not None:
        assert not args.zero_noise
        all_noise = torch.load(args.noise_path).to(device)
        print('Loaded noise from', args.noise_path)
    else:
        all_noise = None

    # set up prompts
    df = pd.read_csv('prompts/aircraft_name.csv')
    # level 3
    df_species_sorted = df.sort_values(by='variant label')
    classnames = df_species_sorted['Variants'].tolist()
    # # level 2
    # df_family_sorted = df.sort_values(by='family label')
    # df_family_sorted = df_family_sorted.drop_duplicates(subset='family label')
    # classnames = df_family_sorted['Families'].tolist()
    # # level 1
    # df_order_sorted = df.sort_values(by='Manufactures label')
    # df_order_sorted = df_order_sorted.drop_duplicates(subset='Manufactures label')
    # classnames = df_order_sorted['Manufactures'].tolist()
    # classnames = ['pickup truck','convertible','coupe','hatchback','minivan','sedan','SUV','van','wagon']
    prompt_learner = PromptLearner(classnames, clip_model)

    formatstr = get_formatstr(len(train_loader) - 1)
    correct = 0
    total = 0
    total_correct = 0
    total_samples = 0   
    epoch = 100
    pbar = tqdm.tqdm(train_loader)

    # set up optimizer
    optimizer = torch.optim.SGD(prompt_learner.parameters(), lr=1e-3)  
    prompt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch * len(train_loader))

    for train_idx in range(epoch):
        train_loss = 0 
        idx = 0
        for i, (image, label) in enumerate(pbar):
            # # get order and family lables
            # label, _ = get_order_family_target(label)    
            idx = i
            if total_samples > 0:
                pbar.set_description(f'Acc: {100 * total_correct / total_samples:.2f}%')
            fname = osp.join(run_folder, '{}.pth'.format(train_idx))

            # get text embeddings
            prompt_learner.train()
            prompts = prompt_learner()
            text_input = prompt_learner.tokenized_prompts                    
            embeddings = []
            text_embeddings = text_encoder(prompts.to(device), text_input.to(device))
            assert len(text_embeddings) == len(classnames)

            with torch.no_grad():
                img_input = image.to(device)
                label = label.to(device)
                if args.dtype == 'float16':
                    img_input = img_input.half()
                x0 = vae.encode(img_input).latent_dist.mean
                x0 *= 0.18215
            
            pred_idx, pred_errors, error = eval_prob_adaptive(unet, x0, text_embeddings, scheduler, args, latent_size, all_noise)
            error = error.to(device)
            pred = torch.tensor(pred_idx).to(device)
            
            num_classes = len(classnames)
            label_one_hot = F.one_hot(label, num_classes)
            loss = F.cross_entropy(error, label_one_hot.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prompt_scheduler.step()

            batch_correct = (pred == label).sum().item()
            batch_samples = label.size(0)
            total_correct += batch_correct
            total_samples += batch_samples   
            train_loss += loss         
        accuracy = total_correct / total_samples
        print("accuracy: {:.2%}, loss: {:.5f}".format(accuracy, train_loss/(idx+1)))
        with open('results_test.txt', 'a') as file:
            file.write('epoch %d, accuracy = %.2f%%, loss = %.5f\n' % (
            train_idx, accuracy*100, train_loss/(idx+1)))

        if train_idx % 5 == 0:
            torch.save({
                'epoch': train_idx,
                'model_state_dict': prompt_learner.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict' : prompt_scheduler.state_dict(),
            }, fname)


if __name__ == '__main__':
    main()
