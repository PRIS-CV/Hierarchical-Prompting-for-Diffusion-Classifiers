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
from collections import OrderedDict
from utils.air_get_tree_target_2 import *

device = ("cuda:0") if torch.cuda.is_available() else "cpu"

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

    if all_noise is None:
        all_noise = torch.randn((max_n_samples * args.n_trials, 4, latent_size, latent_size), device=latent.device)
    if args.dtype == 'float16':
        all_noise = all_noise.half()
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.half()

    data = dict()
    t_evaluated = set()
    remaining_prmpt_idxs = list(range(len(text_embeds)))
    start = T // max_n_samples // 2
    t_to_eval = list(range(start, T, T // max_n_samples))[:max_n_samples] #采样步数n_samples

    for n_samples, n_to_keep in zip(args.n_samples, args.to_keep):
        ts = []
        noise_idxs = []
        text_embed_idxs = []
        curr_t_to_eval = t_to_eval[len(t_to_eval) // n_samples // 2::len(t_to_eval) // n_samples][:n_samples]
        curr_t_to_eval = [t for t in curr_t_to_eval if t not in t_evaluated]

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
            prompt_pred_errors = pred_errors[mask]
            # prompt_pred_errors = torch.masked_select(pred_errors, mask.unsqueeze(0)).reshape(-1)
            if prompt_i not in data:
                data[prompt_i] = dict(t=prompt_ts, pred_errors=prompt_pred_errors)
            else:
                data[prompt_i]['t'] = torch.cat([data[prompt_i]['t'], prompt_ts])
                data[prompt_i]['pred_errors'] = torch.cat([data[prompt_i]['pred_errors'], prompt_pred_errors])

        # compute the next remaining idxs
        errors = [-data[prompt_i]['pred_errors'].mean() for prompt_i in remaining_prmpt_idxs]
        best_idxs = torch.topk(torch.tensor(errors), k=n_to_keep, dim=0).indices.tolist()
        remaining_prmpt_idxs = [remaining_prmpt_idxs[i] for i in best_idxs]

    # organize the output
    assert len(remaining_prmpt_idxs) == 1
    pred_idx = remaining_prmpt_idxs[0]
    error = torch.tensor(errors)

    return pred_idx, data, error



def eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs,
               text_embeds, text_embed_idxs, batch_size=32, dtype='float32', loss='l2'):
    assert len(ts) == len(noise_idxs) == len(text_embed_idxs)
    pred_errors = torch.zeros(len(ts), device='cpu')
    idx = 0
    with torch.inference_mode():
        for _ in tqdm.trange(len(ts) // batch_size + int(len(ts) % batch_size != 0), leave=False):
            batch_ts = torch.tensor(ts[idx: idx + batch_size])
            noise = all_noise[noise_idxs[idx: idx + batch_size]]
            noised_latent = latent * (scheduler.alphas_cumprod[batch_ts] ** 0.5).view(-1, 1, 1, 1).to(device) + \
                            noise * ((1 - scheduler.alphas_cumprod[batch_ts]) ** 0.5).view(-1, 1, 1, 1).to(device)
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
            pred_errors[idx: idx + len(batch_ts)] = error.detach().cpu()
            idx += len(batch_ts)
    return pred_errors

def encode_text(clip_model, text_encoder, classnames, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    prompt_learner = PromptLearner(classnames, clip_model)
    prompt_learner.load_state_dict(checkpoint['model_state_dict'])
    prompts = prompt_learner()
    text_input = prompt_learner.tokenized_prompts                   
    embeddings = []
    with torch.inference_mode():
        for i in range(0, len(text_input), 100):
            text_embeddings = text_encoder(prompts[i:i+100].to(device), text_input[i:i+100].to(device))
            embeddings.append(text_embeddings) 
    text_embeddings = torch.cat(embeddings, dim=0) 
    assert len(text_embeddings) == len(classnames)
    return text_embeddings

def main():
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument('--dataset', type=str, default='oxford_pets',
                        choices=['oxford_pets', 'flowers', 'stl10', 'mnist', 'cifar10', 'food', 'caltech101', 'imagenet',
                                 'objectnet', 'aircraft', 'stanford_cars'], help='Dataset to use')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'], help='Name of split')
    parser.add_argument('--shots', type=int, default=16, choices=('1,2,4,8,16'), help='number of shots')

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
        run_folder = osp.join(LOG_DIR, args.dataset, name)
    os.makedirs(run_folder, exist_ok=True)
    print(f'Run folder: {run_folder}')

    # load pretrained models
    vae, tokenizer, text_encoder, unet, scheduler = get_sd_model(args)
    vae = vae.to(device)
    clip_model = load_clip_to_cpu("ViT-L/14") 
    text_encoder = TextEncoder(clip_model).to(device)
    unet = unet.to(device)
    torch.backends.cudnn.benchmark = True

    # set up dataset
    interpolation = INTERPOLATIONS[args.interpolation]
    transform = get_transform(interpolation, args.img_size)
    latent_size = args.img_size // 8
    target_dataset = build_dataset(args.dataset, DATASET_ROOT, args.shots)  #参数之后再改
    test_loader = build_data_loader(data_source=target_dataset.test, batch_size=1, is_train=False, tfm=transform, shuffle=False)

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
    classnames_speices = df_species_sorted['Variants'].tolist()
    # level 2
    df_family_sorted = df.sort_values(by='family label')
    df_family_sorted = df_family_sorted.drop_duplicates(subset='family label')
    classnames_family = df_family_sorted['Families'].tolist()
    # level 1
    df_order_sorted = df.sort_values(by='Manufactures label')
    df_order_sorted = df_order_sorted.drop_duplicates(subset='Manufactures label')
    classnames_order = df_order_sorted['Manufactures'].tolist()
    
    checkpoint_path = 'data/aircraft/v1-5_1trials_1keep_25samples_16shots_order_csc/30.pth'
    text_embeddings_order = encode_text(clip_model, text_encoder, classnames_order, checkpoint_path)
    checkpoint_path = 'data/aircraft/v1-5_1trials_1keep_25samples_16shots_family_csc/30.pth'
    text_embeddings_family = encode_text(clip_model, text_encoder, classnames_family, checkpoint_path)
    checkpoint_path = 'data/aircraft/v1-5_1trials_1keep_25samples_16shots_species_csc/25.pth'
    text_embeddings_species = encode_text(clip_model, text_encoder, classnames_speices, checkpoint_path)
    
    num_order = 30
    embeddings_family = []
    for idx in range(num_order):
        classes_list = get_family(idx+1)
        text_embeddings = text_embeddings_family[classes_list]
        embeddings_family.append(text_embeddings)    
        
    num_family = 70
    embeddings_species = []
    for idx in range(num_family):
        classes_list = get_species(idx+1)
        text_embeddings = text_embeddings_species[classes_list]
        embeddings_species.append(text_embeddings)    

    formatstr = get_formatstr(len(test_loader) - 1)
    correct = 0
    correct_order = 0
    correct_family = 0
    total = 0
    pbar = tqdm.tqdm(test_loader)
    for i, (image, label) in enumerate(pbar):
        if total > 0:
            pbar.set_description(f'Acc: {100 * correct / total:.2f}%, Acc_family: {100 * correct_family / total:.2f}%, Acc_order: {100 * correct_order / total:.2f}%')

        # get high level label 
        label_order, label_family = get_order_family_target(label)   

        with torch.no_grad():
            img_input = image.to(device)
            label = label.to(device)
            if args.dtype == 'float16':
                img_input = img_input.half()
            x0 = vae.encode(img_input).latent_dist.mean
            x0 *= 0.18215
        
        # stage one
        args.n_samples = [5, 25]
        args.to_keep = [5, 1]
        pred_order, pred_errors, error = eval_prob_adaptive(unet, x0, text_embeddings_order, scheduler, args, latent_size, all_noise)

        # stage two
        text_embeddings_family = embeddings_family[pred_order]
        if (len(text_embeddings_family)>1):
            args.n_samples = [25]
        else:
            args.n_samples = [1]
        args.to_keep = [1]
        pred_idx, pred_errors, error = eval_prob_adaptive(unet, x0, text_embeddings_family, scheduler, args, latent_size, all_noise)        
        pred_family = get_family(pred_order+1)[pred_idx]

        # stage three
        text_embeddings_species = embeddings_species[pred_family]
        if (len(text_embeddings_species)>1):
            args.n_samples = [25]
        else:
            args.n_samples = [1]
        args.to_keep = [1]
        pred_idx, pred_errors, error = eval_prob_adaptive(unet, x0, text_embeddings_species, scheduler, args, latent_size, all_noise)
        pred_species = get_species(pred_family+1)[pred_idx]

        pred_species = torch.from_numpy(np.array([pred_species])).to(device)
        if pred_order == label_order:
            correct_order += 1
        if pred_family == label_family:
            correct_family += 1
        if pred_species == label:
            correct += 1
        total += 1

    with open('results_train.txt', 'a') as file:
        file.write('Acc = %.2f \n' % (100 * correct / total))


if __name__ == '__main__':
    main()
