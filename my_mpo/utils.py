from tensorboardX import SummaryWriter
import shutil
import os
import tqdm
import time
import torch

def copy_model(trained, target):
    """
    copy parameters from trained model to target model
    """
    for target_param, trained_param in zip(target.parameters(), trained.parameters()):
        target_param.data.copy_(trained_param.data)
        target_param.requires_grad = False 
        
    return target

def gaussian_kl(mu_i, mu, cholesky_i, cholesky):
    """
    decoupled KL between two multivariate gaussian distribution
    C_μ = KL(f(x|μi,Σi)||f(x|μ,Σi))
    C_Σ = KL(f(x|μi,Σi)||f(x|μi,Σ))
    :param μi: (B, n)
    :param μ: (B, n)
    :param Ai: (B, n, n)
    :param A: (B, n, n)
    :return: C_μ, C_Σ: scalar
        mean and covariance terms of the KL
    :return: mean of determinanats of Σi, Σ
    ref : https://stanford.edu/~jduchi/projects/general_notes.pdf page.13
    """
    n = cholesky.size(-1)
    mu_i = mu_i.unsqueeze(-1)  # (B, n, 1)
    mu = mu.unsqueeze(-1)  # (B, n, 1)
    cov_i = cholesky_i @ (cholesky_i.transpose(dim0=-2, dim1=-1))  # (B, n, n)
    cov = cholesky @ (cholesky.transpose(dim0=-2, dim1=-1))  # (B, n, n)
    cov_i_det = cov_i.det()  # (B,)
    cov_det = cov.det()  # (B,)
    # determinant can be minus due to numerical calculation error
    # https://github.com/daisatojp/mpo/issues/11
    cov_i_det = torch.clamp_min(cov_i_det, 1e-6)
    cov_det = torch.clamp_min(cov_det, 1e-6)
    cov_i_inv = cov_i.inverse()  # (B, n, n)
    cov_inv = cov.inverse()  # (B, n, n)
    inner_mu = ((mu - mu_i).transpose(-2, -1) @ cov_i_inv @ (mu - mu_i)).squeeze()  # (B,)
    inner_cov = torch.log(cov_det / cov_i_det) - n + ((cov_inv @ cov_i).diagonal(dim1=-2, dim2=-1).sum(-1))  # (B,)
    C_mu = 0.5 * torch.mean(inner_mu)
    C_cov = 0.5 * torch.mean(inner_cov)
    return C_mu, C_cov


_log_path = None

def set_log_path(path):
    global _log_path
    _log_path = path

def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)
            
def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)

class Timer():
    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v
    
def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)

def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer

def separate_info(time_step):
    next_state = time_step.observation['observations']
    reward = time_step.reward
    done = time_step.last()
    return next_state, reward, done, None