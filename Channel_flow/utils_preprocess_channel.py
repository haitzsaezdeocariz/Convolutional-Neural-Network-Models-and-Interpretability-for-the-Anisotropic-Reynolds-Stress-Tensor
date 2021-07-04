import numpy as np
import pandas as pd


## Channel flow DNS parameters

U_TAU = [6.37309e-02, 5.43496e-02, 5.00256e-02, 4.58794e-02, 4.14872e-02]
RE_TAU = [182.088, 543.496, 1000.512, 1994.756, 5185.897]
tags = [180, 550, 1000, 2000, 5200]

channel_params = {tag: {} for tag in tags}
for tag, u_tau, Re_tau in zip(tags, U_TAU, RE_TAU):
    channel_params[tag]['Re_tau'] = Re_tau
    channel_params[tag]['u_tau'] = u_tau
    channel_params[tag]['u_bulk'] = 1.0


## Utility functions for preprocessing

class ChannelDataProcessor:

    def __init__(self, nondim):

        assert nondim in ('h-u_bulk', 'h_nu-u_tau', 'k-eps')
        self.nondim = nondim

    def calc_du_dy(self, grad_u, k, eps, channel_param):

        du_dy = grad_u[:, 0, 1]

        if self.nondim == 'k-eps':
            du_dy *= k / (eps)
        elif self.nondim == 'h-u_bulk':
            du_dy *= channel_param['u_tau'] / channel_param['u_bulk'] * channel_param['Re_tau']
        else:
            pass

        return du_dy

    def calc_anisotropy(self, stresses, channel_param):

        stresses *= channel_param['u_tau']**2
        k_true = 0.5 * (stresses[:, 0, 0] + stresses[:, 1, 1] + stresses[:, 2, 2])
        anisotropy = stresses - 2./3. * k_true[:,None,None] * np.eye(3)

        if self.nondim == 'k-eps':
            anisotropy /= 2.*k_true[:, None, None]

        elif self.nondim == 'h-u_bulk':
            anisotropy /= channel_param['u_bulk']**2

        else:
            anisotropy /= channel_param['u_tau']**2

        return anisotropy


def load_channel_data(filepath):

    data = np.loadtxt(filepath, skiprows=1)

    y_plus = data[:, 0]
    k = data[:, 1]
    eps = data[:, 2]
    grad_u_flat = data[:, 3:12]
    stresses_flat = data[:, 12:21]
    u = data[:, 21:]

    grad_u = grad_u_flat.reshape(-1, 3, 3)
    stresses = stresses_flat.reshape(-1, 3, 3)

    return y_plus, k, eps, grad_u, stresses, u

def load_boundary_data(filepath):
    
    data = np.loadtxt(filepath, skiprows=1)

    y_plus_BC = data[0, 0]
    k_BC = data[0, 1]
    eps_BC = data[0, 2]
    grad_u_flat_BC = data[0, 3:12]
    stresses_flat_BC = data[0, 12:21]
    u_BC = data[0, 21:]

    grad_u_BC = grad_u_flat_BC.reshape(-1, 3, 3)
    stresses_BC = stresses_flat_BC.reshape(-1, 3, 3)

    return y_plus_BC, k_BC, eps_BC, grad_u_BC, stresses_BC, u_BC

def make_dataframe(Retau, source='LM_cg1', nondim='h-u_bulk'):

    filepath = '../data/{0}_Channel_Retau{1}.txt'.format(source, Retau)
    channel_param = channel_params[Retau]

    y_plus, k, eps, grad_u, stresses, u = load_channel_data(filepath)
    y = y_plus / channel_param['Re_tau']
    
    data_processor = ChannelDataProcessor(nondim)
    du_dy = data_processor.calc_du_dy(grad_u, k, eps, channel_param)
    anisotropy = data_processor.calc_anisotropy(stresses, channel_param)

    df = pd.DataFrame(
        {'y+': y_plus,
         'y': y,
         'index': np.arange(len(y)),
         'du_dy': du_dy,
         'a_uv': anisotropy[:, 0, 1],
         'a_uu': anisotropy[:, 0, 0],
         'a_vv': anisotropy[:, 1, 1],
         'a_ww': anisotropy[:, 2, 2],
         'Re_tau': [channel_param['Re_tau']] * len(y),
         'DU_DY': [du_dy] * len(y),
         'Y': [y] * len(y)}
    )

    return df


def BC_dataframe(Retau, source='LM_cg1', nondim='h-u_bulk'):

    filepath = '../data/{0}_Channel_Retau{1}.txt'.format(source, Retau)
    channel_param = channel_params[Retau]

    y_plus_BC, k_BC, eps_BC, grad_u_BC, stresses_BC, u_BC = load_boundary_data(filepath)
    y_BC = y_plus_BC / channel_param['Re_tau']

    data_processor_BC = ChannelDataProcessor(nondim)
    du_dy_BC = data_processor_BC.calc_du_dy(grad_u_BC, k_BC, eps_BC, channel_param)
    anisotropy_BC = data_processor_BC.calc_anisotropy(stresses_BC, channel_param)

    
    df_BC = pd.DataFrame(
        {'y+': y_plus_BC,
         'y': y_BC,
         'index': np.arange(1),
         'du_dy': du_dy_BC,
         'a_uv': anisotropy_BC[:, 0, 1],
         'a_uu': anisotropy_BC[:, 0, 0],
         'a_vv': anisotropy_BC[:, 1, 1],
         'a_ww': anisotropy_BC[:, 2, 2],
         'Re_tau': [channel_param['Re_tau']],
         'DU_DY': [du_dy_BC],
         'Y': [y_BC]}
    )

    return df_BC

