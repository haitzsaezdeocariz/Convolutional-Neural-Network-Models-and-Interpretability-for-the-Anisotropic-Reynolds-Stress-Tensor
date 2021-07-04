import numpy as np
import pandas as pd


## Channel flow DNS parameters

def second_smallest(numbers):
        m1, m2 = float('inf'), float('inf')
        for x in numbers:
            if x <= m1:
                m1, m2 = x, m1
            elif x < m2:
                m2 = x
        return m2

U_TAU = [6.19417e-02, 5.48697e-02, 5.01370e-02]
RE_TAU = [92.913, 219.479, 501.370]
tags = [93, 220, 500]

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

        du_dy = (grad_u[:, 0, 1])
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
        for i in range(len(k_true)):
            if k_true[i]==0:
                k_true[i]=second_smallest(k_true)
        anisotropy = stresses - 2./3. * k_true[:, None, None] * np.eye(3)

        if self.nondim == 'k-eps':
            anisotropy /= 2.*k_true[:, None, None]

        elif self.nondim == 'h-u_bulk':
            anisotropy /= channel_param['u_bulk']**2

        else:
            anisotropy /= channel_param['u_tau']**2

        return anisotropy


def load_channel_data(filepath):

    data = np.loadtxt(filepath, skiprows=1)

    y_plus = data[:, 1]
    k = data[:, 2]
    eps = data[:, 3]
    grad_u_flat = data[:, 4:13]
    stresses_flat = data[:, 13:22]
    u = data[:, 22:]

    grad_u = grad_u_flat.reshape(-1, 3, 3)
    stresses = stresses_flat.reshape(-1, 3, 3)

    return y_plus, k, eps, grad_u, stresses, u

def load_boundary_data(filepath):
    
    data = np.loadtxt(filepath, skiprows=1)

    y_plus_BC = data[0, 1]
    k_BC = data[0, 2]
    eps_BC = data[0, 3]
    grad_u_flat_BC = data[0, 4:13]
    stresses_flat_BC = data[0, 13:22]
    u_BC = data[0, 22:]

    grad_u_BC = grad_u_flat_BC.reshape(-1, 3, 3)
    stresses_BC = stresses_flat_BC.reshape(-1, 3, 3)

    return y_plus_BC, k_BC, eps_BC, grad_u_BC, stresses_BC, u_BC

def make_dataframe(Retau, source='LM', nondim='h-u_bulk'):

    filepath = '../data/{0}_Couette_Retau{1}.txt'.format(source, Retau)
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


def BC_dataframe(Retau, source='LM', nondim='h-u_bulk'):

    filepath = '../data/{0}_Couette_Retau{1}.txt'.format(source, Retau)
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

