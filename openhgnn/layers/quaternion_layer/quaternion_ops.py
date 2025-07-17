##########################################################
# pytorch-qnn v1.0
# Titouan Parcollet
# LIA, Université d'Avignon et des Pays du Vaucluse
# ORKIS, Aix-en-provence
# October 2018
##########################################################

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from numpy.random import RandomState
import sys
import pdb
from scipy.stats import chi

def q_normalize(input, channel=1):

    r = get_r(input)
    i = get_i(input)
    j = get_j(input)
    k = get_k(input)

    norm = torch.sqrt(r*r + i*i + j*j + k*k + 0.0001)
    r = r / norm
    i = i / norm
    j = j / norm
    k = k / norm

    return torch.cat([r,i,j,k], dim=channel)


def check_input(input):

    if input.dim() not in {2, 3, 4, 5}:
        raise RuntimeError(
            "Quaternion linear accepts only input of dimension 2 or 3. Quaternion conv accepts up to 5 dim "
            " input.dim = " + str(input.dim())
        )

    if input.dim() < 4:
        nb_hidden = input.size()[-1]
    else:
        nb_hidden = input.size()[1]

    if nb_hidden % 4 != 0:
        raise RuntimeError(
            "Quaternion Tensors must be divisible by 4."
            " input.size()[1] = " + str(nb_hidden)
        )
#
# Getters
#
def get_r(input):
    check_input(input)
    if input.dim() < 4:
        nb_hidden = input.size()[-1]
    else:
        nb_hidden = input.size()[1]

    if input.dim() == 2:
        return input.narrow(1, 0, nb_hidden // 4)
    if input.dim() == 3:
        return input.narrow(2, 0, nb_hidden // 4)
    if input.dim() >= 4:
        return input.narrow(1, 0, nb_hidden // 4)


def get_i(input):
    if input.dim() < 4:
        nb_hidden = input.size()[-1]
    else:
        nb_hidden = input.size()[1]
    if input.dim() == 2:
        return input.narrow(1, nb_hidden // 4, nb_hidden // 4)
    if input.dim() == 3:
        return input.narrow(2, nb_hidden // 4, nb_hidden // 4)
    if input.dim() >= 4:
        return input.narrow(1, nb_hidden // 4, nb_hidden // 4)

def get_j(input):
    check_input(input)
    if input.dim() < 4:
        nb_hidden = input.size()[-1]
    else:
        nb_hidden = input.size()[1]
    if input.dim() == 2:
        return input.narrow(1, nb_hidden // 2, nb_hidden // 4)
    if input.dim() == 3:
        return input.narrow(2, nb_hidden // 2, nb_hidden // 4)
    if input.dim() >= 4:
        return input.narrow(1, nb_hidden // 2, nb_hidden // 4)

def get_k(input):
    check_input(input)
    if input.dim() < 4:
        nb_hidden = input.size()[-1]
    else:
        nb_hidden = input.size()[1]
    if input.dim() == 2:
        return input.narrow(1, nb_hidden - nb_hidden // 4, nb_hidden // 4)
    if input.dim() == 3:
        return input.narrow(2, nb_hidden - nb_hidden // 4, nb_hidden // 4)
    if input.dim() >= 4:
        return input.narrow(1, nb_hidden - nb_hidden // 4, nb_hidden // 4)


def get_modulus(input, vector_form=False):
    check_input(input)
    r = get_r(input)
    i = get_i(input)
    j = get_j(input)
    k = get_k(input)
    if vector_form:
        return torch.sqrt(r * r + i * i + j * j + k * k)
    else:
        return torch.sqrt((r * r + i * i + j * j + k * k).sum(dim=0))


def get_normalized(input, eps=0.0001):
    check_input(input)
    data_modulus = get_modulus(input)
    if input.dim() == 2:
        data_modulus_repeated = data_modulus.repeat(1, 4)
    elif input.dim() == 3:
        data_modulus_repeated = data_modulus.repeat(1, 1, 4)
    return input / (data_modulus_repeated.expand_as(input) + eps)


def quaternion_exp(input):

    r      = get_r(input)
    i      = get_i(input)
    j      = get_j(input)
    k      = get_k(input)


    norm_v = torch.sqrt(i*i+j*j+k*k) + 0.0001
    exp    = torch.exp(r)

    r      = torch.cos(norm_v)
    i      = (i / norm_v) * torch.sin(norm_v)
    j      = (j / norm_v) * torch.sin(norm_v)
    k      = (k / norm_v) * torch.sin(norm_v)


    return torch.cat([exp*r, exp*i, exp*j, exp*k], dim=1)


def quaternion_conv(input, r_weight, i_weight, j_weight, k_weight, bias, stride,
                    padding, groups, dilatation):
    """
    Applies a quaternion convolution to the incoming data:
    """

    cat_kernels_4_r = torch.cat([r_weight, -i_weight, -j_weight, -k_weight], dim=1)
    cat_kernels_4_i = torch.cat([i_weight,  r_weight, -k_weight, j_weight], dim=1)
    cat_kernels_4_j = torch.cat([j_weight,  k_weight, r_weight, -i_weight], dim=1)
    cat_kernels_4_k = torch.cat([k_weight,  -j_weight, i_weight, r_weight], dim=1)

    cat_kernels_4_quaternion   = torch.cat([cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], dim=0)

    if   input.dim() == 3:
        convfunc = F.conv1d
    elif input.dim() == 4:
        convfunc = F.conv2d
    elif input.dim() == 5:
        convfunc = F.conv3d
    else:
        raise Exception("The convolutional input is either 3, 4 or 5 dimensions."
                        " input.dim = " + str(input.dim()))

    return convfunc(input, cat_kernels_4_quaternion, bias, stride, padding, dilatation, groups)

def quaternion_transpose_conv(input, r_weight, i_weight, j_weight, k_weight, bias, stride,
                    padding, output_padding, groups, dilatation):
    """
    Applies a quaternion trasposed convolution to the incoming data:

    """

    cat_kernels_4_r = torch.cat([r_weight, -i_weight, -j_weight, -k_weight], dim=1)
    cat_kernels_4_i = torch.cat([i_weight,  r_weight, -k_weight, j_weight], dim=1)
    cat_kernels_4_j = torch.cat([j_weight,  k_weight, r_weight, -i_weight], dim=1)
    cat_kernels_4_k = torch.cat([k_weight,  -j_weight, i_weight, r_weight], dim=1)
    cat_kernels_4_quaternion   = torch.cat([cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], dim=0)

    if   input.dim() == 3:
        convfunc = F.conv_transpose1d
    elif input.dim() == 4:
        convfunc = F.conv_transpose2d
    elif input.dim() == 5:
        convfunc = F.conv_transpose3d
    else:
        raise Exception("The convolutional input is either 3, 4 or 5 dimensions."
                        " input.dim = " + str(input.dim()))

    return convfunc(input, cat_kernels_4_quaternion, bias, stride, padding, output_padding, groups, dilatation)


def quaternion_conv_rotation(input, zero_kernel, r_weight, i_weight, j_weight, k_weight, bias, stride,
                    padding, groups, dilatation, quaternion_format, scale=None):
    """
    Applies a quaternion rotation and convolution transformation to the incoming data:

    The rotation W*x*W^t can be replaced by R*x following:
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    Works for unitary and non unitary weights.

    The initial size of the input must be a multiple of 3 if quaternion_format = False and
    4 if quaternion_format = True.
    """

    square_r          = (r_weight*r_weight)
    square_i          = (i_weight*i_weight)
    square_j          = (j_weight*j_weight)
    square_k          = (k_weight*k_weight)

    norm              = torch.sqrt(square_r+square_i+square_j+square_k + 0.0001)

    #print(norm)

    r_n_weight          = (r_weight / norm)
    i_n_weight          = (i_weight / norm)
    j_n_weight          = (j_weight / norm)
    k_n_weight          = (k_weight / norm)

    norm_factor       = 2.0

    square_i          = norm_factor*(i_n_weight*i_n_weight)
    square_j          = norm_factor*(j_n_weight*j_n_weight)
    square_k          = norm_factor*(k_n_weight*k_n_weight)

    ri                = (norm_factor*r_n_weight*i_n_weight)
    rj                = (norm_factor*r_n_weight*j_n_weight)
    rk                = (norm_factor*r_n_weight*k_n_weight)

    ij                = (norm_factor*i_n_weight*j_n_weight)
    ik                = (norm_factor*i_n_weight*k_n_weight)

    jk                = (norm_factor*j_n_weight*k_n_weight)

    if quaternion_format:
        if scale is not None:
            rot_kernel_1  = torch.cat([zero_kernel, scale * (1.0 - (square_j + square_k)), scale *(ij-rk), scale *(ik+rj)], dim=1)
            rot_kernel_2  = torch.cat([zero_kernel, scale *(ij+rk), scale *(1.0 - (square_i + square_k)), scale *(jk-ri)], dim=1)
            rot_kernel_3  = torch.cat([zero_kernel, scale *(ik-rj), scale *(jk+ri), scale *(1.0 - (square_i + square_j))], dim=1)
        else:
            rot_kernel_1  = torch.cat([zero_kernel, (1.0 - (square_j + square_k)), (ij-rk), (ik+rj)], dim=1)
            rot_kernel_2  = torch.cat([zero_kernel, (ij+rk), (1.0 - (square_i + square_k)), (jk-ri)], dim=1)
            rot_kernel_3  = torch.cat([zero_kernel, (ik-rj), (jk+ri), (1.0 - (square_i + square_j))], dim=1)

        zero_kernel2  = torch.cat([zero_kernel, zero_kernel, zero_kernel, zero_kernel], dim=1)
        global_rot_kernel = torch.cat([zero_kernel2, rot_kernel_1, rot_kernel_2, rot_kernel_3], dim=0)

    else:
        if scale is not None:
            rot_kernel_1  = torch.cat([scale * (1.0 - (square_j + square_k)), scale *(ij-rk), scale *(ik+rj)], dim=0)
            rot_kernel_2  = torch.cat([scale *(ij+rk), scale *(1.0 - (square_i + square_k)), scale *(jk-ri)], dim=0)
            rot_kernel_3  = torch.cat([scale *(ik-rj), scale *(jk+ri), scale *(1.0 - (square_i + square_j))], dim=0)
        else:
            rot_kernel_1  = torch.cat([1.0 - (square_j + square_k), (ij-rk), (ik+rj)], dim=0)
            rot_kernel_2  = torch.cat([(ij+rk), 1.0 - (square_i + square_k), (jk-ri)], dim=0)
            rot_kernel_3  = torch.cat([(ik-rj), (jk+ri), (1.0 - (square_i + square_j))], dim=0)

        global_rot_kernel = torch.cat([rot_kernel_1, rot_kernel_2, rot_kernel_3], dim=0)

    #print(input.shape)
    #print(square_r.shape)
    #print(global_rot_kernel.shape)

    if   input.dim() == 3:
        convfunc = F.conv1d
    elif input.dim() == 4:
        convfunc = F.conv2d
    elif input.dim() == 5:
        convfunc = F.conv3d
    else:
        raise Exception("The convolutional input is either 3, 4 or 5 dimensions."
                        " input.dim = " + str(input.dim()))

    return convfunc(input, global_rot_kernel, bias, stride, padding, dilatation, groups)

def quaternion_transpose_conv_rotation(input, zero_kernel, r_weight, i_weight, j_weight, k_weight, bias, stride,
                    padding, output_padding, groups, dilatation, quaternion_format):
    """
    Applies a quaternion rotation and transposed convolution transformation to the incoming data:

    The rotation W*x*W^t can be replaced by R*x following:
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    Works for unitary and non unitary weights.

    The initial size of the input must be a multiple of 3 if quaternion_format = False and
    4 if quaternion_format = True.

    """

    square_r          = (r_weight*r_weight)
    square_i          = (i_weight*i_weight)
    square_j          = (j_weight*j_weight)
    square_k          = (k_weight*k_weight)

    norm              = torch.sqrt(square_r+square_i+square_j+square_k + 0.0001)

    r_weight          = (r_weight / norm)
    i_weight          = (i_weight / norm)
    j_weight          = (j_weight / norm)
    k_weight          = (k_weight / norm)

    norm_factor       = 2.0

    square_i          = norm_factor*(i_weight*i_weight)
    square_j          = norm_factor*(j_weight*j_weight)
    square_k          = norm_factor*(k_weight*k_weight)

    ri                = (norm_factor*r_weight*i_weight)
    rj                = (norm_factor*r_weight*j_weight)
    rk                = (norm_factor*r_weight*k_weight)

    ij                = (norm_factor*i_weight*j_weight)
    ik                = (norm_factor*i_weight*k_weight)

    jk                = (norm_factor*j_weight*k_weight)

    if quaternion_format:
        rot_kernel_1  = torch.cat([zero_kernel, 1.0 - (square_j + square_k), ij-rk, ik+rj], dim=1)
        rot_kernel_2  = torch.cat([zero_kernel, ij+rk, 1.0 - (square_i + square_k), jk-ri], dim=1)
        rot_kernel_3  = torch.cat([zero_kernel, ik-rj, jk+ri, 1.0 - (square_i + square_j)], dim=1)

        zero_kernel2  = torch.zeros(rot_kernel_1.shape).cuda()
        global_rot_kernel = torch.cat([zero_kernel2, rot_kernel_1, rot_kernel_2, rot_kernel_3], dim=0)
    else:
        rot_kernel_1  = torch.cat([1.0 - (square_j + square_k), ij-rk, ik+rj], dim=1)
        rot_kernel_2  = torch.cat([ij+rk, 1.0 - (square_i + square_k), jk-ri], dim=1)
        rot_kernel_3  = torch.cat([ik-rj, jk+ri, 1.0 - (square_i + square_j)], dim=1)
        global_rot_kernel = torch.cat([rot_kernel_1, rot_kernel_2, rot_kernel_3], dim=0)


    if   input.dim() == 3:
        convfunc = F.conv_transpose1d
    elif input.dim() == 4:
        convfunc = F.conv_transpose2d
    elif input.dim() == 5:
        convfunc = F.conv_transpose3d
    else:
        raise Exception("The convolutional input is either 3, 4 or 5 dimensions."
                        " input.dim = " + str(input.dim()))

    return convfunc(input, cat_kernels_4_quaternion, bias, stride, padding, output_padding, groups, dilatation)


def quaternion_linear(input, r_weight, i_weight, j_weight, k_weight, bias=True):
    """
    Applies a quaternion linear transformation to the incoming data:

    It is important to notice that the forward phase of a QNN is defined
    as W * Inputs (with * equal to the Hamilton product). The constructed
    cat_kernels_4_quaternion is a modified version of the quaternion representation
    so when we do torch.mm(Input,W) it's equivalent to W * Inputs.

    """

    cat_kernels_4_r = torch.cat([r_weight, -i_weight, -j_weight, -k_weight], dim=0)
    cat_kernels_4_i = torch.cat([i_weight,  r_weight, -k_weight, j_weight], dim=0)
    cat_kernels_4_j = torch.cat([j_weight,  k_weight, r_weight, -i_weight], dim=0)
    cat_kernels_4_k = torch.cat([k_weight,  -j_weight, i_weight, r_weight], dim=0)
    cat_kernels_4_quaternion   = torch.cat([cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], dim=1)

    if input.dim() == 2 :

        if bias is not None:
            return torch.addmm(bias, input, cat_kernels_4_quaternion)
        else:
            return torch.mm(input, cat_kernels_4_quaternion)
    else:
        output = torch.matmul(input, cat_kernels_4_quaternion)
        if bias is not None:
            return output+bias
        else:
            return output


def quaternion_linear_rotation(input, zero_kernel, r_weight, i_weight, j_weight, k_weight, bias=None,
                               quaternion_format=False, scale=None):
    """
    Applies a quaternion rotation transformation to the incoming data:

    The rotation W*x*W^t can be replaced by R*x following:
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    Works for unitary and non unitary weights.

    The initial size of the input must be a multiple of 3 if quaternion_format = False and
    4 if quaternion_format = True.
    """

    square_r          = (r_weight*r_weight)
    square_i          = (i_weight*i_weight)
    square_j          = (j_weight*j_weight)
    square_k          = (k_weight*k_weight)

    norm              = torch.sqrt(square_r+square_i+square_j+square_k + 0.0001)

    r_n_weight          = (r_weight / norm)
    i_n_weight          = (i_weight / norm)
    j_n_weight          = (j_weight / norm)
    k_n_weight          = (k_weight / norm)

    norm_factor       = 2.0

    square_i          = norm_factor*(i_n_weight*i_n_weight)
    square_j          = norm_factor*(j_n_weight*j_n_weight)
    square_k          = norm_factor*(k_n_weight*k_n_weight)

    ri                = (norm_factor*r_n_weight*i_n_weight)
    rj                = (norm_factor*r_n_weight*j_n_weight)
    rk                = (norm_factor*r_n_weight*k_n_weight)

    ij                = (norm_factor*i_n_weight*j_n_weight)
    ik                = (norm_factor*i_n_weight*k_n_weight)

    jk                = (norm_factor*j_n_weight*k_n_weight)

    if quaternion_format:
        if scale is not None:
            rot_kernel_1  = torch.cat([zero_kernel, scale * (1.0 - (square_j + square_k)), scale *(ij-rk), scale *(ik+rj)], dim=0)
            rot_kernel_2  = torch.cat([zero_kernel, scale *(ij+rk), scale *(1.0 - (square_i + square_k)), scale *(jk-ri)], dim=0)
            rot_kernel_3  = torch.cat([zero_kernel, scale *(ik-rj), scale *(jk+ri), scale *(1.0 - (square_i + square_j))], dim=0)
        else:
            rot_kernel_1  = torch.cat([zero_kernel, (1.0 - (square_j + square_k)), (ij-rk), (ik+rj)], dim=0)
            rot_kernel_2  = torch.cat([zero_kernel, (ij+rk), (1.0 - (square_i + square_k)), (jk-ri)], dim=0)
            rot_kernel_3  = torch.cat([zero_kernel, (ik-rj), (jk+ri), (1.0 - (square_i + square_j))], dim=0)

        zero_kernel2  = torch.cat([zero_kernel, zero_kernel, zero_kernel, zero_kernel], dim=0)
        global_rot_kernel = torch.cat([zero_kernel2, rot_kernel_1, rot_kernel_2, rot_kernel_3], dim=1)

    else:
        if scale is not None:
            rot_kernel_1  = torch.cat([scale * (1.0 - (square_j + square_k)), scale *(ij-rk), scale *(ik+rj)], dim=0)
            rot_kernel_2  = torch.cat([scale *(ij+rk), scale *(1.0 - (square_i + square_k)), scale *(jk-ri)], dim=0)
            rot_kernel_3  = torch.cat([scale *(ik-rj), scale *(jk+ri), scale *(1.0 - (square_i + square_j))], dim=0)
        else:
            rot_kernel_1  = torch.cat([1.0 - (square_j + square_k), (ij-rk), (ik+rj)], dim=0)
            rot_kernel_2  = torch.cat([(ij+rk), 1.0 - (square_i + square_k), (jk-ri)], dim=0)
            rot_kernel_3  = torch.cat([(ik-rj), (jk+ri), (1.0 - (square_i + square_j))], dim=0)

        global_rot_kernel = torch.cat([rot_kernel_1, rot_kernel_2, rot_kernel_3], dim=1)


    if input.dim() == 2 :
        if bias is not None:
            return torch.addmm(bias, input, global_rot_kernel)
        else:
            return torch.mm(input, global_rot_kernel)
    else:
        output = torch.matmul(input, global_rot_kernel)
        if bias is not None:
            return output+bias
        else:
            return output



# Custom AUTOGRAD for lower VRAM consumption
class QuaternionLinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, r_weight, i_weight, j_weight, k_weight, bias=None):
        ctx.save_for_backward(input, r_weight, i_weight, j_weight, k_weight, bias)
        check_input(input)
        cat_kernels_4_r = torch.cat([r_weight, -i_weight, -j_weight, -k_weight], dim=0)
        cat_kernels_4_i = torch.cat([i_weight,  r_weight, -k_weight, j_weight], dim=0)
        cat_kernels_4_j = torch.cat([j_weight,  k_weight, r_weight, -i_weight], dim=0)
        cat_kernels_4_k = torch.cat([k_weight,  -j_weight, i_weight, r_weight], dim=0)
        cat_kernels_4_quaternion = torch.cat([cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], dim=1)
        if input.dim() == 2 :
            if bias is not None:
                return torch.addmm(bias, input, cat_kernels_4_quaternion)
            else:
                return torch.mm(input, cat_kernels_4_quaternion)
        else:
            output = torch.matmul(input, cat_kernels_4_quaternion)
            if bias is not None:
                return output+bias
            else:
                return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):

        input, r_weight, i_weight, j_weight, k_weight, bias = ctx.saved_tensors
        grad_input = grad_weight_r = grad_weight_i = grad_weight_j = grad_weight_k = grad_bias = None

        input_r = torch.cat([r_weight, -i_weight, -j_weight, -k_weight], dim=0)
        input_i = torch.cat([i_weight,  r_weight, -k_weight, j_weight], dim=0)
        input_j = torch.cat([j_weight,  k_weight, r_weight, -i_weight], dim=0)
        input_k = torch.cat([k_weight,  -j_weight, i_weight, r_weight], dim=0)
        cat_kernels_4_quaternion_T = Variable(torch.cat([input_r, input_i, input_j, input_k], dim=1).permute(1,0), requires_grad=False)

        r = get_r(input)
        i = get_i(input)
        j = get_j(input)
        k = get_k(input)
        input_r = torch.cat([r, -i, -j, -k], dim=0)
        input_i = torch.cat([i,  r, -k, j], dim=0)
        input_j = torch.cat([j,  k, r, -i], dim=0)
        input_k = torch.cat([k,  -j, i, r], dim=0)
        input_mat = Variable(torch.cat([input_r, input_i, input_j, input_k], dim=1), requires_grad=False)

        r = get_r(grad_output)
        i = get_i(grad_output)
        j = get_j(grad_output)
        k = get_k(grad_output)
        input_r = torch.cat([r, i, j, k], dim=1)
        input_i = torch.cat([-i,  r, k, -j], dim=1)
        input_j = torch.cat([-j,  -k, r, i], dim=1)
        input_k = torch.cat([-k,  j, -i, r], dim=1)
        grad_mat = torch.cat([input_r, input_i, input_j, input_k], dim=0)

        if ctx.needs_input_grad[0]:
            grad_input  = grad_output.mm(cat_kernels_4_quaternion_T)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_mat.permute(1,0).mm(input_mat).permute(1,0)
            unit_size_x = r_weight.size(0)
            unit_size_y = r_weight.size(1)
            grad_weight_r = grad_weight.narrow(0,0,unit_size_x).narrow(1,0,unit_size_y)
            grad_weight_i = grad_weight.narrow(0,0,unit_size_x).narrow(1,unit_size_y,unit_size_y)
            grad_weight_j = grad_weight.narrow(0,0,unit_size_x).narrow(1,unit_size_y*2,unit_size_y)
            grad_weight_k = grad_weight.narrow(0,0,unit_size_x).narrow(1,unit_size_y*3,unit_size_y)
        if ctx.needs_input_grad[5]:
            grad_bias   = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight_r, grad_weight_i, grad_weight_j, grad_weight_k, grad_bias

def hamilton_product(q0, q1): #能不能用哈密顿积做QK的积
    """
    Applies a Hamilton product q0 * q1:
    Shape:
        - q0, q1 should be (batch_size, quaternion_number)
        (rr' - xx' - yy' - zz')  +
        (rx' + xr' + yz' - zy')i +
        (ry' - xz' + yr' + zx')j +
        (rz' + xy' - yx' + zr')k +
    """

    q1_r = get_r(q1)
    q1_i = get_i(q1)
    q1_j = get_j(q1)
    q1_k = get_k(q1)

    # rr', xx', yy', and zz'
    r_base = torch.mul(q0, q1)
    # (rr' - xx' - yy' - zz')
    r   = get_r(r_base) - get_i(r_base) - get_j(r_base) - get_k(r_base)

    # rx', xr', yz', and zy'
    i_base = torch.mul(q0, torch.cat([q1_i, q1_r, q1_k, q1_j], dim=1))
    # (rx' + xr' + yz' - zy')
    i   = get_r(i_base) + get_i(i_base) + get_j(i_base) - get_k(i_base)

    # ry', xz', yr', and zx'
    j_base = torch.mul(q0, torch.cat([q1_j, q1_k, q1_r, q1_i], dim=1))
    # (rx' + xr' + yz' - zy')
    j   = get_r(j_base) - get_i(j_base) + get_j(j_base) + get_k(j_base)

    # rz', xy', yx', and zr'
    k_base = torch.mul(q0, torch.cat([q1_k, q1_j, q1_i, q1_r], dim=1))
    # (rx' + xr' + yz' - zy')
    k   = get_r(k_base) + get_i(k_base) - get_j(k_base) + get_k(k_base)

    return torch.cat([r, i, j, k], dim=1)

#
# PARAMETERS INITIALIZATION
#

def unitary_init(in_features, out_features, rng, kernel_size=None, criterion='he'):

    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in          = in_features  * receptive_field
        fan_out         = out_features * receptive_field
    else:
        fan_in          = in_features
        fan_out         = out_features


    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    number_of_weights = np.prod(kernel_shape)
    v_r = np.random.uniform(-1.0,1.0,number_of_weights)
    v_i = np.random.uniform(-1.0,1.0,number_of_weights)
    v_j = np.random.uniform(-1.0,1.0,number_of_weights)
    v_k = np.random.uniform(-1.0,1.0,number_of_weights)

    # Unitary quaternion
    for i in range(0, number_of_weights):
        norm = np.sqrt(v_r[i]**2 + v_i[i]**2 + v_j[i]**2 + v_k[i]**2)+0.0001
        v_r[i]/= norm
        v_i[i]/= norm
        v_j[i]/= norm
        v_k[i]/= norm
    v_r = v_r.reshape(kernel_shape)
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    return (v_r, v_i, v_j, v_k)

def random_init(in_features, out_features, rng, kernel_size=None, criterion='glorot'):

    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in          = in_features  * receptive_field
        fan_out         = out_features * receptive_field
    else:
        fan_in          = in_features
        fan_out         = out_features

    if criterion == 'glorot':
        s = 1. / np.sqrt(2*(fan_in + fan_out))
    elif criterion == 'he':
        s = 1. / np.sqrt(2*fan_in)
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    number_of_weights = np.prod(kernel_shape)
    v_r = np.random.uniform(-1.0,1.0,number_of_weights)
    v_i = np.random.uniform(-1.0,1.0,number_of_weights)
    v_j = np.random.uniform(-1.0,1.0,number_of_weights)
    v_k = np.random.uniform(-1.0,1.0,number_of_weights)



    v_r = v_r.reshape(kernel_shape)
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    weight_r = v_r
    weight_i = v_i
    weight_j = v_j
    weight_k = v_k
    return (weight_r, weight_i, weight_j, weight_k)


def quaternion_init(in_features, out_features, rng, kernel_size=None, criterion='glorot'):

    if kernel_size is not None:
        receptive_field = np.prod(kernel_size) #27
        fan_in          = in_features  * receptive_field #27
        fan_out         = out_features * receptive_field #108
    else:
        fan_in          = in_features
        fan_out         = out_features

    if criterion == 'glorot':
        s = 1. / np.sqrt(2*(fan_in + fan_out))
    elif criterion == 'he':
        s = 1. / np.sqrt(2*fan_in)
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    rng = RandomState(np.random.randint(1,1234))

    # Generating randoms and purely imaginary quaternions :
    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    modulus = chi.rvs(4,loc=0,scale=s,size=kernel_shape)
    number_of_weights = np.prod(kernel_shape)
    v_i = np.random.uniform(-1.0,1.0,number_of_weights)
    v_j = np.random.uniform(-1.0,1.0,number_of_weights)
    v_k = np.random.uniform(-1.0,1.0,number_of_weights)

    # Purely imaginary quaternions unitary
    for i in range(0, number_of_weights):
        norm = np.sqrt(v_i[i]**2 + v_j[i]**2 + v_k[i]**2 +0.0001)
        v_i[i]/= norm
        v_j[i]/= norm
        v_k[i]/= norm
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)

    weight_r = modulus * np.cos(phase)
    weight_i = modulus * v_i*np.sin(phase)
    weight_j = modulus * v_j*np.sin(phase)
    weight_k = modulus * v_k*np.sin(phase)

    return (weight_r, weight_i, weight_j, weight_k)

def create_dropout_mask(dropout_p, size, rng, as_type, operation='linear'):
    if operation == 'linear':
        mask = rng.binomial(n=1, p=1-dropout_p, size=size)
        return Variable(torch.from_numpy(mask).type(as_type))
    else:
         raise Exception("create_dropout_mask accepts only 'linear'. Found operation = "
                        + str(operation))

def affect_init(r_weight, i_weight, j_weight, k_weight, init_func, rng, init_criterion):
    if r_weight.size() != i_weight.size() or r_weight.size() != j_weight.size() or \
    r_weight.size() != k_weight.size() :
         raise ValueError('The real and imaginary weights '
                 'should have the same size . Found: r:'
                 + str(r_weight.size()) +' i:'
                 + str(i_weight.size()) +' j:'
                 + str(j_weight.size()) +' k:'
                 + str(k_weight.size()))

    elif r_weight.dim() != 2:
        raise Exception('affect_init accepts only matrices. Found dimension = '
                        + str(r_weight.dim()))
    kernel_size = None
    r, i, j, k  = init_func(r_weight.size(0), r_weight.size(1), rng, kernel_size, init_criterion)
    r, i, j, k  = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j), torch.from_numpy(k)
    r_weight.data = r.type_as(r_weight.data)
    i_weight.data = i.type_as(i_weight.data)
    j_weight.data = j.type_as(j_weight.data)
    k_weight.data = k.type_as(k_weight.data)


def affect_init_conv(r_weight, i_weight, j_weight, k_weight, kernel_size, init_func, rng, init_criterion):
    if r_weight.size() != i_weight.size() or r_weight.size() != j_weight.size() or r_weight.size() != k_weight.size():
         raise ValueError('The real and imaginary weights should have the same size . Found: r:'\
                 + str(r_weight.size()) +' i:'
                 + str(i_weight.size()) +' j:'
                 + str(j_weight.size()) +' k:'
                 + str(k_weight.size()))

    elif 2 >= r_weight.dim():
        raise Exception('affect_conv_init accepts only tensors that have more than 2 dimensions. Found dimension = '
                        + str(r_weight.dim()))

    r, i, j, k = init_func(
        r_weight.size(1),
        r_weight.size(0),
        rng=rng,
        kernel_size=kernel_size,
        criterion=init_criterion
    )
    r, i, j, k = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j), torch.from_numpy(k)
    r_weight.data = r.type_as(r_weight.data)
    i_weight.data = i.type_as(i_weight.data)
    j_weight.data = j.type_as(j_weight.data)
    k_weight.data = k.type_as(k_weight.data)

def get_kernel_and_weight_shape(operation, in_channels, out_channels, kernel_size):
    if operation == 'convolution1d':
        if type(kernel_size) is not int:
            raise ValueError(
                """An invalid kernel_size was supplied for a 1d convolution. The kernel size
                must be integer in the case. Found kernel_size = """ + str(kernel_size)
            )
        else:
            ks = kernel_size
            w_shape = (out_channels, in_channels) + tuple((ks,))
    else:# in case it is 2d or 3d.
        if   operation == 'convolution2d' and type(kernel_size) is int:
            ks = (kernel_size, kernel_size)
        elif operation == 'convolution3d' and type(kernel_size) is int:
            ks = (kernel_size, kernel_size, kernel_size)
        elif type(kernel_size) is not int:
            if   operation == 'convolution2d' and len(kernel_size) != 2:
                raise ValueError(
                    """An invalid kernel_size was supplied for a 2d convolution. The kernel size
                    must be either an integer or a tuple of 2. Found kernel_size = """ + str(kernel_size)
                )
            elif operation == 'convolution3d' and len(kernel_size) != 3:
                raise ValueError(
                    """An invalid kernel_size was supplied for a 3d convolution. The kernel size
                    must be either an integer or a tuple of 3. Found kernel_size = """ + str(kernel_size)
                )
            else:
                ks = kernel_size
        w_shape = (out_channels, in_channels) + (*ks,)
    return ks, w_shape
