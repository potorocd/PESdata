# -*- coding: utf-8 -*-
"""
Created on Tue Aug 2 16:38:44 2022

author: Dr. Dmitrii Potorochin
email:  dmitrii.potorochin@desy.de
        dmitrii.potorochin@physik.tu-freiberg.de
        dm.potorochin@gmail.com
"""

# This section is supposed for importing necessary modules.
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import h5py
import json
import calendar
from types import SimpleNamespace
from time import gmtime
from datetime import datetime, timedelta
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
from matplotlib.widgets import Slider
from matplotlib.widgets import RangeSlider
import matplotlib
from scipy.signal import savgol_filter
import xarray as xr
from PIL import Image, ImageDraw, ImageFont
from lmfit.models import VoigtModel, ConstantModel
import matplotlib.colors as colors
import scipy
from scipy.special import erf
from lmfit import Model
from pandas import read_table
from pandas import read_parquet
from timeit import default_timer as timer
from re import findall
from yaml import safe_load as yaml_safe_load
try:
    from julia.api import Julia
except:
    pass
try:
    from igor2.binarywave import load as loadibw
except:
    pass

# Dictionary for colors
color_dict = {
  0: 'blue',
  1: 'tab:red',
  2: 'black',
  3: 'tab:orange',
  4: 'tab:green',
  5: 'deeppink',
  6: 'tab:cyan',
  7: 'magenta',
  8: 'yellow'
}


class CasaXPS_file:
    def __init__(self, file_full, energy_axis='BE'):
        self.type = 'fit'
        scan_name = file_full.split("\\")[-1]
        scan_name = scan_name.replace('.' + scan_name.split('.')[-1], '')
        self.scan_name = scan_name
        self.energy_axis = energy_axis
        self.bg_subtr = 'off'
        self.data = read_table(file_full, delimiter='\t',
                               skiprows=3, header=0)
        self.data = self.data.T
        self.array = self.data.to_numpy()
        self.data_y = self.array[2:]
        if self.energy_axis == 'KE':
            self.data_x = self.array[0]
        if self.energy_axis == 'BE':
            self.data_x = self.array[1]

        self.data_label = list(self.data.index)[2:]
        self.CPS = [list(range(len(self.data_label)))[0]]
        self.comps = list(range(len(self.data_label)))[1:-2]
        self.bg = [list(range(len(self.data_label)))[-2]]
        self.env = [list(range(len(self.data_label)))[-1]]
        self.data_label = list(self.data.index)[2:]

        with open(file_full, 'r') as fp:
            _number = fp.readlines()[1].count('_')
        data_label_i = []
        for counter, i in enumerate(self.data_label):
            line = i.split('_')
            if counter in self.comps:
                del line[-2-_number:]
                line = '_'.join(line)
            else:
                del line[-1-_number:]
                line = '_'.join(line)
            data_label_i.append(line)
        self.data_label = data_label_i

        with open(file_full, 'r') as fp:
            hv = fp.readlines()[2].split('\t')[2]
            try:
                self.hv = int(hv)
            except ValueError:
                self.hv = round(float(hv), 1)

        if 300 > np.mean(self.data_x) > 280:
            self.core_level = 'C 1s'
        elif 410 > np.mean(self.data_x) > 390:
            self.core_level = 'N 1s'
        elif 540 > np.mean(self.data_x) > 520:
            self.core_level = 'O 1s'
        elif 110 > np.mean(self.data_x) > 90:
            self.core_level = 'Si 2p'
        elif 740 > np.mean(self.data_x) > 700:
            self.core_level = 'Fe 2p'
        elif 210 > np.mean(self.data_x) > 190:
            self.core_level = 'Cl 2p'
        elif 65 > np.mean(self.data_x) > 50:
            self.core_level = 'Fe 3p'
        elif 30 > np.mean(self.data_x) > 0:
            self.core_level = 'VB'
        elif self.energy_axis == 'KE':
            self.core_level = 'Not defined'
        else:
            self.core_level = 'Not defined'

    def exclude_comp(self, exclude_list):
        comp_bin = []
        for i in exclude_list:
            try:
                comp_bin.append(self.comps[i-1])
            except IndexError:
                pass
        for j in comp_bin:
            self.comps.remove(j)

    def bg_sub(self):
        data_y_plot = list(self.data_y)
        for i in range(len(data_y_plot)):
            data_y_plot[i] = data_y_plot[i] - self.data_y[self.bg[0]]
        self.data_y = np.array(data_y_plot)
        self.bg_subtr = 'on'
        return self


def scan_hdf5(hdf5_obj, hdf5_path=[]):
    '''
    This function helps to adapt to changing structure
    of hdf5 files from WESPE.
    '''
    if type(hdf5_obj) in [h5py._hl.group.Group, h5py._hl.files.File]:
        for counter, key in enumerate(hdf5_obj.keys()):
            scan_hdf5(hdf5_obj[key])
    elif type(hdf5_obj) == h5py._hl.dataset.Dataset:
        full_path = hdf5_obj.name
        dataset_name = full_path.replace(hdf5_obj.parent.name, '')
        dataset_name = dataset_name.replace('/', '')
        if dataset_name == 'energy_Grid_ROI':
            hdf5_path.append(hdf5_obj.parent.name)
    return hdf5_path


def text_phantom(text, size):
    '''
    This function helps to create a dummy image with an error message
    if something goes wrong with data handling.
    '''
    # Availability is platform dependent
    font = 'arial'

    # Create font
    pil_font = ImageFont.truetype(font + ".ttf", size=size // (len(text)//4),
                                  encoding="unic")
    text_width, text_height = pil_font.getsize(text)

    # create a blank canvas with extra space between lines
    canvas = Image.new('RGB', [size, size//2], (0, 0, 0))

    # draw the text onto the canvas
    draw = ImageDraw.Draw(canvas)
    offset = ((size - text_width//2) // 2,
              (size//2 - text_height) // 2)
    white = "#FFFFFF"
    draw.text(offset, text, font=pil_font, fill=white)

    # Convert the canvas into an array with values in [0, 1]
    return (255 - np.asarray(canvas)) / 255.0


def exponential_decay(x, tzero, sigma, tconst):
    return 0.5*np.exp(-(x-tzero)/tconst)*np.exp((sigma/(2*np.sqrt(np.log(2))))**2/4/tconst**2)*(1+erf((x-tzero)/(sigma/(2*np.sqrt(np.log(2))))-(sigma/(2*np.sqrt(np.log(2))))/2/tconst))


def biexponential_decay(x, tzero, sigma,
                        tconst1, tconst2, amp1, amp2, offset):
    comp_1 = amp1*exponential_decay(x, tzero, sigma, tconst1)
    comp_2 = amp2*exponential_decay(x, tzero, sigma, tconst2)
    return comp_1 + comp_2 + offset


class create_batch_WESPE:
    '''
    The object for storing data of combined runs.
    '''

    def __init__(self, file_dir, run_list, DLD='DLD4Q'):
        '''
        This initialization happens on 'Upload runs'.
        '''
        self.file_dir = file_dir
        try:
            self.batch_dir, self.batch_list = [], []
            for run_number in run_list:
                file_name = f'{run_number}' + os.sep + f'{run_number}_energy.mat'
                file_full = file_dir + os.sep + file_name
                self.batch_list.append(read_file_WESPE(file_full, DLD=DLD))
                self.batch_dir.append(file_full)
        except:
            self.batch_dir, self.batch_list = [], []
            for run_number in run_list:
                file_full = file_dir + os.sep + f'{run_number}'
                if os.path.isfile(file_full+'.4Q') is False and os.path.isfile(file_full+'s.4Q'):
                    file_name = f'{run_number}' + os.sep + f'{run_number}'
                    file_full = file_dir + os.sep + file_name
                self.batch_list.append(read_file_WESPE(file_full, DLD=DLD))
                self.batch_dir.append(file_full)

        full_info = []
        for i in self.batch_list:
            full_info.append(i.info)
        self.full_info = 'DETAILED INFO:\n\n' + '\n\n'.join(full_info)

        title = 'SHORT SUMMARY:\n'
        run_num, is_static, KE, mono = [], [], [], []
        for i in self.batch_list:
            run_num.append(str(i.run_num))
            is_static.append(i.is_static)
            KE.append(i.KE)
            mono.append(i.mono_mean)
        # Run numbers
        self.run_num = ', '.join(run_num)
        run_num = [int(i) for i in run_num]
        run_num.sort()
        if len(run_num) == 1:
            run_num = f'Uploaded run: {run_num[0]}'
        elif len(run_num) > 6:
            run_num = f'Uploaded runs: {np.min(run_num)}-{np.max(run_num)}'
        else:
            run_num = [str(i) for i in run_num]
            run_num = ', '.join(run_num)
            run_num = 'Uploaded runs: ' + run_num
        self.run_num_o = run_num.replace('Uploaded runs: ', '')
        # Static scan check
        if all(is_static):
            is_static_s = 'Static check: All runs are static (+)'
        elif not any(is_static):
            is_static_s = 'Static check: All runs are delay scans (+)'
        else:
            is_static_s = 'Static check: Delay scans are mixed with static scans (!!!)'
        # Energy region check
        if np.max(KE) - np.min(KE) > 5:
            KE_s = 'Region check: Various energy regions are on the list (!!!)'
        else:
            KE_s = 'Region check: Homogeneous energy regions (+)'
        # Mono check
        if np.max(mono) - np.min(mono) > 0.15:
            mono_s = 'Mono check: Various mono values for different runs (!!!)'
        else:
            mono_s = 'Mono check: No mono energy jumps detected (+)'
        self.en_threshold = np.max(mono) + 50
        if self.en_threshold < 50:
            self.en_threshold = 1000
        static_cut_list = []
        for counter, i in enumerate(is_static):
            if i is True:
                static_cut = np.mean(self.batch_list[counter].DLD_delay)
                static_cut_list.append(static_cut)
        self.static_cut_list = static_cut_list
        short_info = [title, run_num, is_static_s, KE_s, mono_s]
        self.short_info = '\n'.join(short_info) + '\n\n'

    def time_zero(self, t0=1328.2):
        '''
        Method for creating new array coordinate 'Delay relative t0'
        after specification of the delay stage value considered as time zero.
        '''
        check = []
        for i in ['x', 'y', 'z']:
            try:
                if self.Map_2D_plot.attrs[f'{i}_units'] == 'ps':
                    check.append(i)
            except:
                pass
        try:
            check = check[0]
            self.t0 = read_file_WESPE.rounding(t0, getattr(self, f'{check}_step'))
            label_a = self.Map_2D_plot.attrs[f'{check}_label_a']
            label = self.Map_2D_plot.attrs[f'{check}_label']
            image_data_a = self.t0 - self.Map_2D.coords[label].values
            self.Map_2D.coords[label_a] = (f'Dim_{check}', image_data_a)
            self.Map_2D_plot.coords[label_a] = (f'Dim_{check}', image_data_a)
            self.set_T0()
        except:
            pass
        self.set_T0()

    def create_map(self):
        '''
        This method sums delay-energy maps of individual runs
        uploaded to the batch.
        '''
        self.energy_step = self.batch_list[0].energy_step
        self.delay_step = self.batch_list[0].delay_step
        self.ordinate = self.batch_list[0].ordinate
        attrs = self.batch_list[0].Map_2D.attrs
        for counter, i in enumerate(self.batch_list):
            if counter == 0:
                total_map = i.Map_2D
            else:
                total_map = total_map + i.Map_2D
        total_map.attrs = attrs
        try:
            total_map.coords['Binding energy']
            total_map.coords['Kinetic energy']
        except KeyError:
            total_map = xr.DataArray([])
        if np.min(total_map.values.shape) == 0:
            concat_list = []
            for counter, i in enumerate(self.batch_list):
                concat_list.append(i.Map_2D)
            total_map = xr.combine_by_coords(concat_list, compat='override')
            total_map.coords['Delay stage values'] = total_map.coords['Dim_y']
            total_map = total_map.to_array(dim='variable', name=None)
            total_map = total_map.sum(dim='variable')
            total_map.attrs = attrs

        if np.min(total_map.values.shape) == 0:
            total_map.attrs['Merge successful'] = False

        # Filter empty EDCs
        y_check = total_map.sum('Dim_x', skipna=True)
        y_check = y_check/total_map.coords['Dim_x'].shape[0]
        remove_list = np.where(y_check < 1)
        remove_list = y_check.coords['Dim_y'][remove_list]
        for i in remove_list:
            total_map = total_map.where(total_map['Dim_y'] != i, drop=True)

        shape = total_map.coords['Dim_y'].values.shape[0]
        total_map.coords['Delay index'] = ('Dim_y', np.arange(shape))
        self.Map_2D = total_map.fillna(0)
        self.Map_2D = self.Map_2D.where(self.Map_2D.coords['Kinetic energy'].notnull(), drop=True)
        if np.median(np.gradient(self.Map_2D.coords['Binding energy'].values)) > 0:
            self.Map_2D=self.Map_2D.isel(Dim_x=slice(None, None, -1))
        self.Map_2D_plot = self.Map_2D

        self.varied_y_step = False
        try:
            self.y_step = self.Map_2D.coords['Dim_y'].values
            self.y_step = np.min(np.abs(np.gradient(self.y_step)))
            if np.around(np.std(np.gradient(total_map.coords['Dim_y'])), 3) > 0:
                self.varied_y_step = True
        except:
            self.y_step = 1
        self.x_step = self.Map_2D.coords['Dim_x'].values
        self.x_step = np.min(np.abs(np.gradient(self.x_step)))
        self.x_step = np.around(self.x_step, 2)
        self.y_step = np.around(self.y_step, 2)

    def create_dif_map(self):
        '''
        This method generates a difference map by averaging data before
        -0.25 ps and subtracting it from the delay-energy map.
        '''
        attrs = self.Map_2D_plot.attrs
        t_axis_step = self.Map_2D_plot.coords['Dim_y'].values
        try:
            t_axis_step = abs(np.median(np.gradient(t_axis_step)))
        except ValueError:
            t_axis_step = 1
        t_axis_step = int(-2.5*t_axis_step)
        norm = self.Map_2D_plot.loc[t_axis_step:].mean('Dim_y')
        self.Map_2D_dif = self.Map_2D_plot - norm
        self.Map_2D_dif.attrs = attrs

    def set_BE(self):
        '''
        Method for switching visualization to 'Binding energy'
        coordinate of 'Energy' dimension.
        '''
        if self.Map_2D_plot.attrs['x_alt'] is False:
            coord = self.Map_2D_plot.coords[self.Map_2D_plot.attrs['x_label_a']]
            self.Map_2D_plot.coords['Dim_x'] = coord
            self.Map_2D_plot.attrs['x_alt'] = True

    def set_KE(self):
        '''
        Method for switching visualization to 'Kinetic energy'
        coordinate of 'Energy' dimension.
        '''
        if self.Map_2D_plot.attrs['x_alt'] is True:
            coord = self.Map_2D_plot.coords[self.Map_2D_plot.attrs['x_label']]
            self.Map_2D_plot.coords['Dim_x'] = coord
            self.Map_2D_plot.attrs['x_alt'] = False

    def set_T0(self):
        '''
        Method for switching visualization to 'Delay relative t0'
        coordinate of 'Dim_y' dimension.
        '''
        check = []
        for i in ['x', 'y', 'z']:
            try:
                if self.Map_2D_plot.attrs[f'{i}_units'] == 'ps':
                    check.append(i)
            except:
                pass
        try:
            check = check[0]
            if self.Map_2D_plot.attrs[f'{check}_alt'] is False:
                coord = self.Map_2D_plot.coords[self.Map_2D_plot.attrs[f'{check}_label_a']]
                self.Map_2D_plot.coords[f'Dim_{check}'] = coord
                self.Map_2D_plot.attrs[f'{check}_alt'] = True
        except:
            pass
            
    def set_Tds(self):
        '''
        Method for switching visualization to 'Delay stage values'
        coordinate of 'Dim_y' dimension.
        '''
        check = []
        for i in ['x', 'y', 'z']:
            try:
                if self.Map_2D_plot.attrs[f'{i}_units'] == 'ps':
                    check.append(i)
            except:
                pass
        try:
            check = check[0]
            if self.Map_2D_plot.attrs[f'{check}_alt'] is True:
                coord = self.Map_2D_plot.coords[self.Map_2D_plot.attrs[f'{check}_label']]
                self.Map_2D_plot.coords[f'Dim_{check}'] = coord
                self.Map_2D_plot.attrs[f'{check}_alt'] = False
        except:
            pass

    def set_dif_map(self):
        '''
        Method for switching visualization to the difference plot.
        '''
        self.Map_2D_plot = self.Map_2D_dif

    def ROI(self, limits, axis, mod_map=True):
        '''
        Method for selecting the range of values of interest
        on delay-energy map.
        limits - a list for determination of a minimum and
        a maximum of the range
        axis - selection of Time or Dim_x
        mod_map - if True, changes the array used for visualization
        returns a new array after cutting out undesired regions
        '''
        min_val = np.min(limits)
        max_val = np.max(limits)
        if axis == 'Dim_y':
            check = 'y_order_rec'
            if self.Map_2D_plot.attrs['y_alt'] is True:
                check += '_a'
            if 'energy' in self.Map_2D_plot.attrs['y_label']:
                if self.Map_2D_plot.attrs[check] is False:
                    new_a = self.Map_2D_plot.loc[max_val:min_val]
                else:
                    new_a = self.Map_2D_plot.loc[min_val:max_val]
            else:
                if self.Map_2D_plot.attrs[check] is False:
                    new_a = self.Map_2D_plot.loc[min_val:max_val]
                else:
                    new_a = self.Map_2D_plot.loc[max_val:min_val]
        if axis == 'Dim_x':
            check = 'x_order_rec'
            if self.Map_2D_plot.attrs['x_alt'] is True:
                check += '_a'
            if self.Map_2D_plot.attrs[check] is False:
                new_a = self.Map_2D_plot.loc[:, min_val:max_val]
            else:
                new_a = self.Map_2D_plot.loc[:, max_val:min_val]
        if mod_map is True:
            self.Map_2D_plot = new_a
        else:
            return new_a

    def norm_total_e(self):
        '''
        Method for normalization of delay-energy map in terms of the concept
        that every time delay line should contain the same number of detected
        electrons, i.e., we have only redistribution of electrons in the
        energy domain.
        '''
        arr = self.Map_2D_plot
        attrs = self.Map_2D_plot.attrs

        norm = arr.sum('Dim_x', skipna=True)
        new_arr = arr/norm * norm.mean('Dim_y')

        self.Map_2D_plot = new_arr
        self.Map_2D_plot.attrs = attrs
        self.Map_2D_plot.attrs['Normalized'] = True

    def norm_01(self):
        '''
        Method for normalization of delay-energy map to zero to one intensity.
        '''
        arr = self.Map_2D_plot
        attrs = self.Map_2D_plot.attrs

        norm = arr.min('Dim_x', skipna=True)
        # norm = norm.min('Dim_y', skipna=True)
        new_arr = arr - norm
        norm = new_arr.max('Dim_x', skipna=True)
        # norm = norm.max('Dim_y', skipna=True)
        new_arr = new_arr/norm

        self.Map_2D_plot = new_arr
        self.Map_2D_plot.attrs = attrs
        self.Map_2D_plot.attrs['Normalized'] = True

    def norm_11(self):
        '''
        Method for normalization of delay-energy map to minus one to one
        intensity range. Either high or low limit absolute value is one.
        The other limit is scaled accordingly.
        It suits well for the difference plot.
        '''
        arr = self.Map_2D_plot
        attrs = self.Map_2D_plot.attrs

        pos_norm = arr.max('Dim_x', skipna=True)
        pos_norm = pos_norm.max('Dim_y', skipna=True)
        neg_norm = arr.min('Dim_x', skipna=True)
        neg_norm = neg_norm.min('Dim_y', skipna=True)
        norm = xr.concat((pos_norm, neg_norm), dim='New')
        norm = np.abs(norm)
        norm = norm.max('New', skipna=True)
        new_arr = arr/norm

        self.Map_2D_plot = new_arr
        self.Map_2D_plot.attrs = attrs
        self.Map_2D_plot.attrs['Normalized'] = True

    def t0_cut(self, position='Main', hv=2.407, axis='Dim_x'):
        '''
        Method for simplification of finding the position of the most
        prominent feature or sidebands for the t0 finder.
        '''
        position_list = str(position)
        position_list = position_list.split(',')
        position_list = [i.strip().lower() for i in position_list]
        position = position_list[0]

        array = self.Map_2D_plot
        if axis == 'Dim_x':
            array_mean = array.median('Dim_y')
            pos_e = array_mean.idxmax('Dim_x').values
        else:
            array_mean = array.median('Dim_x')
            pos_e = array_mean.idxmax('Dim_y').values

        if position == 'sb':
            try:
                hv = position_list[1]
                hv = float(hv)
                pos_e += hv
            except IndexError:
                pos_e += hv

        try:
            pos_e = float(position)
        except ValueError:
            pass

        pos_e = np.around(pos_e, 2)
        return pos_e

    def save_map_dat(self):
        '''
        Method for saving the delay-energy map from visualization
        to ASCII format.
        One can find the saved result in the 'ASCII_output' folder.
        '''
        arr = self.Map_2D_plot
        length = arr.shape[0]
        ts = calendar.timegm(gmtime())
        date_time = datetime.fromtimestamp(ts)
        str_date_time = date_time.strftime("%d.%m.%Y_%H-%M-%S")
        path = self.file_dir + os.sep + 'ASCII_output'
        if os.path.isdir(path) is False:
            os.mkdir(path)
        path = path + os.sep + 'Maps'
        if os.path.isdir(path) is False:
            os.mkdir(path)
        path = path + os.sep + str_date_time + os.sep
        if os.path.isdir(path) is False:
            os.mkdir(path)

        if arr.attrs['x_alt'] is True:
            x_label = arr.attrs['x_label_a']
            x_units = arr.attrs['x_units_a']
            x_order = arr.attrs['x_order_rec_a']
        else:
            x_label = arr.attrs['x_label']
            x_units = arr.attrs['x_units']
            x_order = arr.attrs['x_order_rec']

        if arr.attrs['y_alt'] is True:
            y_label = arr.attrs['y_label_a']
            y_units = arr.attrs['y_units_a']
            y_order = arr.attrs['y_order_rec_a']
        else:
            y_label = arr.attrs['y_label']
            y_units = arr.attrs['y_units']
            y_order = arr.attrs['y_order_rec']

        with open(path+"Summary.txt", "w") as text_file:
            text_file.write(f'Loaded runs: {self.run_num_o}\n')
            text_file.write(f'Dim_x step: {self.x_step} {x_units}\n')
            text_file.write(f'Dim_y step: {self.y_step} {y_units}\n')
            text_file.write(f'Dim_x: {x_label} (column 1)\n')
            text_file.write(f'Dim_y: {y_label} (file name)\n')
            try:
                text_file.write(f'Time zero: {self.t0} {y_units}\n')
            except:
                text_file.write('Time zero: None\n')
            if arr.attrs['Normalized'] is True:
                text_file.write('Normalized: True\n')
            else:
                text_file.write('Normalized: False\n')
        for i in range(length):
            x = arr.coords['Dim_x'].values
            x = list(x)
            x = [read_file_WESPE.rounding(i, self.x_step) for i in x]
            x = np.array(x)
            x = np.expand_dims(x, axis=0)
            y = arr.isel(Dim_y=i).values
            y = np.expand_dims(y, axis=0)
            out = np.append(x, y, axis=0)
            out = np.rot90(out)

            file_full = path
            delay_val = arr.coords['Dim_y'].values[i]
            delay_val = np.around(delay_val, 2)
            if y_order is True:
                order = length - 1 - i
            else:
                order = i
            if len(str(order)) == len(str(length)):
                order = str(order)
            else:
                order = str(order)
                for j in range(len(str(length))-len(str(order))):
                    order = '0' + order
            file_full = file_full + f'{order}_{delay_val} {y_units}.dat'
            np.savetxt(file_full, out, delimiter='    ')
            print(f"Saved as {file_full}")

    def axs_plot(self, axs, dif_3D=False):
        # Loading configs from json file.
        try:
            with open('config.json', 'r') as json_file:
                config = json.load(json_file)
        except FileNotFoundError:
            with open('packages/config.json', 'r') as json_file:
                config = json.load(json_file)
        config = json.dumps(config)
        config = json.loads(config,
                            object_hook=lambda d: SimpleNamespace(**d))
        '''
        Method for creating matplotlib axes for delay-energy map visualization.
        '''
        if np.min(self.Map_2D_plot.values.shape) == 0:
            if self.Map_2D_plot.attrs['Merge successful'] is False:
                label = ['Merge was not successful. You can:',
                         '1) Change delay/energy step;',
                         '2) Switch to different axis (e.g., KE/BE).']
                label = '\n'.join(label)
                im1 = axs.imshow(text_phantom(label, 1000))
                axs.axis('off')
            else:
                label = ['The delay-energy map is empty!',
                         '1) Check ROI values;',
                         '2) Check bunch filtering.']
                label = '\n'.join(label)
                im1 = axs.imshow(text_phantom(label, 1000))
                axs.axis('off')
        else:
            image_data = self.Map_2D_plot.values
            if image_data.ndim == 3:
                switch_3D = True
            else:
                switch_3D = False
            image_data_y = self.Map_2D_plot.coords['Dim_y'].values
            image_data_x = self.Map_2D_plot.coords['Dim_x'].values
            order_x = 'x_order_rec'
            order_y = 'y_order_rec'
            if self.Map_2D_plot.attrs['x_alt'] is True:
                order_x += '_a'
            if self.Map_2D_plot.attrs['y_alt'] is True:
                order_y += '_a'

            if image_data.shape[0] == 1:
                image_data = np.pad(image_data, [(1, 1), (0, 0)],
                                    mode='constant')
                image_data_y = [image_data_y[0]-1,
                                image_data_y[0],
                                image_data_y[0]+1]
                image_data_y = np.array(image_data_y)
            self.varied_y_step = False
            if image_data_y.shape[0] > 1:
                if np.around(np.std(np.gradient(image_data_y)), 3) > 0:
                    self.varied_y_step = True
                    image_data_y = np.arange(image_data_y.shape[0])
                    self.image_data_y = image_data_y
                    if self.Map_2D_plot.attrs[order_y] == False:
                        pos_list = np.linspace(np.min(image_data_y),
                                               np.max(image_data_y),
                                               config.map_n_ticks_y,
                                               dtype=int)
                    else:
                        pos_list = np.linspace(np.max(image_data_y),
                                               np.min(image_data_y),
                                               config.map_n_ticks_y,
                                               dtype=int)
                    label_list = self.Map_2D_plot.coords['Dim_y']
                    label_list = label_list[pos_list].values
                    try:
                        decimals = self.decimal_n(self.delay_step)
                    except:
                        decimals = self.decimal_n(self.y_step)
                    label_list = np.around(label_list, decimals)
            self.map_z_max = np.nanmax(image_data)
            self.map_z_min = np.nanmin(image_data)
            self.map_z_tick = (self.map_z_max - self.map_z_min)/config.map_n_ticks_z
            if self.map_z_tick < 1:
                self.map_z_tick_decimal = 1
            else:
                self.map_z_tick_decimal = 0
            self.map_z_tick = round(self.map_z_tick, self.map_z_tick_decimal)
            if self.map_z_tick == 0:
                self.map_z_tick = 1

            self.map_y_max = np.nanmax(image_data_y)
            self.map_y_min = np.nanmin(image_data_y)
            self.map_y_tick = (self.map_y_max - self.map_y_min)/config.map_n_ticks_y
            if self.map_y_max - self.map_y_min > 10:
                self.map_y_tick = math.ceil(self.map_y_tick)
                if self.map_y_tick == 0:
                    self.map_y_tick = 1
            else:
                for option in [1, 0.5, 0.2, 0.1, 0.05, 0.01, 0.001, 0.0001]:
                    if self.rounding(self.map_y_tick, option) > 0:
                        self.map_y_tick = self.rounding(self.map_y_tick,
                                                        option)
                        self.map_y_tick_decimal = self.decimal_n(option)
                        self.map_y_tick = round(self.map_y_tick,
                                                self.map_y_tick_decimal)
                        break
                    else:
                        if option == 0.0001:
                            self.map_y_tick = 0.0001

            self.map_x_max = np.nanmax(image_data_x)
            self.map_x_min = np.nanmin(image_data_x)
            self.map_x_tick = (self.map_x_max - self.map_x_min)/config.map_n_ticks_x
            if self.map_x_max - self.map_x_min > 5:
                self.map_x_tick = math.ceil(self.map_x_tick)
                if self.map_x_tick == 0:
                    self.map_x_tick = 1
            else:
                for option in [1, 0.5, 0.2, 0.1, 0.05, 0.01, 0.001, 0.0001]:
                    if self.rounding(self.map_x_tick, option) > 0:
                        self.map_x_tick = self.rounding(self.map_x_tick,
                                                        option)
                        self.map_x_tick_decimal = self.decimal_n(option)
                        self.map_x_tick = round(self.map_x_tick,
                                                self.map_x_tick_decimal)
                        break
                    else:
                        if option == 0.0001:
                            self.map_x_tick = 0.0001

            if self.Map_2D_plot.attrs[order_x] is False:
                x_start = np.min(image_data_x)
                x_end = np.max(image_data_x)
            else:
                x_start = np.max(image_data_x)
                x_end = np.min(image_data_x)

            if self.Map_2D_plot.attrs[order_y] is False:
                y_start = np.max(image_data_y)
                y_end = np.min(image_data_y)
            elif self.varied_y_step is True:
                y_start = np.max(image_data_y)
                y_end = np.min(image_data_y)
            else:
                y_start = np.min(image_data_y)
                y_end = np.max(image_data_y)

            if 'energy' in self.Map_2D_plot.attrs['y_label']:
                y_start, y_end = y_end, y_start

            extent = [x_start, x_end,
                      y_start, y_end]

            TwoSlopeNorm = config.TwoSlopeNorm
            if config.interpolation == 'on':
                interpolation = 'gaussian'
            else:
                interpolation = None
            # if image_data.ndim == 3:
            #     i_max = np.sum(image_data,axis=(0,1)).argmax()
            #     image_data = image_data[:,:,i_max]
                
            if switch_3D is False:
                vmin = np.min(image_data)
                try:
                    vmax = np.max(image_data[np.where(image_data<np.mean(image_data)*1000)])*config.map_scale
                except:
                    vmax = np.max(image_data)
                self.map_z_tick = self.map_z_tick*config.map_scale
                if vmin < 0:
                    vmin = vmin*config.map_scale

                if TwoSlopeNorm < 1 and TwoSlopeNorm > 0:
                    im1 = axs.imshow(image_data, origin='upper',
                                     interpolation=interpolation,
                                     extent=extent,
                                     cmap=config.cmap, aspect='auto',
                                     norm=colors.TwoSlopeNorm(vmin=vmin,
                                                              vcenter=TwoSlopeNorm*vmax,
                                                              vmax=vmax))
                else:
                    im1 = axs.imshow(image_data, origin='upper',
                                     interpolation=interpolation,
                                     extent=extent,
                                     vmin=vmin,
                                     vmax=vmax,
                                     cmap=config.cmap, aspect='auto')
                cbar_pad = 0.09
            else:
                if dif_3D is True:
                    self.Map_3D = self.Map_2D_plot
                    cbar_pad = '2%'
                    I_curve = np.sum(self.Map_3D, axis=(0, 1))
                    length = I_curve.shape[0]
                    i_max = I_curve.argmax()
                    if i_max < length*0.05 or i_max > length*0.95:
                        i_max = int(length*0.5)
                    self.image_data_z = self.Map_2D_plot.coords['Dim_z'].values
                    i_init_1 = (int(0.7*length), int(0.9*length))
                    i_init_2 = (int(0.1*length), int(0.3*length))
                    valinit_1 = (self.image_data_z[i_init_1[0]], self.image_data_z[i_init_1[1]])
                    valinit_2 = (self.image_data_z[i_init_2[0]], self.image_data_z[i_init_2[1]])
                    val_min = self.image_data_z.min()
                    val_max = self.image_data_z.max()
                    selection_1 = slice(i_init_1[0], i_init_1[1])
                    selection_2 = slice(i_init_2[0], i_init_2[1])
                    slice_1 = self.Map_3D[:, :, selection_1].mean(dim='Dim_z')
                    slice_2 = self.Map_3D[:, :, selection_2].mean(dim='Dim_z')
                    self.Map_2D_plot = slice_2 - slice_1
                    image_data = self.Map_2D_plot.values
                    self.Map_2D_plot.attrs = self.Map_3D.attrs

                    vmin = np.min(image_data)
                    # vmax = np.max(image_data[np.where(image_data<np.mean(image_data)*1000)])*config.map_scale
                    vmax = np.max(image_data)
                    self.map_z_tick = self.map_z_tick*config.map_scale
                    if vmin < 0:
                        vmin = vmin*config.map_scale

                    if TwoSlopeNorm < 1 and TwoSlopeNorm > 0:
                        im1 = axs.imshow(image_data, origin='upper',
                                         interpolation=interpolation,
                                         extent=extent,
                                         cmap=config.cmap, aspect='auto',
                                         norm=colors.TwoSlopeNorm(vmin=vmin,
                                                                  vcenter=TwoSlopeNorm*vmax,
                                                                  vmax=vmax))
                    else:
                        im1 = axs.imshow(image_data, origin='upper',
                                         interpolation=interpolation,
                                         extent=extent,
                                         vmin=vmin,
                                         vmax=vmax,
                                         cmap=config.cmap, aspect='auto')
    
                    divider = make_axes_locatable(axs)
                    cax2 = divider.append_axes("right", size="5%", pad='5%')
                    self.t_slider_1 = RangeSlider(ax=cax2,
                                                  label='s1',
                                                  track_color='dimgrey',
                                                  valmin=val_min,
                                                  valmax=val_max,
                                                  valinit=valinit_1,
                                                  orientation="vertical",
                                                  valstep=self.image_data_z,
                                                  handle_style={'facecolor': 'white',
                                                                'edgecolor': '.9',
                                                                'size': '20'
                                                               }
                                                  )
    
                    cax3 = divider.append_axes("right", size="5%", pad='5%')
                    self.t_slider_2 = RangeSlider(ax=cax3,
                                                  label='s2',
                                                  track_color='dimgrey',
                                                  valmin=val_min,
                                                  valmax=val_max,
                                                  valinit=valinit_2,
                                                  orientation="vertical",
                                                  valstep=self.image_data_z,
                                                  handle_style={'facecolor': 'white',
                                                                'edgecolor': '.9',
                                                                'size': '20'
                                                               }
                                                  )
    
                    def update(val):
                        pos_1 = self.t_slider_1.val
                        pos_2 = self.t_slider_2.val
                        selection_1 = np.where((self.image_data_z>=pos_1[0]) & (self.image_data_z<=pos_1[1]))[0]
                        selection_2 = np.where((self.image_data_z>=pos_2[0]) & (self.image_data_z<=pos_2[1]))[0]
                        slice_1 = self.Map_3D[:,:,selection_1].mean(dim='Dim_z')
                        slice_2 = self.Map_3D[:,:,selection_2].mean(dim='Dim_z')
                        self.Map_2D_plot = slice_2 - slice_1
                        self.Map_2D_plot.attrs = self.Map_3D.attrs
                        image_data = self.Map_2D_plot.values
                        vmin = np.min(image_data)
                        # vmax = np.max(image_data[np.where(image_data<np.mean(image_data)*1000)])*config.map_scale
                        vmax = np.max(image_data)
                        im1.set_data(image_data)
    
                        self.map_z_tick = (vmax - vmin)/config.map_n_ticks_z
                        if self.map_z_tick < 1:
                            self.map_z_tick_decimal = 1
                        else:
                            self.map_z_tick_decimal = 0
                        self.map_z_tick = round(self.map_z_tick, self.map_z_tick_decimal)
                        if self.map_z_tick == 0:
                            self.map_z_tick = 1
    
                        if TwoSlopeNorm < 1 and TwoSlopeNorm > 0:
                            im1.set_clim(vmin=vmin, vmax=vmax,
                                         vcenter=TwoSlopeNorm*vmax)
                        else:
                            im1.set_clim(vmin=vmin, vmax=vmax)
                        self.cbar.set_ticks(MultipleLocator(self.map_z_tick))
                        # plt.gcf().canvas.draw_idle()
                        # plt.gcf().canvas.draw()
                        # plt.gcf().canvas.flush_events()
                        return self.t_slider_1, self.t_slider_1
    
                    self.t_slider_1.on_changed(update)
                    self.t_slider_2.on_changed(update)
                else:
                    self.Map_3D = self.Map_2D_plot
                    cbar_pad = '2%'
                    I_curve = np.sum(self.Map_3D, axis=(0, 1))
                    length = I_curve.shape[0]
                    i_max = I_curve.argmax()
                    if i_max < length*0.05 or i_max > length*0.95:
                        i_max = int(length*0.5)
                    self.image_data_z = self.Map_2D_plot.coords['Dim_z'].values
                    valinit = self.image_data_z[i_max]
                    val_min = self.image_data_z.min()
                    val_max = self.image_data_z.max()
                    self.Map_2D_plot = self.Map_3D[:, :, i_max]
                    image_data = self.Map_2D_plot.values
    
                    vmin = np.min(image_data)
                    try:
                        vmax = np.max(image_data[np.where(image_data<np.mean(image_data)*1000)])*config.map_scale
                    except:
                        vmax = np.max(image_data)
                    self.map_z_tick = self.map_z_tick*config.map_scale
                    if vmin < 0:
                        vmin = vmin*config.map_scale
    
                    if TwoSlopeNorm < 1 and TwoSlopeNorm > 0:
                        im1 = axs.imshow(image_data, origin='upper',
                                         interpolation=interpolation,
                                         extent=extent,
                                         cmap=config.cmap, aspect='auto',
                                         norm=colors.TwoSlopeNorm(vmin=vmin,
                                                                  vcenter=TwoSlopeNorm*vmax,
                                                                  vmax=vmax))
                    else:
                        im1 = axs.imshow(image_data, origin='upper',
                                         interpolation=interpolation,
                                         extent=extent,
                                         vmin=vmin,
                                         vmax=vmax,
                                         cmap=config.cmap, aspect='auto')
                    # plt.ion()
                    divider = make_axes_locatable(axs)
                    cax2 = divider.append_axes("right", size="5%", pad='5%')
                    self.t_slider = Slider(ax=cax2,
                                           label='z',
                                           track_color='dimgrey',
                                           valmin=val_min,
                                           valmax=val_max,
                                           valinit=valinit,
                                           orientation="vertical",
                                           valstep=self.image_data_z,
                                           handle_style={'facecolor': 'white',
                                                         'edgecolor': '.9',
                                                         'size': '20'
                                                        }
                                           )
    
                    def update(val):
                        pos = self.t_slider.val
                        self.Map_2D_plot = self.Map_3D[:,:,np.where(self.image_data_z==pos)[0][0]]
                        image_data = self.Map_2D_plot.values
                        vmin = np.min(image_data)
                        try:
                            vmax = np.max(image_data[np.where(image_data<np.mean(image_data)*1000)])*config.map_scale
                        except:
                            vmax = np.max(image_data)
                        im1.set_data(image_data)
    
                        self.map_z_tick = (vmax - vmin)/config.map_n_ticks_z
                        if self.map_z_tick < 1:
                            self.map_z_tick_decimal = 1
                        else:
                            self.map_z_tick_decimal = 0
                        self.map_z_tick = round(self.map_z_tick, self.map_z_tick_decimal)
                        if self.map_z_tick == 0:
                            self.map_z_tick = 1
    
                        if TwoSlopeNorm < 1 and TwoSlopeNorm > 0:
                            im1.set_clim(vmin=vmin, vmax=vmax,
                                         vcenter=TwoSlopeNorm*vmax)
                        else:
                            im1.set_clim(vmin=vmin, vmax=vmax)
                        self.cbar.set_ticks(MultipleLocator(self.map_z_tick))
                        # plt.gcf().canvas.draw_idle()
                        # plt.gcf().canvas.draw()
                        # plt.gcf().canvas.flush_events()
                        return self.t_slider
    
                    self.t_slider.on_changed(update)

            if switch_3D is False:
                divider = make_axes_locatable(axs)
            cax1 = divider.append_axes("right", size="3.5%", pad=cbar_pad)
            self.cbar = plt.colorbar(im1, cax=cax1,
                                     ticks=MultipleLocator(self.map_z_tick))
            self.cbar.minorticks_on()
            if self.Map_2D_plot.attrs['Normalized'] is True:
                cax1.set_ylabel('Intensity (arb. units)', rotation=270,
                                labelpad=30,
                                fontsize=config.font_size_axis*0.8)
            else:
                cax1.set_ylabel('Intensity (counts)', rotation=270,
                                labelpad=30,
                                fontsize=config.font_size_axis*0.8)

            run_list_s = self.run_num.split(', ')
            try:
                run_list = [int(i) for i in run_list_s]
                run_list.sort()
            except ValueError:
                run_list = run_list_s
            if len(run_list) == 1:
                run_string = f'Run {run_list_s[0]}'
            elif len(run_list) > 4:
                try:
                    run_string = f'Runs {np.min(run_list)}-{np.max(run_list)}'
                except:
                    run_list = [int(i.split('_')[-1]) for i in run_list]
                    run_string = f'Runs {np.min(run_list)}-{np.max(run_list)}'
            else:
                run_list_s = [str(i) for i in run_list_s]
                run_string = ', '.join(run_list_s)
                run_string = 'Runs ' + run_string
            axs.set_title(run_string, pad=15,
                          fontsize=config.font_size_axis*1.2,
                          fontweight="light")

            if self.Map_2D_plot.attrs['x_alt'] is True:
                x_label = self.Map_2D_plot.attrs['x_label_a']
                x_units = self.Map_2D_plot.attrs['x_units_a']
            else:
                x_label = self.Map_2D_plot.attrs['x_label']
                x_units = self.Map_2D_plot.attrs['x_units']

            if self.Map_2D_plot.attrs['y_alt'] is True:
                y_label = self.Map_2D_plot.attrs['y_label_a']
                y_units = self.Map_2D_plot.attrs['y_units_a']
            else:
                y_label = self.Map_2D_plot.attrs['y_label']
                y_units = self.Map_2D_plot.attrs['y_units']

            axs.set_xlabel(f'{x_label} ({x_units})', labelpad=5,
                           fontsize=config.font_size_axis)
            axs.set_ylabel(f'{y_label} ({y_units})', labelpad=10,
                           fontsize=config.font_size_axis*0.8)

            if self.Map_2D_plot.attrs['y_alt'] is True and self.Map_2D_plot.attrs['y_units'] == 'ps':
                position = 0
                if self.varied_y_step is True:
                    coord = self.Map_2D_plot.coords['Dim_y']
                    position = coord.sel(Dim_y=position, method="nearest")
                    position = coord.where(coord == position, drop=True)
                    position = position['Delay index'].values
                axs.axhline(y=position, color=config.color_t0_line,
                            linewidth=config.line_width_t0_line,
                            alpha=config.line_op_t0_line/100,
                            linestyle=config.line_type_t0_line)

            # y axis
            if self.varied_y_step is True:
                axs.set_yticks(pos_list, label_list)
            else:
                axs.yaxis.set_major_locator(MultipleLocator(self.map_y_tick))
                axs.yaxis.set_minor_locator(MultipleLocator(self.map_y_tick /
                                                            config.map_n_ticks_minor))

            try:
                axs.yaxis.get_major_formatter().set_useOffset(False)
            except:
                pass

            # x axis
            axs.xaxis.set_major_locator(MultipleLocator(self.map_x_tick))
            axs.xaxis.set_minor_locator(MultipleLocator(self.map_x_tick /
                                                        config.map_n_ticks_minor))
            axs.tick_params(axis='both', which='major',
                            length=config.map_tick_length,
                            width=config.map_tick_length/4)
            axs.tick_params(axis='both', which='minor',
                            length=config.map_tick_length/1.5,
                            width=config.map_tick_length/4)
            if switch_3D is False:
                cax1.tick_params(axis='both', which='major',
                                 length=config.map_tick_length,
                                 width=config.map_tick_length/4)
                cax1.tick_params(axis='both', which='minor',
                                 length=config.map_tick_length/1.5,
                                 width=config.map_tick_length/4)
            if self.map_y_min == self.map_y_max:
                axs.set_ylim(self.map_y_min-1, self.map_y_max+1)
            if self.map_x_min == self.map_x_max:
                axs.set_xlim(self.map_x_min-1, self.map_x_max+1)

    @staticmethod
    def rounding(x, y):
        '''
        The function rounds energy and delay values to the closest
        values separated by the desired step.
        x - input value
        y - desired step
        '''
        result = np.floor(x/y)*y
        check = (x / y) - np.floor(x/y)
        result = result + (check >= 0.5)*y
        return result

    @staticmethod
    def decimal_n(x):
        '''
        Determines the number of decimal points.
        '''
        result = len(str(x)) - 2
        if isinstance(x, int):
            result = 0
        return result


class read_file_WESPE:
    '''
    The object for storing data from individual hdf5 files.
    It is used further for creating create_batch_WESPE objects.
    '''

    def __init__(self, file_full, DLD='DLD4Q'):
        '''
        Object initialization where reading out of data from hdf5 files occurs.
        '''
        self.unit_dict = {'x': ['X Pixel', 'Alt X Pixel', 'arb. units', 'x'],
                          'y': ['Y Pixel', 'Alt Y Pixel', 'arb. units', 'y'],
                          't': ['Kinetic energy', 'Binding energy',
                                'eV', 'DLD_energy'],
                          'd': ['Delay stage values', 'Delay',
                                'ps', 'DLD_delay'],
                          'b': ['MicroBunch ID', 'Alt MicroBunch ID',
                                'units', 'MB_ID'],
                          'bam': ['BAM', 'Alt BAM',
                                  'arb. units', 'MB_ID'],
                          'mono': ['Mono', 'Alt Mono',
                                   'eV', 'mono']
                          }
        try:
            self.run_num = file_full.split(os.sep)[-1].replace('_energy.mat', '')
            if self.run_num[-1] == 's':
                self.file_full = file_full
                self.file_full = self.file_full.replace(self.run_num,
                                                        self.run_num[:-1])
                self.run_num = self.run_num[:-1]
                self.is_static = True
            else:
                self.file_full = file_full
                self.is_static = False

            f = h5py.File(self.file_full, 'r')
            self.file_folder = self.file_full.split(os.sep)[:-1]
            self.file_folder = f'{os.sep}'.join(self.file_folder)
            self.run_num = str(self.run_num)
            self.static = int(self.run_num)
            self.DLD = DLD

            hdf5_path_read = scan_hdf5(f)
            if len(hdf5_path_read) > 1:
                for path_i in hdf5_path_read:
                    if DLD in path_i:
                        self.hdf5_path = path_i
                        break
                    else:
                        self.hdf5_path = hdf5_path_read[0]
            else:
                self.hdf5_path = hdf5_path_read[0]

            self.DLD_energy = f.get(f'{self.hdf5_path}/energy_Grid_ROI')[0]
            self.e_num = self.DLD_energy.shape[0]
            try:
                self.BAM = f.get(f'{self.hdf5_path}/BAM')[0]
            except:
                self.BAM = 0
            try:
                self.GMD = f.get(f'{self.hdf5_path}/GMDBDA_Electrons')[0]
            except TypeError:
                self.GMD = 0
            try:
                self.mono = f.get(f'{self.hdf5_path}/mono')[0]
            except TypeError:
                self.mono = 0
            try:
                self.x = f.get(f'{self.hdf5_path}/x')[0]
            except TypeError:
                self.x = 0
            try:
                self.y = f.get(f'{self.hdf5_path}/y')[0]
            except TypeError:
                self.y = 0
            self.B_ID = f.get(f'{self.hdf5_path}/bunchID')[0]
            self.MB_ID = f.get(f'{self.hdf5_path}/microbunchID')[0]
            try:
                self.diode = f.get(f'{self.hdf5_path}/Pulse_Energy_DiodeBB')[0]
            except TypeError:
                self.diode = 0
            try:
                self.KE = f.get(f'param_backconvert_GUI/kinenergie_{self.DLD[-2:]}')
                self.KE = int(self.KE[0, 0])
            except TypeError:
                self.KE = f.get('param_backconvert_GUI/kinenergie')
                self.KE = int(self.KE[0])
            try:
                self.PE = f.get(f'param_backconvert_GUI/passenergie_{self.DLD[-2:]}')
                self.PE = int(self.PE[0, 0])
            except TypeError:
                self.PE = f.get('param_backconvert_GUI/passenergie')
                self.PE = int(self.PE[0])
            if self.is_static is False:
                try:
                    if config.BAM_cor == 'on':
                        self.DLD_delay = f.get(f'{self.hdf5_path}/delay_corrBAM')[0]
                    else:
                        self.DLD_delay = f.get(f'{self.hdf5_path}/delay')[0]
                except TypeError:
                    self.DLD_delay = np.full(self.DLD_energy.shape, self.static)
                    self.is_static = True
            else:
                self.DLD_delay = np.full(self.DLD_energy.shape, self.static)
            f.close()
        except:
            if DLD == 'DLD4Q':
                self.file_full = file_full + '.4Q'
            elif DLD == 'DLD1Q':
                self.file_full = file_full + '.1Q'

            self.run_num = self.file_full.split(os.sep)[-1]
            self.run_num = self.run_num.split('.')[0]
            self.run_num = str(self.run_num)
            if self.run_num[-1] == 's':
                self.file_full = self.file_full.replace(self.run_num,
                                                        self.run_num[:-1])
                self.run_num = self.run_num[:-1]
                self.is_static = True
            else:
                self.is_static = False

            f = read_parquet(self.file_full, engine='fastparquet')
            self.file_folder = self.file_full.split(os.sep)[:-1]
            self.file_folder = f'{os.sep}'.join(self.file_folder)

            self.static = int(self.run_num)
            self.DLD = DLD

            self.DLD_energy = f['energy_Grid_ROI'].values
            self.e_num = self.DLD_energy.shape[0]
            try:
                self.BAM = f['BAM'].values
            except:
                self.BAM = 0
            try:
                self.GMD = f['GMDBDA_Electrons'].values
            except TypeError:
                self.GMD = 0
            try:
                self.x = f['x'].values
            except TypeError:
                self.x = 0
            try:
                self.y = f['y'].values
            except TypeError:
                self.y = 0
            try:
                self.mono = f['mono'].values
            except TypeError:
                self.mono = 0
            self.B_ID = f['bunchID'].values
            self.MB_ID = f['microbunchID'].values
            try:
                self.diode = f['Pulse_Energy_DiodeBB'].values
            except:
                self.diode = 0
            try:
                self.KE = f.get(f'param_backconvert_GUI/kinenergie_{self.DLD[-2:]}')
                self.KE = int(self.KE[0, 0])
            except TypeError:
                try:
                    self.KE = f.get('param_backconvert_GUI/kinenergie')
                    self.KE = int(self.KE[0])
                except:
                    self.KE = 0
            try:
                self.PE = f.get(f'param_backconvert_GUI/passenergie_{self.DLD[-2:]}')
                self.PE = int(self.PE[0, 0])
            except TypeError:
                try:
                    self.PE = f.get('param_backconvert_GUI/passenergie')
                    self.PE = int(self.PE[0])
                except:
                    self.PE = 0
            if self.is_static is False:
                try:
                    if config.BAM_cor == 'on':
                        self.DLD_delay = f['delay_corrBAM'].values
                    else:
                        self.DLD_delay = f['delay'].values
                except TypeError:
                    self.DLD_delay = np.full(self.DLD_energy.shape, self.static)
                    self.is_static = True
            else:
                self.DLD_delay = np.full(self.DLD_energy.shape, self.static)
            del f

        self.info = []
        self.info.append(f'File name: {self.file_full.split(os.sep)[-1]} / Electrons detected: {self.e_num}')
        self.info.append(f'Detector: {self.DLD} / KE: {self.KE} eV / PE: {self.PE} eV / Static: {str(self.is_static)}')
        
        self.B_num = int(np.max(self.B_ID) - np.min(self.B_ID))
        self.MB_num = int(np.max(self.MB_ID))
        self.mono_mean = np.around(np.mean(self.mono), 2)
        self.info.append(f'FEL mono: {self.mono_mean} eV / MacroBunches: {self.B_num} / MicroBunches: {self.MB_num}')

        self.KE_min = np.around(np.min(self.DLD_energy), 2)
        self.KE_max = np.around(np.max(self.DLD_energy), 2)
        self.KE_mean = np.around(np.mean(self.DLD_energy), 2)
        self.info.append(f'Min KE: {self.KE_min} eV / Max KE: {self.KE_max} eV / Mean KE: {self.KE_mean} eV')

        self.BE_min = np.around(self.mono_mean - self.KE_max - 4.5, 2)
        self.BE_max = np.around(self.mono_mean - self.KE_min - 4.5, 2)
        self.BE_mean = np.around(self.mono_mean - self.KE_mean - 4.5, 2)
        self.info.append(f'Min BE: {self.BE_min} eV / Max BE: {self.BE_max} eV / Mean BE: {self.BE_mean} eV')

        self.delay_min = np.around(np.min(self.DLD_delay), 2)
        self.delay_max = np.around(np.max(self.DLD_delay), 2)
        self.delay_mean = np.around(np.mean(self.DLD_delay), 2)
        self.info.append(f'Min delay: {self.delay_min} ps / Max delay: {self.delay_max} ps / Mean delay: {self.delay_mean} ps')

        self.GMD_min = np.around(np.min(self.GMD), 2)
        self.GMD_max = np.around(np.max(self.GMD), 2)
        self.GMD_mean = np.around(np.mean(self.GMD), 2)
        self.info.append(f'Min GMD: {self.GMD_min} / Max GMD: {self.GMD_max} / Mean GMD: {self.GMD_mean}')
        self.info = '\n'.join(self.info)

        self.B_ID_const = self.B_ID
        self.B_filter = False
        self.Macro_B_filter = 'All_Macro_B'
        self.Micro_B_filter = 'All_Micro_B'

    def Bunch_filter(self, B_range, B_type='MacroBunch'):
        '''
        Method for bunch filtering.
        B_range - a list for the determination of minimum and maximum values
        for bunch range of interest
            in percent for macrobunches
            in units for microbunches
        B_type - allows to select between 'MacroBunch' and 'MicroBunch'
        '''
        self.B_filter = True
        if B_type == 'MacroBunch':
            self.B_num = np.max(self.B_ID_const) - np.min(self.B_ID_const)
            B_min = np.min(self.B_ID_const)+(self.B_num)*min(B_range)/100
            B_max = np.min(self.B_ID_const)+(self.B_num)*max(B_range)/100
            min_list = np.where(self.B_ID < B_min)
            max_list = np.where(self.B_ID > B_max)
            self.Macro_B_filter = f'{int(B_min)}-{int(B_max)}_Macro_B'
        else:
            B_min = min(B_range)
            B_max = max(B_range)
            if B_type == 'MicroBunch':
                min_list = np.where(self.MB_ID < B_min)
                max_list = np.where(self.MB_ID > B_max)
                self.Micro_B_filter = f'{int(B_min)}-{int(B_max)}_Micro_B'
            else:
                attr = getattr(self, self.unit_dict[B_type][3])
                min_list = np.where(attr < B_min)
                max_list = np.where(attr > B_max)
                if self.Micro_B_filter == 'All_Dims':
                    self.Micro_B_filter = f'{B_min}-{B_max}_{B_type}'
                else:
                    if f'{B_min}-{B_max}_{B_type}' not in self.Micro_B_filter:
                        self.Micro_B_filter += f'_{B_min}-{B_max}_{B_type}'
        del_list = np.append(min_list, max_list)
        print(f'Result of \'{B_type}\' dimension filtering:')
        print(f'{len(del_list)} electrons removed from Run {self.run_num}')
        if del_list.size != 0:
            self.DLD_energy = np.delete(self.DLD_energy, del_list)
            self.DLD_delay = np.delete(self.DLD_delay, del_list)
            if isinstance(self.x, int) is False:
                self.x = np.delete(self.x, del_list)
            if isinstance(self.y, int) is False:
                self.y = np.delete(self.y, del_list)
            if isinstance(self.BAM, int) is False:
                self.BAM = np.delete(self.BAM, del_list)
            if isinstance(self.GMD, int) is False:
                self.GMD = np.delete(self.GMD, del_list)
            if isinstance(self.mono, int) is False:
                self.mono = np.delete(self.mono, del_list)
            self.B_ID = np.delete(self.B_ID, del_list)
            self.MB_ID = np.delete(self.MB_ID, del_list)
            if isinstance(self.diode, int) is False:
                self.diode = np.delete(self.diode, del_list)

    def create_map(self, energy_step=0.05, delay_step=0.1,
                   ordinate='delay', save='on'):
        self.ordinate = ordinate
        '''
        Method for creation of delay-energy map from the data loaded at
        initialization.
        energy_step and delay_step determine the bin size for
        'Dim_x' and 'Delay dimensions'
        '''
        self.energy_step = energy_step
        self.delay_step = delay_step
        save_path = self.file_folder + os.sep + 'netCDF_maps'
        if os.path.isdir(save_path) is False:
            os.mkdir(save_path)
        if self.is_static is True:
            save_path = save_path + os.sep + "Static"
        else:
            save_path = save_path + os.sep + "Delay_scan"
        save_path = save_path + f"_{self.run_num}"
        save_path = save_path + f"_{self.ordinate}"
        save_path = save_path + f"_{self.DLD}_{energy_step}eV"
        save_path = save_path + f"_{delay_step}ps"
        save_path = save_path + f"_{self.Macro_B_filter}"
        save_path = save_path + f"_{self.Micro_B_filter}.nc"
        try:
            # if self.ordinate != 'delay':
            #     raise FileNotFoundError

            if save != 'on':
                raise FileNotFoundError

            with xr.open_dataset(save_path) as ds:
                loaded_map = ds

            for i in list(loaded_map.variables.keys()):
                if 'Run' in i:
                    data_name = i

            x_label = 'Kinetic energy'
            if self.ordinate == 'MB_ID':
                y_label = 'MicroBunch ID'
                y_units = 'units'
            else:
                y_label = 'Delay stage values'
                y_units = 'ps'
            x_units = 'eV'
            x_label_a = 'Binding energy'
            y_label_a = 'Delay'
            x_units_a = 'eV'
            y_units_a = 'ps'
            x_order_rec = False
            y_order_rec = False

            try:
                if loaded_map.variables[data_name].attrs['DLD'] != self.DLD:
                    raise FileNotFoundError
            except:
                raise FileNotFoundError

            image_data = loaded_map.variables[data_name].values
            image_data_y = loaded_map.variables[y_label].values
            image_data_x = loaded_map.variables[x_label].values

            coords = {y_label: ('Dim_y', image_data_y),
                      x_label: ('Dim_x', image_data_x)}
            Map_2D = xr.DataArray(np.array(image_data),
                                  dims=['Dim_y', 'Dim_x'],
                                  coords=coords)
            Map_2D.name = data_name
            BE = loaded_map.variables[x_label_a].values
            Map_2D.coords[x_label_a] = ('Dim_x', BE)

            Map_2D.coords['Dim_x'] = Map_2D.coords[x_label]
            Map_2D.coords['Dim_y'] = Map_2D.coords[y_label]

            Map_2D.attrs = {'x_label': x_label,
                            'x_units': x_units,
                            'x_order_rec': x_order_rec,
                            'y_label': y_label,
                            'y_units': y_units,
                            'y_order_rec': y_order_rec,
                            'x_label_a': x_label_a,
                            'x_units_a': x_units_a,
                            'x_order_rec_a': not x_order_rec,
                            'y_label_a': y_label_a,
                            'y_units_a': y_units_a,
                            'y_order_rec_a': not y_order_rec,
                            'x_alt': False,
                            'y_alt': False,
                            'Normalized': False}

            self.Map_2D = Map_2D
            self.Map_2D_plot = self.Map_2D
            self.en_calib = '-'
        except FileNotFoundError:
            start = timer()
            '''
            This part is supposed to filter artifact values
            in the energy domain.
            '''
            for j in ['DLD_energy', 'DLD_delay']:
                i = getattr(self, j)
                mean = np.mean(i)
                std = np.std(i)
                min_list = np.where(i < mean-3*std)
                if j == 'DLD_energy' and self.mono_mean > 100:
                    max_list = np.where(i > self.mono_mean + 50)
                else:
                    max_list = np.where(i > mean+3*std)
                del_list = np.append(min_list, max_list)
                if del_list.size != 0:
                    self.DLD_energy = np.delete(self.DLD_energy, del_list)
                    self.DLD_delay = np.delete(self.DLD_delay, del_list)
                    try:
                        self.BAM = np.delete(self.BAM, del_list)
                    except:
                        self.BAM = 0
                    if isinstance(self.GMD, int) is False:
                        self.GMD = np.delete(self.GMD, del_list)
                    if isinstance(self.mono, int) is False:
                        self.mono = np.delete(self.mono, del_list)
                    self.B_ID = np.delete(self.B_ID, del_list)
                    self.MB_ID = np.delete(self.MB_ID, del_list)
                    if isinstance(self.diode, int) is False:
                        self.diode = np.delete(self.diode, del_list)
            '''
            Picking Delay or MB_ID as the ordinate axis.
            '''
            if ordinate == 'delay':
                parameter = self.DLD_delay
            elif ordinate == 'MB_ID':
                parameter = self.MB_ID

            DLD_delay_r = self.rounding(parameter, delay_step)
            DLD_energy_r = self.rounding(self.DLD_energy, energy_step)
            DLD_delay_r = np.around(DLD_delay_r,
                                    self.decimal_n(delay_step))
            DLD_energy_r = np.around(DLD_energy_r,
                                     self.decimal_n(energy_step))

            image_data_x = np.arange(DLD_energy_r.min(),
                                     DLD_energy_r.max()+energy_step,
                                     energy_step)
            image_data_x = np.around(image_data_x,
                                     self.decimal_n(energy_step))
            image_data_y = np.arange(DLD_delay_r.min(),
                                     DLD_delay_r.max()+delay_step,
                                     delay_step)
            image_data_y = np.around(image_data_y,
                                     self.decimal_n(delay_step))

            image_data = []
            for i in image_data_y:
                array_1 = DLD_energy_r[np.where(DLD_delay_r == i)]
                line = []
                array_1 = array_1.astype('f')
                for j in image_data_x:
                    array_2 = np.where(array_1 == j)[0]
                    line.append(array_2.shape[0])
                image_data.append(line)

            x_label = 'Kinetic energy'
            if self.ordinate == 'MB_ID':
                y_label = 'MicroBunch ID'
                y_units = 'units'
            else:
                y_label = 'Delay stage values'
                y_units = 'ps'
            x_units = 'eV'
            x_label_a = 'Binding energy'
            y_label_a = 'Delay'
            x_units_a = 'eV'
            y_units_a = 'ps'
            x_order_rec = False
            y_order_rec = False

            coords = {x_label: ("Dim_x", image_data_x),
                      y_label: ("Dim_y", image_data_y)}

            Map_2D = xr.DataArray(np.array(image_data),
                                  dims=["Dim_y", "Dim_x"],
                                  coords=coords)
            Map_2D.coords["Dim_y"] = Map_2D.coords[y_label]
            Map_2D.coords["Dim_x"] = Map_2D.coords[x_label]

            BE = self.mono_mean - np.array(image_data_x) - 4.5
            BE = np.around(self.rounding(BE, energy_step), self.decimal_n(energy_step))
            image_data_x_a = BE

            try:
                Map_2D.coords[x_label_a] = ('Dim_x', image_data_x_a)
            except:
                pass
            try:
                Map_2D.coords[y_label_a] = ('Dim_y', image_data_y_a)
            except:
                pass
            Map_2D.attrs = {'x_label': x_label,
                            'x_units': x_units,
                            'x_order_rec': x_order_rec,
                            'y_label': y_label,
                            'y_units': y_units,
                            'y_order_rec': y_order_rec,
                            'x_label_a': x_label_a,
                            'x_units_a': x_units_a,
                            'x_order_rec_a': not x_order_rec,
                            'y_label_a': y_label_a,
                            'y_units_a': y_units_a,
                            'y_order_rec_a': not y_order_rec,
                            'x_alt': False,
                            'y_alt': False,
                            'Normalized': False,
                            'DLD': self.DLD}
            Map_2D.name = 'Run ' + str(self.run_num)
            self.Map_2D = Map_2D
            self.Map_2D_plot = Map_2D
            self.y_step = self.Map_2D.coords['Dim_y'].values
            try:
                self.y_step = np.min(np.abs(np.gradient(self.y_step)))
            except:
                self.y_step = 1
            self.x_step = self.Map_2D.coords['Dim_x'].values
            self.x_step = np.gradient(self.x_step)[0]
            self.en_calib = '-'

            if save != 'off':  # and self.ordinate == 'delay':
                self.Map_2D.to_netcdf(save_path, engine="scipy")
                print('Delay-energy map saved as:')
                print(save_path)
            end = timer()
            print(f'Run {self.run_num} done')
            print(f'Elapsed time: {round(end-start, 1)} s')

    def time_zero(self, t0=1328.2):
        '''
        Method for creating new array coordinate 'Delay relative t0'
        after specification of the delay stage value considered as time zero.
        '''
        check = []
        for i in ['x', 'y', 'z']:
            try:
                if self.Map_2D_plot.attrs[f'{i}_units'] == 'ps':
                    check.append(i)
            except:
                pass
        try:
            check = check[0]
            self.t0 = read_file_WESPE.rounding(t0, getattr(self, f'{check}_step'))
            label_a = self.Map_2D_plot.attrs[f'{check}_label_a']
            label = self.Map_2D_plot.attrs[f'{check}_label']
            image_data_a = self.t0 - self.Map_2D.coords[label].values
            self.Map_2D.coords[label_a] = (f'Dim_{check}', image_data_a)
            self.Map_2D_plot.coords[label_a] = (f'Dim_{check}', image_data_a)
            self.set_T0()
        except:
            pass

    def set_T0(self):
        '''
        Method for switching visualization to 'Delay relative t0'
        coordinate of 'Dim_y' dimension.
        '''
        check = []
        for i in ['x', 'y', 'z']:
            try:
                if self.Map_2D_plot.attrs[f'{i}_units'] == 'ps':
                    check.append(i)
            except:
                pass
        try:
            check = check[0]
            if self.Map_2D_plot.attrs[f'{check}_alt'] is False:
                coord = self.Map_2D_plot.coords[self.Map_2D_plot.attrs[f'{check}_label_a']]
                self.Map_2D_plot.coords[f'Dim_{check}'] = coord
                self.Map_2D_plot.attrs[f'{check}_alt'] = True
        except:
            pass

    def set_Tds(self):
        '''
        Method for switching visualization to 'Delay stage values'
        coordinate of 'Dim_y' dimension.
        '''
        check = []
        for i in ['x', 'y', 'z']:
            try:
                if self.Map_2D_plot.attrs[f'{i}_units'] == 'ps':
                    check.append(i)
            except:
                pass
        try:
            check = check[0]
            if self.Map_2D_plot.attrs[f'{check}_alt'] is True:
                coord = self.Map_2D_plot.coords[self.Map_2D_plot.attrs[f'{check}_label']]
                self.Map_2D_plot.coords[f'Dim_{check}'] = coord
                self.Map_2D_plot.attrs[f'{check}_alt'] = False
        except:
            pass

    def create_dif_map(self):
        '''
        This method generates a difference map by averaging data before
        -0.25 ps and subtracting it from the delay-energy map.
        '''
        attrs = self.Map_2D_plot.attrs
        t_axis_step = self.Map_2D_plot.coords['Dim_y'].values
        try:
            t_axis_step = abs(np.median(np.gradient(t_axis_step)))
        except ValueError:
            t_axis_step = 1
        t_axis_step = int(-2.5*t_axis_step)
        norm = self.Map_2D_plot.loc[t_axis_step:].mean('Dim_y')
        self.Map_2D_dif = self.Map_2D_plot - norm
        self.Map_2D_dif.attrs = attrs

    def set_BE(self):
        '''
        Method for switching visualization to 'Binding energy'
        coordinate of 'Energy' dimension.
        '''
        if self.Map_2D_plot.attrs['x_alt'] is False:
            coord = self.Map_2D_plot.coords[self.Map_2D_plot.attrs['x_label_a']]
            self.Map_2D_plot.coords['Dim_x'] = coord
            self.Map_2D_plot.attrs['x_alt'] = True

    def axs_plot(self, axs):
        # Loading configs from json file.
        try:
            with open('config.json', 'r') as json_file:
                config = json.load(json_file)
        except FileNotFoundError:
            with open('packages/config.json', 'r') as json_file:
                config = json.load(json_file)
        config = json.dumps(config)
        config = json.loads(config,
                            object_hook=lambda d: SimpleNamespace(**d))
        '''
        Method for creating matplotlib axes for delay-energy map visualization.
        Uses the corresponding method from the create_batch_WESPE object.
        '''
        create_batch_WESPE.axs_plot(self, axs)

    @staticmethod
    def rounding(x, y):
        '''
        The function rounds energy and delay values to the closest
        values separated by the desired step.
        x - input value
        y - desired step
        '''
        result = np.floor(x/y)*y
        check = (x / y) - np.floor(x/y)
        result = result + (check >= 0.5)*y
        return result

    @staticmethod
    def decimal_n(x):
        '''
        Determines the number of decimal points.
        '''
        result = len(str(x)) - 2
        if isinstance(x, int):
            result = 0
        return result


class create_batch_ALS(create_batch_WESPE):
    '''
    The object for storing data of combined runs.
    '''

    def __init__(self, file_dir, run_list, DLD='DLD4Q'):
        '''
        This initialization happens on 'Upload runs'.
        '''
        self.file_dir = file_dir
        self.batch_dir, self.batch_list = [], []
        for run_number in run_list:
            file_name = 'PS_Scan_' + f'{run_number}'
            file_full = file_dir + os.sep + file_name
            if os.path.isdir(file_full) is False:
                file_name = f'{run_number}'
                file_full = file_dir + os.sep + file_name
            self.batch_list.append(read_file_ALS(file_full,
                                                 DLD=DLD))
            self.batch_dir.append(file_full)

        full_info = []
        for i in self.batch_list:
            full_info.append(i.info)
        self.full_info = 'DETAILED INFO:\n\n' + '\n\n'.join(full_info)

        title = 'SHORT SUMMARY:\n'
        run_num, is_static, KE, mono = [], [], [], []
        for i in self.batch_list:
            # run_num.append(str(i.run_num))
            run_num.append(str(i.run_num_full))
            is_static.append(i.is_static)
            KE.append(i.KE)
            mono.append(i.mono_mean)
        # Run numbers
        self.run_num = ', '.join(run_num)
        try:
            run_num = [int(i) for i in run_num]
            run_num.sort()
        except ValueError:
            run_num = run_num
        if len(run_num) == 1:
            run_num = f'Uploaded run: {run_num[0]}'
        elif len(run_num) > 6:
            temp_list = [int(i.split('-run')[-1]) for i in run_num]
            run_num = f'Uploaded runs: {np.min(temp_list)}-{np.max(temp_list)}'
        else:
            run_num = [str(i) for i in run_num]
            run_num = ', '.join(run_num)
            run_num = 'Uploaded runs: ' + run_num
        self.run_num_o = run_num.replace('Uploaded runs: ', '')
        # Static scan check
        if all(is_static):
            is_static_s = 'Static check: All runs are static (+)'
        elif not any(is_static):
            is_static_s = 'Static check: All runs are delay scans (+)'
        else:
            is_static_s = 'Static check: Delay scans are mixed with static scans (!!!)'
        # Energy region check
        if np.max(KE) - np.min(KE) > 5:
            KE_s = 'Region check: Various energy regions are on the list (!!!)'
        else:
            KE_s = 'Region check: Homogeneous energy regions (+)'
        # Mono check
        if np.max(mono) - np.min(mono) > 0.15:
            mono_s = 'Mono check: Various mono values for different runs (!!!)'
        else:
            mono_s = 'Mono check: No mono energy jumps detected (+)'
        self.en_threshold = np.max(mono) + 50
        if self.en_threshold < 50:
            self.en_threshold = 1000
        static_cut_list = []
        for counter, i in enumerate(is_static):
            if i is True:
                static_cut = np.mean(self.batch_list[counter].DLD_delay)
                static_cut_list.append(static_cut)
        self.static_cut_list = static_cut_list
        # short_info = [title, run_num, is_static_s, KE_s, mono_s]
        short_info = [title, run_num]
        self.short_info = '\n'.join(short_info) + '\n\n'

    def create_map(self):
        '''
        This method sums delay-energy maps of individual runs
        uploaded to the batch.
        '''
        self.x_step = self.batch_list[0].x_step
        self.y_step = self.batch_list[0].y_step
        self.ordinate = self.batch_list[0].ordinate
        attrs = self.batch_list[0].Map_2D.attrs
        for counter, i in enumerate(self.batch_list):
            if counter == 0:
                total_map = i.Map_2D
            else:
                total_map = total_map + i.Map_2D
        total_map.attrs = attrs
        if np.min(total_map.values.shape) == 0:
            concat_list = []
            for counter, i in enumerate(self.batch_list):
                concat_list.append(i.Map_2D)
            total_map = xr.concat(concat_list, dim='Dim_y', coords="minimal",
                                  compat='override')
        if np.min(total_map.values.shape) == 0:
            total_map.attrs['Merge successful'] = False
        self.Map_2D = total_map
        self.Map_2D_plot = self.Map_2D
        if total_map.coords['Dim_y'].values.shape[0] > 1:
            if np.around(np.std(np.gradient(total_map.coords['Dim_y'])), 3) > 0:
                self.varied_y_step = True


class create_batch_MM(create_batch_WESPE):
    '''
    The object for storing data of combined runs.
    '''

    def __init__(self, file_dir, run_list, DLD='DLD4Q'):
        '''
        This initialization happens on 'Upload runs'.
        '''
        self.file_dir = file_dir
        try:
            self.batch_dir, self.batch_list = [], []
            for run_number in run_list:
                file_name = f'{run_number}' + os.sep + f'{run_number}_energy.mat'
                file_full = file_dir + os.sep + file_name
                self.batch_list.append(read_file_MM(file_full, DLD=DLD))
                self.batch_dir.append(file_full)
        except:
            self.batch_dir, self.batch_list = [], []
            for run_number in run_list:
                for ext in ['.HEXTOF', '.SXP', '.parquet']:
                    file_full = file_dir + os.sep + f'{run_number}' + ext
                    if os.path.isfile(file_full+ext) is False and os.path.isfile(file_full+f's{ext}'):
                        file_name = f'{run_number}' + os.sep + f'{run_number}'
                        file_full = file_dir + os.sep + file_name + ext
                    try:
                        self.batch_list.append(read_file_MM(file_full, DLD=DLD))
                        self.batch_dir.append(file_full)
                        break
                    except:
                        continue

        full_info = []
        for i in self.batch_list:
            full_info.append(i.info)
        self.full_info = 'DETAILED INFO:\n\n' + '\n\n'.join(full_info)

        title = 'SHORT SUMMARY:\n'
        run_num, is_static, KE, mono = [], [], [], []
        for i in self.batch_list:
            run_num.append(str(i.run_num))
            is_static.append(i.is_static)
            KE.append(i.KE)
            mono.append(i.mono_mean)
        # Run numbers
        self.run_num = ', '.join(run_num)
        try:
            run_num = [int(i) for i in run_num]
        except:
            run_num = [int(i.split('_')[-1]) for i in run_num]
        run_num.sort()
        if len(run_num) == 1:
            run_num = f'Uploaded run: {run_num[0]}'
        elif len(run_num) > 6:
            run_num = f'Uploaded runs: {np.min(run_num)}-{np.max(run_num)}'
        else:
            run_num = [str(i) for i in run_num]
            run_num = ', '.join(run_num)
            run_num = 'Uploaded runs: ' + run_num
        self.run_num_o = run_num.replace('Uploaded runs: ', '')
        # Static scan check
        if all(is_static):
            is_static_s = 'Static check: All runs are static (+)'
        elif not any(is_static):
            is_static_s = 'Static check: All runs are delay scans (+)'
        else:
            is_static_s = 'Static check: Delay scans are mixed with static scans (!!!)'
        # Energy region check
        if np.max(KE) - np.min(KE) > 5:
            KE_s = 'Region check: Various energy regions are on the list (!!!)'
        else:
            KE_s = 'Region check: Homogeneous energy regions (+)'
        # Mono check
        if np.max(mono) - np.min(mono) > 0.15:
            mono_s = 'Mono check: Various mono values for different runs (!!!)'
        else:
            mono_s = 'Mono check: No mono energy jumps detected (+)'
        self.en_threshold = np.max(mono) + 50
        self.mono = mono
        if self.en_threshold < 50:
            self.en_threshold = 1000
        static_cut_list = []
        for counter, i in enumerate(is_static):
            if i is True:
                static_cut = np.mean(self.batch_list[counter].DLD_delay)
                static_cut_list.append(static_cut)
        self.static_cut_list = static_cut_list
        short_info = [title, run_num, is_static_s, KE_s, mono_s]
        self.short_info = '\n'.join(short_info) + '\n\n'

    def set_KE(self):
        '''
        Method for switching visualization to 'Kinetic energy'
        coordinate of 'Energy' dimension.
        '''
        check = []
        for i in ['x', 'y', 'z']:
            try:
                if self.Map_2D_plot.attrs[f'{i}_units'] == 'eV':
                    check.append(i)
                elif 'energy' in self.Map_2D_plot.attrs[f'{i}_label']:
                    check.append(i)
            except:
                pass
        try:
            check = check[0]
            if self.Map_2D_plot.attrs[f'{check}_alt'] is True:
                coord = self.Map_2D_plot.coords[self.Map_2D_plot.attrs[f'{check}_label']]
                self.Map_2D_plot.coords[f'Dim_{check}'] = coord
                self.Map_2D_plot.attrs[f'{check}_alt'] = False
        except:
            pass
            
    def set_BE(self):
        '''
        Method for switching visualization to 'Binding energy'
        coordinate of 'Energy' dimension.
        '''
        check = []
        for i in ['x', 'y', 'z']:
            try:
                if self.Map_2D_plot.attrs[f'{i}_units'] == 'eV':
                    check.append(i)
                elif 'energy' in self.Map_2D_plot.attrs[f'{i}_label']:
                    check.append(i)
            except:
                pass
        try:
            check = check[0]
            if self.Map_2D_plot.attrs[f'{check}_alt'] is False:
                coord = self.Map_2D_plot.coords[self.Map_2D_plot.attrs[f'{check}_label_a']]
                self.Map_2D_plot.coords[f'Dim_{check}'] = coord
                self.Map_2D_plot.attrs[f'{check}_alt'] = True
        except:
            pass
        
    def norm_total_e(self):
        '''
        Method for normalization of delay-energy map in terms of the concept
        that every time delay line should contain the same number of detected
        electrons, i.e., we have only redistribution of electrons in the
        energy domain.
        '''
        check = []
        for i in ['x', 'y', 'z']:
            try:
                if self.Map_2D_plot.attrs[f'{i}_units'] == 'eV':
                    check.append(i)
                elif 'energy' in self.Map_2D_plot.attrs[f'{i}_label']:
                    check.append(i)
            except:
                pass
        try:
            check = check[0]
            arr = self.Map_2D_plot
            attrs = self.Map_2D_plot.attrs
    
            norm = arr.sum(f'Dim_{check}', skipna=True)
            new_arr = arr/norm * norm.mean()
    
            self.Map_2D_plot = new_arr
            self.Map_2D_plot.attrs = attrs
            self.Map_2D_plot.attrs['Normalized'] = True
        except:
            pass

    def create_map(self):
        '''
        This method sums delay-energy maps of individual runs
        uploaded to the batch.
        '''
        self.energy_step = self.batch_list[0].energy_step
        self.delay_step = self.batch_list[0].delay_step
        self.ordinate = self.batch_list[0].ordinate
        attrs = self.batch_list[0].Map_2D.attrs
        for counter, i in enumerate(self.batch_list):
            if counter == 0:
                total_map = i.Map_2D
            else:
                total_map = total_map + i.Map_2D
        total_map.attrs = attrs
        try:
            total_map.coords['Binding energy']
            total_map.coords['Kinetic energy']
        except KeyError:
            total_map = xr.DataArray([])
        if np.min(total_map.values.shape) == 0:
            concat_list = []
            for counter, i in enumerate(self.batch_list):
                concat_list.append(i.Map_2D)
            total_map = xr.combine_by_coords(concat_list, compat='override')
            total_map.coords[attrs['y_label']] = total_map.coords['Dim_y']
            total_map = total_map.to_array(dim='variable', name=None)
            total_map = total_map.sum(dim='variable')
            total_map.attrs = attrs

        if np.min(total_map.values.shape) == 0:
            total_map.attrs['Merge successful'] = False

        # Filter empty EDCs
        check = []
        for i in ['x', 'y', 'z']:
            try:
                if total_map.attrs[f'{i}_units'] == 'ps':
                    check.append(i)
                elif 'ID' in total_map.attrs[f'{i}_label']:
                    check.append(i)
            except:
                pass
            
        if len(check) > 0:
            try:
                for check_i in check:
                    y_check = total_map
                    # shape_list = []
                    for i in total_map.dims:
                        if i != f'Dim_{check_i}':
                            # shape_list.append(total_map.coords[i].shape[0])
                            y_check = y_check.sum(i, skipna=True)
                    # y_check = y_check/sum(shape_list)
                    remove_list = np.where(y_check < np.max(y_check)*0.01)
                    total_map = total_map.drop_isel({f'Dim_{check_i}': remove_list})
            except:
                pass

        shape = total_map.coords['Dim_y'].values.shape[0]
        total_map.coords['Delay index'] = ('Dim_y', np.arange(shape))

        self.Map_2D = total_map.fillna(0)
        self.Map_2D = self.Map_2D.where(self.Map_2D.coords[attrs['x_label']].notnull(), drop=True)
        try:
            # BE must have negative step
            if np.median(np.gradient(self.Map_2D.coords[attrs['x_label_a']].values)) > 0:
                self.Map_2D = self.Map_2D.isel(Dim_x=slice(None, None, -1))
        except:
            pass
        self.Map_2D_plot = self.Map_2D

        try:
            yaml_name = [i for i in os.listdir(self.file_dir) if i.endswith('.yaml')]
            yaml_full = self.file_dir + os.sep + yaml_name[0]
            with open(yaml_full, 'r') as yaml_file:
                yaml_data = yaml_safe_load(yaml_file)
            energy_offset = yaml_data['energy']['calibration']['E0']
            energy_scale = yaml_data['energy']['calibration']['energy_scale']
            tof_distance = yaml_data['energy']['calibration']['d']
            time_offset = yaml_data['energy']['calibration']['t0']
            binwidth = yaml_data['dataframe']['tof_binwidth']
            binning = yaml_data['dataframe']['tof_binning']
            if energy_scale == 'binding':
                sign = -1
            else:
                sign = 1
            t = self.Map_2D.coords['Binding energy']*1000
            E = (
                2.84281e-12 * sign * (tof_distance / (t * binwidth * 2**binning - time_offset)) ** 2
                + energy_offset
                )
            mono = np.nanmean(self.mono)
            E_alt = mono - E - 4.5
            if energy_scale == 'binding':
                self.Map_2D.coords['Kinetic energy'].values = E_alt.values
                self.Map_2D.coords['Binding energy'].values = E.values
            else:
                self.Map_2D.coords['Kinetic energy'].values = E.values
                self.Map_2D.coords['Binding energy'].values = E_alt.values

            check = []
            for i in ['x', 'y', 'z']:
                try:
                    if self.Map_2D_plot.attrs[f'{i}_units'] == 'eV':
                        check.append(i)
                    elif 'energy' in self.Map_2D_plot.attrs[f'{i}_label']:
                        check.append(i)
                except:
                    pass
            try:
                check = check[0]
                if self.Map_2D.attrs[f'{check}_alt'] is False:
                    state = 'Kinetic energy'
                else:
                    state = 'Binding energy'
                self.Map_2D.coords[f'Dim_{check}'] = self.Map_2D.coords[state]
                self.Map_2D.attrs[f'{check}_units'] = 'eV'
                self.Map_2D.attrs[f'{check}_units_a'] = 'eV'
                self.Map_2D_plot = self.Map_2D
            except:
                pass
        except:
            print('***No energy calibration was applied***')

        self.varied_y_step = False
        try:
            self.y_step = self.Map_2D.coords['Dim_y'].values
            self.y_step = np.min(np.abs(np.gradient(self.y_step)))
            if np.around(np.std(np.gradient(total_map.coords['Dim_y'])), 3) > 0:
                self.varied_y_step = True
        except:
            self.y_step = 1
        self.x_step = self.Map_2D.coords['Dim_x'].values
        self.x_step = np.min(np.abs(np.gradient(self.x_step)))
        self.x_step = np.around(self.x_step, self.decimal_n(self.energy_step))
        self.y_step = np.around(self.y_step, self.decimal_n(self.delay_step))


class read_file_MM(create_batch_WESPE):
    '''
    The object for storing data from individual hdf5 files.
    It is used further for creating create_batch_WESPE objects.
    '''

    def __init__(self, file_full, DLD='DLD4Q'):
        '''
        Object initialization where reading out of data from hdf5 files occurs.
        '''
        self.unit_dict = {'x': ['X Pixel', 'Alt X Pixel', 'karb. units', 'x'],
                          'y': ['Y Pixel', 'Alt Y Pixel', 'karb. units', 'y'],
                          't': ['Kinetic energy', 'Binding energy',
                                'karb. units', 'DLD_energy'],
                          'd': ['Delay stage values', 'Delay',
                                'ps', 'DLD_delay'],
                          'b': ['MicroBunch ID', 'Alt MicroBunch ID',
                                'units', 'MB_ID'],
                          'bam': ['BAM', 'Alt BAM',
                                  'arb. units', 'MB_ID'],
                          'mono': ['Mono', 'Alt Mono',
                                   'eV', 'mono']
                          }

        self.file_full = file_full
        self.run_num = self.file_full.split(os.sep)[-1]
        self.detector = self.run_num.split('.')[-1]
        self.run_num = self.run_num.split('.')[0]
        self.run_num = str(self.run_num)
        if self.run_num[-1] == 's':
            self.file_full = self.file_full.replace(self.run_num,
                                                    self.run_num[:-1])
            self.run_num = self.run_num[:-1]
            self.is_static = True
        else:
            self.is_static = False

        f = read_parquet(self.file_full, engine='fastparquet')

        try:
            f = f.rename(columns={"dldTime": "dldTimeSteps"})
        except:
            pass

        try:
            subset = []
            for i in ['trainId', 'pulseId', 'dldPosX', 'dldPosY', 'x', 'y',
                      'dldTimeSteps', 'delayStage']:
                if i in f.keys():
                    subset.append(i)
            f = f.dropna(subset=subset, how='any')
        except:
            pass

        self.file_folder = self.file_full.split(os.sep)[:-1]
        self.file_folder = f'{os.sep}'.join(self.file_folder)

        try:
            self.static = int(self.run_num)
        except:
            self.static = int(self.run_num.split('_')[-1])
        self.DLD = DLD
        
        try:
            self.DLD_energy = f['dldTime'].values/1000
        except:
            self.DLD_energy = f['dldTimeSteps'].values/1000
        try:
            self.x = f['x']/1000
            self.y = f['y']/1000
        except:
            self.x = f['dldPosX']/1000
            self.y = f['dldPosY']/1000
        self.e_num = self.DLD_energy.shape[0]
        try:
            self.BAM = f['bam'].values
        except:
            self.BAM = 0
        try:
            self.GMD = f['gmdBda'].values
        except:
            self.GMD = 0
        try:
            self.mono = f['monochromatorPhotonEnergy'].values
        except:
            self.mono = 0
        self.B_ID = f['trainId'].values
        self.MB_ID = f['pulseId'].values
        try:
            self.diode = f['Pulse_Energy_DiodeBB'].values
        except:
            self.diode = 0
        try:
            self.KE = f.get(f'param_backconvert_GUI/kinenergie_{self.DLD[-2:]}')
            self.KE = int(self.KE[0, 0])
        except:
            try:
                self.KE = f.get('param_backconvert_GUI/kinenergie')
                self.KE = int(self.KE[0])
            except:
                self.KE = 0
        try:
            self.PE = f.get(f'param_backconvert_GUI/passenergie_{self.DLD[-2:]}')
            self.PE = int(self.PE[0, 0])
        except:
            try:
                self.PE = f.get('param_backconvert_GUI/passenergie')
                self.PE = int(self.PE[0])
            except:
                self.PE = 0
        if self.is_static is False:
            try:
                if config.BAM_cor == 'on':
                    self.DLD_delay = f['delayStage'].values
                else:
                    self.DLD_delay = f['delayStage'].values
            except:
                self.DLD_delay = np.full(self.DLD_energy.shape, self.static)
                self.is_static = True
        else:
            self.DLD_delay = np.full(self.DLD_energy.shape, self.static)
        del f

        self.info = []
        self.info.append(f'File name: {self.file_full.split(os.sep)[-1]} / Electrons detected: {self.e_num}')
        self.info.append(f'Detector: {self.detector} / KE: {self.KE} eV / PE: {self.PE} eV / Static: {str(self.is_static)}')

        self.B_num = int(np.max(self.B_ID) - np.min(self.B_ID))
        try:
            self.MB_num = int(np.max(self.MB_ID[self.MB_ID < 3*np.mean(self.MB_ID)]))
        except:
            self.MB_num = [0, 1]

        MB_hist = np.histogram(self.MB_ID, bins=4000)
        self.MB_num = int(np.max(MB_hist[-1][1:][MB_hist[0]>np.max(MB_hist[0])*0.001]))
        EN_hist = np.histogram(self.DLD_energy, bins=1000)
        KE = EN_hist[-1][1:][EN_hist[0]>np.max(EN_hist[0])*0.001]
        if self.KE == 0:
            self.KE = [0,1]
        self.KE = [0,1]

        #self.mono_mean = np.around(np.mean(self.mono[self.mono>0]), 2)
        self.mono_mean = np.nanmean(self.mono)
        self.info.append(f'FEL mono: {self.mono_mean:.2f} eV / MacroBunches: {self.B_num} / MicroBunches: {self.MB_num}')

        self.KE_min = np.around(np.min(KE), 3)
        self.KE_max = np.around(np.max(KE), 3)
        self.KE_mean = np.around(np.mean(KE), 3)
        #self.info.append(f'Min TOF: {self.KE_min:.3f} ka.u. / Max TOF: {self.KE_max:.3f} ka.u. / Mean TOF: {self.KE_mean:.3f} ka.u.')

        self.delay_min = np.around(np.min(self.DLD_delay), 2)
        self.delay_max = np.around(np.max(self.DLD_delay), 2)
        self.delay_mean = np.around(np.mean(self.DLD_delay), 2)
        self.info.append(f'Min delay: {self.delay_min:.2f} ps / Max delay: {self.delay_max:.2f} ps / Mean delay: {self.delay_mean:.2f} ps')

        self.GMD_min = np.around(np.min(self.GMD), 2)
        self.GMD_max = np.around(np.max(self.GMD), 2)
        self.GMD_mean = np.around(np.mean(self.GMD), 2)
        self.info.append(f'Min GMD: {self.GMD_min:.3f} / Max GMD: {self.GMD_max:.3f} / Mean GMD: {self.GMD_mean:.3f}')
        self.info = '\n'.join(self.info)

        self.B_ID_const = self.B_ID
        self.B_filter = False
        self.Macro_B_filter = 'All_Macro_B'
        self.Micro_B_filter = 'All_Dims'

    def Bunch_filter(self, B_range, B_type='MacroBunch'):
        '''
        Method for bunch filtering.
        B_range - a list for the determination of minimum and maximum values
        for bunch range of interest
            in percent for macrobunches
            in units for microbunches
        B_type - allows to select between 'MacroBunch' and 'MicroBunch'
        '''
        self.B_filter = True
        if B_type == 'MacroBunch':
            self.B_num = np.max(self.B_ID_const) - np.min(self.B_ID_const)
            B_min = np.min(self.B_ID_const)+(self.B_num)*min(B_range)/100
            B_max = np.min(self.B_ID_const)+(self.B_num)*max(B_range)/100
            min_list = np.where(self.B_ID < B_min)
            max_list = np.where(self.B_ID > B_max)
            self.Macro_B_filter = f'{int(B_min)}-{int(B_max)}_Macro_B'
        else:
            B_min = min(B_range)
            B_max = max(B_range)
            if B_type == 'MicroBunch':
                min_list = np.where(self.MB_ID < B_min)
                max_list = np.where(self.MB_ID > B_max)
                self.Micro_B_filter = f'{int(B_min)}-{int(B_max)}_Micro_B'
            else:
                attr = getattr(self, self.unit_dict[B_type][3])
                min_list = np.where(attr < B_min)
                max_list = np.where(attr > B_max)
                if self.Micro_B_filter == 'All_Dims':
                    self.Micro_B_filter = f'{B_min}-{B_max}_{B_type}'
                else:
                    if f'{B_min}-{B_max}_{B_type}' not in self.Micro_B_filter:
                        self.Micro_B_filter += f'_{B_min}-{B_max}_{B_type}'
        del_list = np.append(min_list, max_list)
        print(f'Result of \'{B_type}\' dimension filtering:')
        print(f'{len(del_list)} electrons removed from Run {self.run_num}')
        if del_list.size != 0:
            self.DLD_energy = np.delete(self.DLD_energy, del_list)
            self.DLD_delay = np.delete(self.DLD_delay, del_list)
            self.x = np.delete(self.x, del_list)
            self.y = np.delete(self.y, del_list)
            if isinstance(self.BAM, int) is False:
                self.BAM = np.delete(self.BAM, del_list)
            if isinstance(self.GMD, int) is False:
                self.GMD = np.delete(self.GMD, del_list)
            if isinstance(self.mono, int) is False:
                self.mono = np.delete(self.mono, del_list)
            self.B_ID = np.delete(self.B_ID, del_list)
            self.MB_ID = np.delete(self.MB_ID, del_list)
            if isinstance(self.diode, int) is False:
                self.diode = np.delete(self.diode, del_list)

    def create_map(self, energy_step=0.05, delay_step=0.1, z_step=100,
                   ordinate='td', save='on'):
        '''
        Method for creation of delay-energy map from the data loaded at
        initialization.
        energy_step and delay_step determine the bin size for
        'Dim_x' and 'Delay dimensions'
        '''

        self.ordinate = ordinate
        
        if len(ordinate) > 2:
            self.switch_3D = True
        else:
            self.switch_3D = False

        if ordinate[0] in ['x', 'y', 't'] and energy_step < 0.001:
            energy_step = 0.001
        if ordinate[1] in ['x', 'y', 't'] and delay_step < 0.001:
            delay_step = 0.001

        if ordinate[0] in ['b'] and energy_step < 1:
            energy_step = 1
        if ordinate[1] in ['b'] and delay_step < 1:
            delay_step = 1

        self.energy_step = energy_step
        self.delay_step = delay_step
        save_path = self.file_folder + os.sep + 'netCDF_maps'
        if os.path.isdir(save_path) is False:
            os.mkdir(save_path)
        if self.is_static is True:
            save_path = save_path + os.sep + "Static"
        else:
            save_path = save_path + os.sep + "Delay_scan"
        save_path = save_path + f"_{self.run_num}"
        save_path = save_path + f"_{self.ordinate}"
        save_path = save_path + f"_{energy_step}kau"
        save_path = save_path + f"_{delay_step}ps"
        if self.switch_3D is True:
            self.z_step = z_step
            save_path = save_path + f"_{z_step}ps"
        save_path = save_path + f"_{self.Macro_B_filter}"
        save_path = save_path + f"_{self.Micro_B_filter}.nc"
        try:
            # if self.ordinate != 'delay':
            #     raise FileNotFoundError

            if save != 'on':
                raise FileNotFoundError

            with xr.open_dataset(save_path) as ds:
                loaded_map = ds

            for i in list(loaded_map.variables.keys()):
                if 'Run' in i:
                    data_name = i

            x_label = self.unit_dict[ordinate[0]][0]
            y_label = self.unit_dict[ordinate[1]][0]
            y_units = self.unit_dict[ordinate[1]][2]
            x_units = self.unit_dict[ordinate[0]][2]
            x_label_a = self.unit_dict[ordinate[0]][1]
            y_label_a = self.unit_dict[ordinate[1]][1]
            x_units_a = self.unit_dict[ordinate[0]][2]
            y_units_a = self.unit_dict[ordinate[1]][2]
            x_order_rec = False
            y_order_rec = False

            if self.switch_3D is True:
                z_label = self.unit_dict[ordinate[2]][0]
                z_units = self.unit_dict[ordinate[2]][2]
                z_label_a = self.unit_dict[ordinate[2]][1]
                z_units_a = self.unit_dict[ordinate[2]][2]
                z_order_rec = False
                image_data_z = loaded_map.variables[z_label].values

            try:
                if loaded_map.variables[data_name].attrs['DLD'] != self.DLD:
                    raise FileNotFoundError
            except:
                raise FileNotFoundError

            image_data = loaded_map.variables[data_name].values
            image_data_y = loaded_map.variables[y_label].values
            image_data_x = loaded_map.variables[x_label].values

            if self.switch_3D is False:
                coords = {x_label: ("Dim_x", image_data_x),
                          y_label: ("Dim_y", image_data_y)}

                Map_2D = xr.DataArray(np.array(image_data),
                                      dims=["Dim_y", "Dim_x"],
                                      coords=coords)
            else:
                coords = {x_label: ("Dim_x", image_data_x),
                          y_label: ("Dim_y", image_data_y),
                          z_label: ("Dim_z", image_data_z)}

                Map_2D = xr.DataArray(np.array(image_data),
                                      dims=["Dim_y", "Dim_x", "Dim_z"],
                                      coords=coords)
                Map_2D.coords["Dim_z"] = Map_2D.coords[z_label]

            Map_2D.name = data_name

            Map_2D.coords['Dim_x'] = Map_2D.coords[x_label]
            Map_2D.coords['Dim_y'] = Map_2D.coords[y_label]

            if self.switch_3D is False:
                Map_2D.attrs = {'x_label': x_label,
                                'x_units': x_units,
                                'x_order_rec': x_order_rec,
                                'y_label': y_label,
                                'y_units': y_units,
                                'y_order_rec': y_order_rec,
                                'x_label_a': x_label_a,
                                'x_units_a': x_units_a,
                                'x_order_rec_a': not x_order_rec,
                                'y_label_a': y_label_a,
                                'y_units_a': y_units_a,
                                'y_order_rec_a': not y_order_rec,
                                'x_alt': False,
                                'y_alt': False,
                                'Normalized': False}
            else:
                Map_2D.attrs = {'x_label': x_label,
                                'x_units': x_units,
                                'x_order_rec': x_order_rec,
                                'y_label': y_label,
                                'y_units': y_units,
                                'y_order_rec': y_order_rec,
                                'x_label_a': x_label_a,
                                'x_units_a': x_units_a,
                                'x_order_rec_a': not x_order_rec,
                                'y_label_a': y_label_a,
                                'y_units_a': y_units_a,
                                'y_order_rec_a': not y_order_rec,
                                'x_alt': False,
                                'y_alt': False,
                                'Normalized': False,
                                'DLD': self.DLD,
                                'z_label': z_label,
                                'z_units': z_units,
                                'z_order_rec': z_order_rec,
                                'z_label_a': z_label_a,
                                'z_units_a': z_units_a,
                                'z_order_rec_a': not z_order_rec,
                                'z_alt': False}

            try:
                check = []
                for i in ['x', 'y', 'z']:
                    try:
                        if Map_2D.attrs[f'{i}_units'] == 'eV':
                            check.append(i)
                        elif 'energy' in Map_2D.attrs[f'{i}_label']:
                            check.append(i)
                    except:
                        pass
                check = check[0]
                BE = loaded_map.variables[Map_2D.attrs[f'{check}_label_a']].values
                Map_2D.coords[Map_2D.attrs[f'{check}_label_a']] = (f'Dim_{check}', BE)
            except:
                pass

            self.Map_2D = Map_2D
            self.Map_2D_plot = self.Map_2D
            self.en_calib = '-'
        except FileNotFoundError:
            try:
                julia_start = timer()
                print('Loading Julia...')
                jl = Julia(runtime=config.jpath, compiled_modules=False)
                make_hist = jl.eval('''
                                    using FHist
                                    function make_hist(x,y,x_a,y_a)
                                        return bincounts(Hist2D((x_a,y_a), (x,y)))
                                    end
                                    ''')
                make_hist_3D = jl.eval('''
                                        using FHist
                                        function make_hist(x,y,z,x_a,y_a,z_a)
                                            return bincounts(Hist3D((x_a,y_a,z_a), (x,y,z)))
                                        end
                                        ''')
                rounding_jl = jl.eval('''
                                      function rounding_jl(x,y)
                                            return floor.(x./y).*y .+ ((x./y .- floor.(x./y)) .>= 0.5).*y
                                        end
                                    ''')
                use_julia = True
                julia_end = timer()
                print('***Julia-enabled mode***')
                if julia_end-julia_start > 1:
                    print(f'Julia loading time: {np.around(julia_end-julia_start, 2)} s')
            except:
                use_julia = False
                print('***All-Python mode***')

            if self.switch_3D is True and use_julia is False:
                raise Exception('***For 3D histogramming, please set up Julia-enabled mode***')

            start = timer()
            '''
            This part is supposed to filter artifact values
            in the energy domain.
            '''
            for j in ['DLD_energy', 'DLD_delay']:
                i = getattr(self, j)
                mean = np.mean(i)
                std = np.std(i)
                min_list = np.where(i < mean-3*std)
                if j == 'DLD_energy' and self.mono_mean > 100:
                    max_list = np.where(i > self.mono_mean + 50)
                else:
                    max_list = np.where(i > mean+3*std)
                del_list = np.append(min_list, max_list)
                if del_list.size != 0:
                    self.DLD_energy = np.delete(self.DLD_energy, del_list)
                    self.DLD_delay = np.delete(self.DLD_delay, del_list)
                    self.x = np.delete(self.x, del_list)
                    self.y = np.delete(self.y, del_list)
                    try:
                        self.BAM = np.delete(self.BAM, del_list)
                    except:
                        self.BAM = 0
                    if isinstance(self.GMD, int) is False:
                        self.GMD = np.delete(self.GMD, del_list)
                    if isinstance(self.mono, int) is False:
                        self.mono = np.delete(self.mono, del_list)
                    self.B_ID = np.delete(self.B_ID, del_list)
                    self.MB_ID = np.delete(self.MB_ID, del_list)
                    if isinstance(self.diode, int) is False:
                        self.diode = np.delete(self.diode, del_list)
            '''
            Picking Delay or MB_ID as the ordinate axis.
            '''

            DLD_energy = getattr(self, self.unit_dict[ordinate[0]][3])
            parameter = getattr(self, self.unit_dict[ordinate[1]][3])
            if self.switch_3D is True:
                z_parameter = getattr(self, self.unit_dict[ordinate[2]][3])                

            if use_julia is True:
                DLD_delay_r = np.array(rounding_jl(parameter, delay_step))
                DLD_energy_r = np.array(rounding_jl(DLD_energy, energy_step))
            else:
                DLD_delay_r = self.rounding(parameter, delay_step)
                DLD_energy_r = self.rounding(DLD_energy, energy_step)
                DLD_delay_r = np.around(DLD_delay_r,
                                        self.decimal_n(delay_step))
                DLD_energy_r = np.around(DLD_energy_r,
                                         self.decimal_n(energy_step))

            image_data_x = np.arange(DLD_energy_r.min(),
                                     DLD_energy_r.max()+energy_step,
                                     energy_step)
            image_data_x = np.around(image_data_x,
                                     self.decimal_n(energy_step))
            image_data_y = np.arange(DLD_delay_r.min(),
                                     DLD_delay_r.max()+delay_step,
                                     delay_step)
            image_data_y = np.around(image_data_y,
                                     self.decimal_n(delay_step))

            if self.switch_3D is True:
                if z_step >= 100:
                    z_step = (z_parameter.max()-z_parameter.min())/z_step
                    z_step = np.around(z_step, 3)
                image_data_z = np.arange(z_parameter.min(),
                                         z_parameter.max()+z_step,
                                         z_step)
                image_data_z = np.around(image_data_z,
                                         self.decimal_n(z_step))

            if use_julia is True:
                yedges = np.append(image_data_y,
                                   image_data_y[-1]+delay_step)
                yedges = yedges-0.5*delay_step
                xedges = np.append(image_data_x,
                                   image_data_x[-1]+energy_step)
                xedges = xedges-0.5*energy_step
                if self.switch_3D is False:
                    image_data = make_hist(xedges, yedges,
                                           DLD_energy_r, DLD_delay_r)
                    image_data = np.array(image_data).T
                else:
                    zedges = np.append(image_data_z,
                                       image_data_z[-1]+z_step)
                    zedges = zedges-0.5*z_step
                    image_data = make_hist_3D(xedges, yedges, zedges,
                                                 DLD_energy_r, DLD_delay_r,
                                                 z_parameter)
                    image_data = np.array(image_data).transpose(1,0,2)
                    # for i in range(image_data.shape[2]):
                    #     plt.imshow(image_data[:,:,i])
                    #     plt.show()
            else:
                image_data = []
                for i in image_data_y:
                    array_1 = DLD_energy_r[np.where(DLD_delay_r == i)]
                    line = []
                    array_1 = array_1.astype('f')
                    for j in image_data_x:
                        array_2 = np.where(array_1 == j)[0]
                        line.append(array_2.shape[0])
                    image_data.append(line)

            x_label = self.unit_dict[ordinate[0]][0]
            y_label = self.unit_dict[ordinate[1]][0]
            y_units = self.unit_dict[ordinate[1]][2]
            x_units = self.unit_dict[ordinate[0]][2]
            x_label_a = self.unit_dict[ordinate[0]][1]
            y_label_a = self.unit_dict[ordinate[1]][1]
            x_units_a = self.unit_dict[ordinate[0]][2]
            y_units_a = self.unit_dict[ordinate[1]][2]
            x_order_rec = False
            y_order_rec = False

            if ordinate[0] == 't':
                image_data_x_a = image_data_x
                image_data_x = 100 - image_data_x
            elif ordinate[1] == 't':
                image_data_y_a = image_data_y
                image_data_y = 100 - image_data_y
            try:
                if ordinate[2] == 't':
                    image_data_z_a = image_data_z
                    image_data_z = 100 - image_data_z
            except:
                pass

            # finding onset position
            # y = np.abs(np.gradient(np.sum(image_data, axis=0)))
            # mid = int(y.shape[0]/2)
            # y = y[:mid]
            # x_check = image_data_x_a[:mid][np.where(y>0.01*np.max(y))]
            # plt.plot(image_data_x_a[np.where(y>0.01*np.max(y))],y[np.where(y>0.01*np.max(y))], 'o')
            # plt.plot(image_data_x_a, np.abs(np.gradient(np.sum(image_data, axis=0))))
            # plt.show()
            # grad = np.gradient(x_check)
            # photons = np.mean(x_check[x_check<=x_check[np.argmax(grad)]])
            # photons = self.rounding(photons, energy_step)
            # onset = np.min(x_check[x_check>x_check[np.argmax(grad)]])
            # photons = np.around(photons, self.decimal_n(energy_step))
            # onset = np.around(onset, self.decimal_n(energy_step))

            #image_data_x_a = image_data_x_a - onset

            # BE = self.mono_mean - np.array(image_data_x) - 4.5
            # BE = np.around(self.rounding(BE, energy_step), self.decimal_n(energy_step))
            # image_data_x_a = BE
            
            if self.switch_3D is False:
                coords = {x_label: ("Dim_x", image_data_x),
                          y_label: ("Dim_y", image_data_y)}
    
                Map_2D = xr.DataArray(np.array(image_data),
                                      dims=["Dim_y", "Dim_x"],
                                      coords=coords)
            else:
                z_label = self.unit_dict[ordinate[2]][0]
                z_units = self.unit_dict[ordinate[2]][2]
                z_label_a = self.unit_dict[ordinate[2]][1] 
                z_units_a = self.unit_dict[ordinate[2]][2]
                z_order_rec = False

                coords = {x_label: ("Dim_x", image_data_x),
                          y_label: ("Dim_y", image_data_y),
                          z_label: ("Dim_z", image_data_z)}
    
                Map_2D = xr.DataArray(np.array(image_data),
                                      dims=["Dim_y", "Dim_x","Dim_z"],
                                      coords=coords)
                Map_2D.coords["Dim_z"] = Map_2D.coords[z_label]

            Map_2D.coords["Dim_y"] = Map_2D.coords[y_label]
            Map_2D.coords["Dim_x"] = Map_2D.coords[x_label]

            try:
                Map_2D.coords[x_label_a] = ('Dim_x', image_data_x_a)
            except:
                pass
            try:
                Map_2D.coords[y_label_a] = ('Dim_y', image_data_y_a)
            except:
                pass
            try:
                Map_2D.coords[z_label_a] = ('Dim_z', image_data_z_a)
            except:
                pass
            if self.switch_3D is False:
                Map_2D.attrs = {'x_label': x_label,
                                'x_units': x_units,
                                'x_order_rec': x_order_rec,
                                'y_label': y_label,
                                'y_units': y_units,
                                'y_order_rec': y_order_rec,
                                'x_label_a': x_label_a,
                                'x_units_a': x_units_a,
                                'x_order_rec_a': not x_order_rec,
                                'y_label_a': y_label_a,
                                'y_units_a': y_units_a,
                                'y_order_rec_a': not y_order_rec,
                                'x_alt': False,
                                'y_alt': False,
                                'Normalized': False,
                                'DLD': self.DLD}
            else:
                Map_2D.attrs = {'x_label': x_label,
                                'x_units': x_units,
                                'x_order_rec': x_order_rec,
                                'y_label': y_label,
                                'y_units': y_units,
                                'y_order_rec': y_order_rec,
                                'x_label_a': x_label_a,
                                'x_units_a': x_units_a,
                                'x_order_rec_a': not x_order_rec,
                                'y_label_a': y_label_a,
                                'y_units_a': y_units_a,
                                'y_order_rec_a': not y_order_rec,
                                'x_alt': False,
                                'y_alt': False,
                                'Normalized': False,
                                'DLD': self.DLD,
                                'z_label': z_label,
                                'z_units': z_units,
                                'z_order_rec': z_order_rec,
                                'z_label_a': z_label_a,
                                'z_units_a': z_units_a,
                                'z_order_rec_a': not z_order_rec,
                                'z_alt': False}
            Map_2D.name = 'Run ' + str(self.run_num)
            self.Map_2D = Map_2D
            self.Map_2D_plot = Map_2D
            self.y_step = self.Map_2D.coords['Dim_y'].values
            try:
                self.y_step = np.min(np.abs(np.gradient(self.y_step)))
            except:
                self.y_step = 1
            self.x_step = self.Map_2D.coords['Dim_x'].values
            self.x_step = np.gradient(self.x_step)[0]
            self.en_calib = '-'

            if save != 'off':  # and self.ordinate == 'delay':
                self.Map_2D.to_netcdf(save_path, engine="scipy")
                print('Delay-energy map saved as:')
                print(save_path)
            end = timer()
            print(f'Run {self.run_num} done')
            print(f'Elapsed time: {round(end-start, 1)} s')

    def create_dif_map(self):
        '''
        This method generates a difference map by averaging data before
        -0.25 ps and subtracting it from the delay-energy map.
        '''
        attrs = self.Map_2D_plot.attrs
        t_axis_step = self.Map_2D_plot.coords['Dim_y'].values
        try:
            t_axis_step = abs(np.median(np.gradient(t_axis_step)))
        except ValueError:
            t_axis_step = 1
        t_axis_step = int(-2.5*t_axis_step)
        norm = self.Map_2D_plot.loc[t_axis_step:].mean('Dim_y')
        self.Map_2D_dif = self.Map_2D_plot - norm
        self.Map_2D_dif.attrs = attrs

    def set_BE(self):
        '''
        Method for switching visualization to 'Binding energy'
        coordinate of 'Energy' dimension.
        '''
        check = []
        for i in ['x', 'y', 'z']:
            try:
                if self.Map_2D_plot.attrs[f'{i}_units'] == 'eV':
                    check.append(i)
                elif 'energy' in self.Map_2D_plot.attrs[f'{i}_label']:
                    check.append(i)
            except:
                pass
        try:
            check = check[0]
            if self.Map_2D_plot.attrs[f'{check}_alt'] is False:
                coord = self.Map_2D_plot.coords[self.Map_2D_plot.attrs[f'{check}_label_a']]
                self.Map_2D_plot.coords[f'Dim_{check}'] = coord
                self.Map_2D_plot.attrs[f'{check}_alt'] = True
        except:
            pass

    def axs_plot(self, axs):
        # Loading configs from json file.
        try:
            with open('config.json', 'r') as json_file:
                config = json.load(json_file)
        except FileNotFoundError:
            with open('packages/config.json', 'r') as json_file:
                config = json.load(json_file)
        config = json.dumps(config)
        config = json.loads(config,
                            object_hook=lambda d: SimpleNamespace(**d))
        '''
        Method for creating matplotlib axes for delay-energy map visualization.
        Uses the corresponding method from the create_batch_WESPE object.
        '''
        create_batch_WESPE.axs_plot(self, axs)

    @staticmethod
    def rounding(x, y):
        '''
        The function rounds energy and delay values to the closest
        values separated by the desired step.
        x - input value
        y - desired step
        '''
        result = np.floor(x/y)*y
        check = (x / y) - np.floor(x/y)
        result = result + (check >= 0.5)*y
        return result

    @staticmethod
    def decimal_n(x):
        '''
        Determines the number of decimal points.
        '''
        result = len(str(x)) - 2
        if isinstance(x, int):
            result = 0
        return result


class read_file_ALS(create_batch_WESPE):
    '''
    The object for storing data from individual hdf5 files.
    It is used further for creating create_batch_ALS objects.
    '''

    def __init__(self, file_full, DLD='DLD4Q'):
        '''
        Object initialization where reading out of data from hdf5 files occurs.
        '''
        self.is_static = False
        self.file_full = file_full
        self.file_folder = file_full.split(os.sep)[:-1]
        self.file_folder = f'{os.sep}'.join(self.file_folder)
        self.run_num_full = file_full.split(os.sep)[-1].split('_')[-1]
        self.date = str(self.run_num_full.split('-')[0])
        try:
            self.run_num = str(self.run_num_full.split('-')[1].replace('run', ''))
        except IndexError:
            self.run_num = self.run_num_full

        listdir = os.listdir(self.file_full)
        listdir_static = listdir.copy()
        h5_detected = True
        for i in listdir_static:
            if i.split('.')[-1] != 'h5':
                listdir.remove(i)

        if len(listdir) == 0:
            h5_detected = False
            listdir_static = os.listdir(f'{self.file_full+os.sep}'+'netCDF_maps')
            for i in listdir_static:
                line = '_'.join(i.split('_')[:-i.count('_')+1])+'.h5'
                listdir.append(line)

        ps_str_list = []
        ps_int_list = []
        self.listdir = listdir

        for i in listdir:
            try:
                ps_value = i.replace('.h5', '')
                ps_value = ps_value.split('_')[-1]
                ps_value = ps_value.replace('ps', '')
                ps_str_list.append(ps_value)
                ps_int_list.append(int(ps_value))
            except ValueError:
                ps_value = i.replace('.h5', '')
                ps_value = ps_value.split('-')[-1]
                ps_value = ps_value.replace('run', '')
                ps_str_list.append(ps_value)
                ps_int_list.append(int(ps_value))

        e_num_abs = True
        try:
            e_num = []
            f_size = []
            for counter, i in enumerate(listdir):
                iter_start = timer()
                file_path = self.file_full + os.sep + i
                size = os.path.getsize(file_path)
                f_size.append(size)
                if e_num_abs is True:
                    with h5py.File(file_path, 'r') as f:
                        try:
                            energy = f.get('/x')[:]
                            self.binned = False
                            e_num.append(energy.shape[0])
                            if counter == 0:
                                weight = os.path.getsize(file_path)/energy.shape[0]
                        except:
                            binned_data = f.get('TRXPS_binned')[:]
                            self.binned = True
                            e_num.append(np.sum(binned_data))
                            if counter == 0:
                                weight = os.path.getsize(file_path)/np.sum(binned_data)
                else:
                    e_num.append(os.path.getsize(file_path)/weight)
                iter_end = timer()
                if (iter_end-iter_start)*len(listdir) > 0.5:
                    e_num_abs = False
            self.e_num = int(sum(e_num))
            self.f_size = int(sum(f_size))
        except:
            e_num = [0, 1]
            self.e_num = int(sum(e_num))
            f_size = [0, 1]
            self.f_size = int(sum(f_size))

        file_dir = self.file_full+os.sep+'netCDF_maps'
        try:
            listdir_nc = os.listdir(file_dir)
            listdir_nc_static = listdir_nc.copy()
            for i in listdir_nc_static:
                if i.split('.')[-1] != 'nc':
                    listdir_nc.remove(i)
        except:
            listdir_nc = []

        calib_path = self.file_full + os.sep + 'Energy calibration.txt'
        if os.path.isfile(calib_path):
            self.en_calib = '+'
        else:
            self.en_calib = '-'

        calib_path = self.file_full + os.sep + 'Transmission function.txt'
        if os.path.isfile(calib_path):
            self.tr_calib = '+'
        else:
            self.tr_calib = '-'

        self.DLD = DLD
        self.KE = 0
        self.PE = 0
        self.mono_mean = 0

        self.info = []

        if h5_detected is True:
            self.info.append(f'Scan name: {self.run_num_full} / Electrons detected: {self.e_num:,}')
            self.info.append(f'Uploaded h5 files: {len(self.listdir)} / Min PS value: {min(ps_int_list)} / Max PS value: {max(ps_int_list)}')
            self.info.append(f'Electrons detected per file - Min: {int(min(e_num)):,} / Max: {int(max(e_num)):,}')
            self.info.append(f'Size per file - Min: {int(min(f_size)/(1024**2))} MB / Max: {int(max(f_size)/(1024**2))} MB / {np.around(100*(max(f_size)-min(f_size))/np.mean(f_size)):.0f} % variation')
        else:
            self.info.append(f'Scan name: {self.run_num_full} / Electrons detected: None')
            self.info.append(f'Uploaded h5 files: 0 / Min PS value: {min(ps_int_list)} / Max PS value: {max(ps_int_list)}')
            self.info.append(f'Electrons detected per file - Min: None / Max: None')
            self.info.append(f'Size per file - Min: 0 MB / Max: 0 MB / 0 % variation') 
        if np.around(100*len(listdir_nc)/len(self.listdir)) > 100:
            self.info.append(f'Processed (netCDF) files: {len(listdir_nc)} (100 % done)')
        else:
            self.info.append(f'Processed (netCDF) files: {len(listdir_nc)} ({np.around(100*len(listdir_nc)/len(self.listdir)):.0f} % done)')
        self.info.append(f'Energy calibration: {self.en_calib}  /  Transmission function: {self.tr_calib}')
        self.info = '\n'.join(self.info)

        self.listdir = listdir
        self.ps_str_list = ps_str_list
        self.ps_int_list = ps_int_list

    def create_map(self, ordinate='delay', bunch_sel=3,
                   save='on', bunches='all', DLD_t_res=None):
        # Loading configs from json file.
        try:
            with open('config.json', 'r') as json_file:
                config = json.load(json_file)
        except FileNotFoundError:
            with open('packages/config.json', 'r') as json_file:
                config = json.load(json_file)
        config = json.dumps(config)
        config = json.loads(config,
                            object_hook=lambda d: SimpleNamespace(**d))

        ALS_mode = None
        MB_filter = None
        if DLD_t_res is None:
            DLD_t_res = config.DLD_t_res
        en_ch_min = config.en_ch_min
        en_ch_max = config.en_ch_max
        measured_periods = config.measured_periods
        self.bunches = bunches
        self.bunch_sel = bunch_sel
        self.ordinate = ordinate
        listdir = self.listdir
        ps_str_list = self.ps_str_list
        ps_int_list = self.ps_int_list
        file_full = self.file_full
        rounding = self.rounding
        decimal_n = self.decimal_n
        date = self.date
        run_num = self.run_num

        data_dict = {}
        bunches_cutoff = np.inf
        first_file = True

        start = timer()
        storage = []
        for counter_g, i in enumerate(listdir):
            test_lag_1 = timer()
            nc_name_static = i.split(os.sep)[-1].replace('.h5', '')
            nc_name = nc_name_static + '_' + str(bunches) + '_bunches'
            nc_name = nc_name + '_' + f'{DLD_t_res}' + '_bins.nc'
            nc_name_all = nc_name_static + '_' + 'all' + '_bunches'
            nc_name_all = nc_name_all + '_' + f'{DLD_t_res}' + '_bins.nc'
            save_path_static = file_full + os.sep + 'netCDF_maps'
            if os.path.isdir(save_path_static) is False:
                os.mkdir(save_path_static)
            save_path = save_path_static + os.sep + f"{nc_name}"
            save_path_all = save_path_static + os.sep + f"{nc_name_all}"
            try:
                if save != 'on':
                    raise FileNotFoundError
                try:
                    with xr.open_dataset(save_path_all) as ds:
                        loaded_map = ds
                except FileNotFoundError:
                    with xr.open_dataset(save_path) as ds:
                        loaded_map = ds

                for i in list(loaded_map.variables.keys()):
                    if 'Run' in i:
                        data_name = i
                        try:
                            DLD_t_res = loaded_map[i].attrs['DLD_t_res']
                        except:
                            DLD_t_res = None

                image_data = loaded_map.variables[data_name].values
                try:
                    image_data_y = loaded_map.variables['Delay stage values'].values
                except:
                    image_data_y = loaded_map.variables['Bunch ID'].values
                image_data_x = loaded_map.variables['Kinetic energy'].values
                coords = {'Bunch ID': ('Dim_y', image_data_y),
                          "Kinetic energy": ('Dim_x', image_data_x)}
                Map_2D = xr.DataArray(np.array(image_data),
                                      dims=['Dim_y', 'Dim_x'],
                                      coords=coords)
                Map_2D.name = data_name
                BE = loaded_map.variables['Binding energy'].values
                Map_2D.coords['Binding energy'] = ('Dim_x', BE)

                Map_2D.coords['Dim_x'] = Map_2D.coords['Kinetic energy']
                Map_2D.coords['Dim_y'] = Map_2D.coords['Bunch ID']

                if self.ordinate == 'MB_ID':
                    y_label = 'Bunch ID'
                    y_units = 'units'
                    y_label_a = 'Bunch ID reversed'
                    y_units_a = 'units'
                else:
                    y_label = 'Delay stage values'
                    y_units = 'ps'
                    y_label_a = 'Delay'
                    y_units_a = 'ps'
                x_label = 'Kinetic energy'
                x_units = 'eV'
                x_label_a = 'Binding energy'
                x_units_a = 'eV'
                x_order_rec = False
                y_order_rec = False

                Map_2D.attrs = {'x_label': x_label,
                                'x_units': x_units,
                                'x_order_rec': x_order_rec,
                                'y_label': y_label,
                                'y_units': y_units,
                                'y_order_rec': y_order_rec,
                                'x_label_a': x_label_a,
                                'x_units_a': x_units_a,
                                'x_order_rec_a': not x_order_rec,
                                'y_label_a': y_label_a,
                                'y_units_a': y_units_a,
                                'y_order_rec_a': not y_order_rec,
                                'x_alt': False,
                                'y_alt': False,
                                'Normalized': False,
                                'DLD_t_res': DLD_t_res}

                if isinstance(bunches, int):
                    Map_2D = Map_2D.loc[0:bunches]

                Map_2D_plot = Map_2D
                x_step = Map_2D.coords['Dim_x'].values
                x_step = np.gradient(x_step)[0]
                if counter_g == 0:
                    print(f'Run {self.run_num_full} processing')
                    print('***NetCDF pre-processed data loaded***')

            except FileNotFoundError:
                if counter_g == 0:
                    print(f'Run {self.run_num_full} processing')
                    print('***Start of raw data processing***')
                try:
                    if self.binned is True:
                        raise Exception
                    julia_start = timer()
                    if counter_g == 0:
                        print('Loading Julia...')
                    jl = Julia(runtime=config.jpath, compiled_modules=False)
                    make_hist = jl.eval('''
                                        using FHist
                                        function make_hist(x,y,x_a,y_a)
                                            return bincounts(Hist2D((x_a,y_a), (x,y)))
                                        end
                                        ''')
                    rounding_jl = jl.eval('''
                                          function rounding_jl(x,y)
                                                return floor.(x./y).*y .+ ((x./y .- floor.(x./y)) .>= 0.5).*y
                                            end
                                        ''')
                    use_julia = True
                    julia_end = timer()
                    if counter_g == 0:
                        print('***Julia-enabled mode***')
                    if julia_end-julia_start > 1:
                        print(f'Julia loading time: {np.around(julia_end-julia_start, 2)} s')
                except:
                    use_julia = False
                    if counter_g == 0:
                        print('***All-Python mode***')

                file_path = file_full + os.sep + i
                try:
                    with h5py.File(file_path, 'r') as f:
                        try:
                            energy = f.get('/x')[:]
                            delay = f.get('/t')[:]
                            binned = False
                        except:
                            binned_data = np.array(f.get('TRXPS_binned'))[::-1]
                            binned_data = np.rot90(binned_data,
                                                   k=3)
                            binned = True
                            en_ch_max = binned_data.shape[1] - 1
                except OSError:
                    print('Can not open h5 file:')
                    print(file_path)
                    continue
                
                if binned is False:
                    min_list = np.where(energy < 0)
                    max_list = np.where(energy > 150)
                    del_list = np.append(min_list, max_list)
                    energy = np.delete(energy, del_list)
                    delay = np.delete(delay, del_list)
    
                    if bunches_cutoff != np.inf:
                        min_list = np.where(delay < 0)
                        max_list = np.where(delay > bunches_cutoff)
                        del_list = np.append(min_list, max_list)
                        y_step_trigger = False
                        if len(del_list) > 0:
                            y_step_trigger = True
                            energy = np.delete(energy, del_list)
                            delay = np.delete(delay, del_list)
    
                    if delay.shape[0] == 0 or energy.shape[0] == 0:
                        print('Energy or time arrays are missing in:')
                        print(file_path)
                        continue
    
                    if np.max(delay) > 1e10:  # some runs have 10e16 values
                        check = np.median(delay)
                    else:
                        check = np.mean(delay)
                    min_list = np.where(delay < check-2*check)
                    max_list = np.where(delay > check+2*check)
                    del_list = np.append(min_list, max_list)
                    if del_list.size != 0:
                        energy = np.delete(energy, del_list)
                        delay = np.delete(delay, del_list)

                x_step = 1
                should_restart = True
                while should_restart:
                    if binned is False:
                        if ALS_mode == '2B' or ALS_mode is None:
                            y_step = np.max(delay)/(DLD_t_res*0.1)
                            y_step = y_step*12/config.measured_periods
                            y_step = rounding(y_step, 1000)
                            if first_file is False and isinstance(bunches, int):
                                y_step = rounding(y_step*2*config.measured_periods/bunches, 1000)
                        else:
                            y_step = np.max(delay)/DLD_t_res
                            y_step = y_step*12/config.measured_periods
                            y_step = rounding(y_step, 10)
                            if first_file is False and isinstance(bunches, int):
                                y_step = rounding(y_step*config.measured_periods/bunches, 10)
    
                        DLD_energy_r = energy
                        if use_julia is False:
                            DLD_delay_r = rounding(delay, y_step)
                            DLD_delay_r = np.around(DLD_delay_r,
                                                    decimal_n(y_step))
                            DLD_energy_r = np.around(DLD_energy_r,
                                                     decimal_n(x_step))
                        else:
                            DLD_delay_r = np.array(rounding_jl(delay, y_step))
    
                        image_data_x = np.arange(en_ch_min,
                                                 en_ch_max+x_step,
                                                 x_step)
                        image_data_y = np.arange(DLD_delay_r.min(),
                                                 DLD_delay_r.max()+y_step,
                                                 y_step)
    
                        if MB_filter is not None and use_julia is False and binned is False:
                            if image_data_y.shape[0] <= np.max(MB_filter):
                                MB_filter_f = np.array(MB_filter)[np.where(np.array(MB_filter) < image_data_y.shape[0])[0]]
                            else:
                                MB_filter_f = MB_filter
                            image_data_y = np.delete(image_data_y, MB_filter_f)
    
                        if use_julia is True:
                            yedges = np.append(image_data_y,
                                               image_data_y[-1]+y_step)
                            yedges = yedges-0.5*y_step
                            xedges = np.append(image_data_x,
                                               image_data_x[-1]+x_step)
                            xedges = xedges-0.5*x_step
                            image_data = make_hist(xedges, yedges,
                                                   DLD_energy_r, DLD_delay_r)
                            image_data = np.array(image_data).T
                        else:
                            image_data = []
                            for i in image_data_y:
                                array_1 = DLD_energy_r[np.where(DLD_delay_r == i)]
                                line = []
                                array_1 = array_1.astype('f')
                                for j in image_data_x:
                                    array_2 = np.where(array_1 == j)[0]
                                    line.append(array_2.shape[0])
                                image_data.append(line)
    
                        if use_julia is True:
                            energy_ROI = np.sum(np.array(image_data), axis=0)
                            energy_ROI_i = np.where(energy_ROI>np.median(energy_ROI))[0]                        
                            image_data_check = np.array(image_data)[:, energy_ROI_i]
                            image_data_1D = np.sum(image_data_check, axis=1)
                        else:
                            image_data_check = np.array(image_data)
                            image_data_1D = np.sum(image_data_check, axis=1)
    
                        # check for intensity artifacts
                        if ALS_mode == 'MB':
                            check = np.where(image_data_1D > np.median(image_data_1D)*6)
                            image_data_1D[check] = 0   
                    else:
                        if DLD_t_res != 0:
                            n_bins = DLD_t_res
                            bin_size = binned_data.shape[0]//n_bins
    
                            new_binned_data = np.pad(binned_data,
                                                   [(0, (bin_size+1)*n_bins-binned_data.shape[0]),
                                                    (0, 0)], mode='constant', constant_values=0)
    
                            # new_binned_data = binned_data[:bin_size*n_bins]
                            new_binned_data = new_binned_data.reshape(n_bins,
                                                                      bin_size+1,
                                                                      binned_data.shape[1])
                            new_binned_data = new_binned_data.sum(axis=1)
                        
                        else:
                            new_binned_data = binned_data
                            
                        image_data = new_binned_data
                        energy_ROI = np.sum(np.array(image_data), axis=0)
                        energy_ROI_i = np.where(energy_ROI>np.max(energy_ROI)/2)[0]                        
                        image_data_check = np.array(image_data)[:, energy_ROI_i]
                        image_data_1D = np.sum(image_data_check, axis=1)

                        # image_data = new_binned_data
                        # image_data_check = new_binned_data
                        # image_data_1D = np.sum(image_data, axis=1)
                        image_data_x = np.arange(en_ch_min,
                                                 en_ch_max+x_step,
                                                 x_step)
                        # plt.plot(np.sum(image_data, axis=1)[30000:])
                        # plt.show()

                    if ALS_mode is None:
                        if np.median(image_data_1D) < np.max(image_data_1D)*0.3:
                            ALS_mode = '2B'
                            print('***Data is processed in 2B mode***')
                        else:
                            ALS_mode = 'MB'
                            print('***Data is processed in MB mode***')
                            continue

                    state = 'off'
                    bunch = 1
                    if binned is False:
                        threshold = np.max(image_data_1D)/10
                    else:
                        threshold = np.median(image_data_1D)/4.5
                    Bunch_d = {}
                    Bunch_d[bunch] = []
                    for counter, i in enumerate(image_data_1D):
                        if i <= threshold and state == 'on':
                            state = 'off'
                            bunch += 1
                            Bunch_d[bunch] = []
                        elif i <= threshold and state == 'off':
                            pass
                        elif i > threshold and state == 'off':
                            state = 'on'
                            Bunch_d[bunch].append(counter)
                        elif i > threshold and state == 'on':
                            Bunch_d[bunch].append(counter)

                    for i in list(Bunch_d.keys()):
                        if len(Bunch_d[i]) == 0:
                            del Bunch_d[i]

                    if len(list(Bunch_d.keys())) < 1:
                        print('Zero bunches detected!')
                        print(f'Skipping: {file_path}')
                        break

                    image_data_b = []
                    image_data_y_b = []
                    image_data_x_b = image_data_x

                    Bunch_d_f = {}
                    Bunch_d_MB = {}
                    
                    # merging bunches which are too close
                    if ALS_mode == 'MB':
                        check = [item for sublist in [*Bunch_d.values()] for item in sublist]
                        check = np.ediff1d(check)
                        Bunch_d_m = {}
                        counter_r = 1
                        counter_f = 1
                        Bunch_d_m[counter_r] = [counter_f]
                        for i in check:
                            if i == 2:
                                counter_f += 1
                                Bunch_d_m[counter_r] += [counter_f]
                            if i > 2:
                                counter_r += 1
                                counter_f += 1
                                Bunch_d_m[counter_r] = [counter_f]

                        Bunch_d_n = {}
                        for key, value in Bunch_d_m.items():
                            Bunch_d_n[key] = Bunch_d[Bunch_d_m[key][0]]
                            if len(Bunch_d_m[key]) > 1:
                                for j in range(1, len(Bunch_d_m[key])):
                                    Bunch_d_n[key] = Bunch_d_n[key] + Bunch_d[Bunch_d_m[key][j]]

                        Bunch_d = Bunch_d_n
                    
                    if binned is True:
                        length_list = []
                        for i in list(Bunch_d.keys()):
                            length_list.append(len(Bunch_d[i]))
                        length_lim_max = np.histogram(length_list, bins=150)[-1][4]
                        length_lim_min = 0
                        length_list_f = np.array(length_list)[np.where(length_list<length_lim_max)[0]]
                        if np.max(length_list_f)/np.min(length_list_f) > 3:
                            length_lim_min = (np.max(length_list_f)-np.min(length_list_f))*0.5 
                        length_list_f = np.array(length_list_f)[np.where(length_list_f>length_lim_min)[0]]
                        if len(length_list_f) != measured_periods:
                            length_lim_min = np.median(length_list_f)*0.8
                            length_lim_max = np.median(length_list_f)*1.2
                            # print(length_list_f)
                            # print(length_lim_max)
                            # print(length_lim_min)
                        

                    if ALS_mode == 'MB':
                        for i in list(Bunch_d.keys()):
                            image_data_y_b.append(i)
                            check = np.array(image_data_check)[Bunch_d[i]]
                            check = np.sum(np.max(check, axis=0))
                            line = np.array(image_data[Bunch_d[i][0]])
                            if len(Bunch_d[i]) > 1:
                                for j in Bunch_d[i][1:]:
                                    line = line + np.array(image_data[j])
                                    
                            if binned is False:
                                if check < np.max(image_data_1D)*0.4:
                                    image_data_b.append(list(line))
                                    Bunch_d_f[i] = Bunch_d[i]
                                else:
                                    if MB_filter is None and use_julia is False:
                                        Bunch_d_MB[i] = Bunch_d[i]
                            else:
                                if (len(Bunch_d[i]) < length_lim_max) and (len(Bunch_d[i]) > length_lim_min):
                                    image_data_b.append(list(line))
                                    Bunch_d_f[i] = Bunch_d[i]
                                else:
                                    if MB_filter is None and use_julia is False:
                                        Bunch_d_MB[i] = Bunch_d[i]
                        image_data_y_b = np.arange(len(image_data_b))+1
                    else:
                        for i in list(Bunch_d.keys()):
                            image_data_y_b.append(i)
                            line = np.array(image_data[Bunch_d[i][0]])
                            if len(Bunch_d[i]) > 1:
                                for j in Bunch_d[i][1:]:
                                    line = line + np.array(image_data[j])
                            image_data_b.append(list(line))
                        Bunch_d_f = Bunch_d

                    if MB_filter is None and ALS_mode == 'MB' and use_julia is False and binned is False:
                        pads = []
                        for value in Bunch_d_MB.values():
                            if int(len(value)/100) == 0:
                                pad = int(math.ceil(len(value)/100)*5)
                            else:
                                pad = int(math.floor(len(value)/100)*5)
                            pads.append(pad)
                        pad = max(pads)
                        MB_filter = [item for sublist in [*Bunch_d_MB.values()] for item in sublist[pad:-pad]]

                    i = 1
                    Bunch_d = {}
                    for key, value in Bunch_d_f.items():
                        Bunch_d[i] = value
                        i += 1

                    # fig, ax = plt.subplots(figsize=(160, 20))
                    # ax.plot(image_data_1D)
                    # fig.suptitle(f'File {nc_name_static} - {max(Bunch_d.keys())} bunches detected', fontsize=150)
                    # for key, value in Bunch_d.items():
                    #     ax.plot(value, np.array(image_data_1D)[value],
                    #             'X', markersize=25, linewidth=1)
                        
                    # # ax.axhline(threshold)
                    # plt.show()
                    # plt.clf()
                    # plt.close()

                    if isinstance(bunches, int) and first_file is False:
                        if ALS_mode == 'MB':
                            if bunches < measured_periods:
                                measured_periods = bunches
                        elif ALS_mode == '2B':
                            if bunches < measured_periods*2:
                                measured_periods = bunches/2

                    if max(Bunch_d.keys()) == measured_periods and ALS_mode == 'MB':
                        should_restart = False
                    elif max(Bunch_d.keys()) > measured_periods and ALS_mode == 'MB':
                        should_restart = False
                        if __name__ == "__main__":
                            fig, ax = plt.subplots(figsize=(160, 20))
                            ax.plot(image_data_1D)
                            fig.suptitle(f'File {nc_name_static} - {max(Bunch_d.keys())} bunches detected', fontsize=150)
                            for key, value in Bunch_d.items():
                                ax.plot(value, np.array(image_data_1D)[value],
                                        'X', markersize=25, linewidth=1)
                            plt.show()
                            plt.clf()
                            plt.close()
                        print(f'***File {nc_name_static} - {max(Bunch_d.keys())} bunches detected***')
                    elif max(Bunch_d.keys()) == measured_periods*2 and ALS_mode == '2B':
                        should_restart = False
                    elif max(Bunch_d.keys()) != measured_periods*2 and ALS_mode == '2B':
                        should_restart = False
                        if __name__ == "__main__":
                            fig, ax = plt.subplots(figsize=(160, 20))
                            ax.plot(image_data_1D)
                            fig.suptitle(f'File {nc_name_static} - {max(Bunch_d.keys())} bunches detected', fontsize=150)
                            for key, value in Bunch_d_f.items():
                                ax.plot(value, np.array(image_data_1D)[value],
                                        'X', markersize=25, linewidth=1)
                            plt.show()
                            plt.clf()
                            plt.close()
                        print(f'***File {nc_name_static} - {max(Bunch_d.keys())} bunches detected***')
                    else:
                        if DLD_t_res == 0:
                            DLD_t_res += 20000
                        else:
                            DLD_t_res += 500
                        MB_filter = None
                        if __name__ == "__main__":
                            fig, ax = plt.subplots(figsize=(160, 20))
                            ax.plot(image_data_1D)
                            fig.suptitle(f'File {nc_name_static} - {max(Bunch_d.keys())} bunches detected, time resolution was increased to {DLD_t_res} bins', fontsize = 150)
                            for key, value in Bunch_d_f.items():
                                ax.plot(value, np.array(image_data_1D)[value],
                                        'X', markersize=25, linewidth=1)
                            # ax.axhline(threshold)
                            plt.show()
                            plt.clf()
                            plt.close()
                        print(f'***File {nc_name_static} - {max(Bunch_d.keys())} bunches detected, time resolution was increased to {DLD_t_res} bins***')

                if len(list(Bunch_d.keys())) < 1:  # in case of empty files
                    continue

                if isinstance(bunches, int) and bunches_cutoff == np.inf:
                    try:
                        bunches_cutoff = np.max(Bunch_d[bunches]) + 5
                        bunches_cutoff = image_data_y[bunches_cutoff]
                    except KeyError:
                        pass

                if first_file is True:
                    first_file = False

                x_label = 'Kinetic energy'
                y_label = 'Bunch ID'
                x_units = 'eV'
                y_units = 'units'
                x_label_a = 'Binding energy'
                y_label_a = 'Delay'
                x_units_a = 'eV'
                y_units_a = 'ps'
                x_order_rec = False
                y_order_rec = False

                coords = {x_label: ("Dim_x", image_data_x_b),
                          y_label: ("Dim_y", image_data_y_b)}

                Map_2D = xr.DataArray(np.array(image_data_b),
                                      dims=["Dim_y", "Dim_x"],
                                      coords=coords)
                Map_2D.coords["Dim_y"] = Map_2D.coords[y_label]
                Map_2D.coords["Dim_x"] = Map_2D.coords[x_label]

                BE = 1000 - np.array(image_data_x_b)
                image_data_x_b_a = BE

                try:
                    Map_2D.coords[x_label_a] = ('Dim_x', image_data_x_b_a)
                except:
                    pass
                try:
                    Map_2D.coords[y_label_a] = ('Dim_y', image_data_y_b_a)
                except:
                    pass
                Map_2D.attrs = {'x_label': x_label,
                                'x_units': x_units,
                                'x_order_rec': x_order_rec,
                                'y_label': y_label,
                                'y_units': y_units,
                                'y_order_rec': y_order_rec,
                                'x_label_a': x_label_a,
                                'x_units_a': x_units_a,
                                'x_order_rec_a': not x_order_rec,
                                'y_label_a': y_label_a,
                                'y_units_a': y_units_a,
                                'y_order_rec_a': not y_order_rec,
                                'x_alt': False,
                                'y_alt': False,
                                'Normalized': False,
                                'DLD_t_res': DLD_t_res}
                Map_2D.name = 'Run ' + '_' + date + '_' + run_num + '_' + ps_str_list[counter_g]

                if isinstance(bunches, int):
                    Map_2D = Map_2D.loc[0:bunches]

                Map_2D_plot = Map_2D

                if save != 'off':
                    Map_2D.to_netcdf(save_path, engine='scipy')
                    print('Delay-energy map saved as:')
                    print(save_path)

                test_lag_2 = timer()
                progress_t = np.around((counter_g+1)*100/len(listdir), 2)
                storage.append(np.around(test_lag_2-test_lag_1, 2))
                if counter_g > 0 and len(storage) > 1:
                    sec_left = (len(listdir)-counter_g-1)*np.mean(storage[1:])
                    if sec_left > 3600:
                        s_format = '%Hh%Mm%Ss'
                    else:
                        s_format = '%Mm %Ss'
                    sec_left = timedelta(seconds=sec_left)
                    sec_left = datetime(2000, 1, 1) + sec_left
                    sec_left = sec_left.strftime(s_format)
                    print(f'File {nc_name_static} done')
                    print(f'Progress: {progress_t} % | Time left: {sec_left}')
                else:
                    print(f'File {nc_name_static} done')
                    print(f'Progress: {progress_t} %')

            data_dict[ps_int_list[counter_g]] = Map_2D

        '''Checking data homogeneity'''
        check = []
        for i in data_dict.values():
            check.append([i.sum(dim='Dim_y')])
        try:
            check = np.concatenate(check, axis=0)
        except:
            check = [[0]]
        check = np.sum(check, axis=1)
        diff = np.abs(np.diff(check))
        check = 100*(np.max(diff)-np.min(diff))/np.median(check)
        self.check_h = check
        '''Checking file size homogeneity'''
        check = []
        for counter, i in enumerate(listdir):
            file_path = self.file_full + os.sep + i
            try:
                check.append(os.path.getsize(file_path))
            except:
                pass
        diff = np.abs(np.diff(check))
        try:
            check = 100*(np.max(diff)-np.min(diff))/np.median(check)
        except:
            check = 0
        self.check_s = check

        end = timer()
        print(f'Elapsed time: {round(end-start, 1)} s')
        data_dict_sorted = {}
        for i in sorted(data_dict.keys()):
            data_dict_sorted[i] = data_dict[i]
        self.data_dict = data_dict_sorted
        '''
        Method for creation of delay-energy map from the data loaded at
        initialization.
        x_step and y_step determine the bin size for
        'Dim_x' and 'Delay dimensions'
        '''
        if self.ordinate == 'MB_ID':
            attrs = list(self.data_dict.values())[0].attrs
            for counter, i in enumerate(list(self.data_dict.values())):
                if counter == 0:
                    total_map = i
                else:
                    total_map = total_map + i
            total_map.attrs = attrs
        elif self.ordinate == 'delay':
            concat_list = []
            for i in list(self.data_dict.keys()):
                Data = self.data_dict[i]
                try:
                    bunch_sel = int(bunch_sel)
                    Data = Data.sel(Dim_y=bunch_sel, method="nearest")
                    # Data = Data.loc[bunch_sel]
                    Data = Data.drop_vars('Dim_y')
                    Data = Data.drop_vars('Bunch ID')
                except ValueError:
                    attrs = Data.attrs
                    bunch_sel_1 = int(bunch_sel.split(',')[0])
                    bunch_sel_2 = int(bunch_sel.split(',')[-1])
                    # Data = Data.sel(Delay=slice(bunch_sel_1, bunch_sel_2))
                    Data = Data.loc[slice(bunch_sel_1, bunch_sel_2)]
                    Data = Data.mean('Dim_y')
                    Data.attrs = attrs
                Data = Data.expand_dims(dim={'Dim_y': [i]}, axis=0)
                concat_list.append(Data)
            total_map = xr.concat(concat_list, dim='Dim_y', coords="minimal",
                                  compat='override')
            total_map.coords['Phase shifter values'] = total_map.coords['Dim_y']
            shape = total_map.coords['Dim_y'].values.shape[0]
            total_map.coords['Delay index'] = ('Dim_y', np.arange(shape))
            total_map.attrs['y_label'] = 'Phase shifter values'
            total_map.attrs['y_units'] = 'ps'

        if np.min(total_map.values.shape) == 0:
            total_map.attrs['Merge successful'] = False

        self.Map_2D = total_map.fillna(0)
        en_calib_path = file_full + os.sep + 'Energy calibration.txt'
        try:
            with open(en_calib_path, 'r') as open_file:
                en_calib = open_file.readlines()[0].split(',')
                en_ch = self.Map_2D.coords['Kinetic energy'].values
                self.Map_2D.coords['Energy channel'] = ('Dim_x',
                                                        en_ch)
                if en_calib[0] == 'KE':
                    KE = self.Map_2D.coords['Energy channel'] * float(en_calib[1])
                    KE = KE + float(en_calib[2])
                    BE = float(en_calib[3]) - KE - float(en_calib[4])
                elif en_calib[0] == 'BE':
                    BE = self.Map_2D.coords['Energy channel'] * float(en_calib[1])
                    BE = BE + float(en_calib[2])
                    KE = float(en_calib[3]) - BE - float(en_calib[4])
                self.Map_2D.coords['Kinetic energy'].values = KE.values
                self.Map_2D.coords['Binding energy'].values = BE.values
                if Map_2D_plot.attrs['x_alt'] is False:
                    state = 'Kinetic energy'
                else:
                    state = 'Binding energy'
                self.Map_2D.coords['Dim_x'] = self.Map_2D.coords[state]
        except FileNotFoundError:
            print('No energy calibration file for:')
            print(f'{file_full}')
        try:
            tr_calib_path = file_full + os.sep + 'Transmission function.txt'
            numpy_a = np.loadtxt(tr_calib_path)
            numpy_a = np.rot90(numpy_a)
            data_y = numpy_a[0]
            data_x = numpy_a[1]
            en_step = np.gradient(data_x).mean()
            if en_step < 0:
                data_y = np.flip(data_y)
                data_x = np.flip(data_x)
            data_y = data_y/np.max(data_y)
            data_y[data_y < 0.05] = np.nan
            mean = np.nanmean(data_y)
            data_y = data_y/mean
            data_y[np.isnan(data_y)] = 1
            attrs = self.Map_2D.attrs
            self.Map_2D = self.Map_2D / data_y
            self.Map_2D.attrs = attrs
        except:
            print('No transmission calibration file for:')
            print(f'{file_full}')
        self.Map_2D_plot = self.Map_2D
        self.y_step = self.Map_2D.coords['Dim_y'].values
        self.y_step = np.min(np.abs(np.gradient(self.y_step)))
        self.x_step = self.Map_2D.coords['Dim_x'].values
        self.x_step = np.gradient(self.x_step)[0]

    def erase_nc(self, check_mode=True):
        file_dir = self.file_full+os.sep+'netCDF_maps'
        listdir = os.listdir(file_dir)
        listdir_static = listdir.copy()
        for i in listdir_static:
            if i.split('.')[-1] != 'nc':
                listdir.remove(i)

        for file in listdir:
            try:
                file_full = file_dir + os.sep + file
                if check_mode is False:
                    os.remove(file_full)
                    print('File removed:')
                else:
                    print('File would be removed:')
                print(file_full)
            except Exception as err:
                print('The file can not be removed:')
                print(file_full)
                print('Full error message:')
                print(err)


class read_file_CasaXPS(create_batch_WESPE):
    def __init__(self, file_dir, energy_axis='BE', bg_sub='off'):
        self.info = ''
        self.file_dir = file_dir
        self.file_full = file_dir
        self.file_name = self.file_full.split(os.sep)[-1]
        self.energy_axis = energy_axis

        listdir = sorted(os.listdir(self.file_dir))
        listdir_static = listdir.copy()
        for i in listdir_static:
            if (i.split('.')[-1]).lower() != 'txt':
                listdir.remove(i)

        file_bin = []
        for i in listdir:
            file_full = self.file_dir + os.sep + i
            file = CasaXPS_file(file_full, energy_axis=self.energy_axis)
            if bg_sub == 'on':
                file.bg_sub()
            file_bin.append(file)

        self.file_bin = file_bin

        x_parameter_list = []
        hv_list = []
        for i in file_bin:
            x_parameter = i.data_x
            x_parameter_list.append(x_parameter)
            hv_list.append(i.hv)
        self.hv_list = hv_list

        try:
            x_parameter_check = abs(np.around(np.mean(np.gradient(x_parameter_list,
                                                                  axis=0)), 2))
        except:
            x_parameter_check = 1

        try:
            x_parameter_step = abs(np.around(np.mean(np.gradient(x_parameter_list,
                                                                 axis=1)), 2))
        except:
            x_parameter_step = []
            for i in x_parameter_list:
                x_parameter_step_i = np.around(np.mean(np.gradient(i)), 2)
                x_parameter_step.append(x_parameter_step_i)
            x_parameter_step = abs((np.around(np.max(x_parameter_step), 2)))

        self.x_parameter_list = x_parameter_list

        y_parameter_list = []
        for i in file_bin:
            y_parameter = findall("-?\d+\.\d+", i.scan_name)
            if len(y_parameter) == 0:
                y_parameter = findall("-?\d+", i.scan_name)
            y_parameter = [float(i) for i in y_parameter]
            y_parameter_list.append(y_parameter)

        if len(y_parameter_list[0]) > 1:
            check = np.gradient(y_parameter_list, axis=0)
            check = np.mean(check, axis=0)
            check = np.where(check > 0)
            if len(check) > 1:
                print('More than one y variable detected!')
            check = int(check[0])
            y_parameter_list = [i[check] for i in y_parameter_list]
        else:
            y_parameter_list = [i[0] for i in y_parameter_list]

        self.x_parameter_list = x_parameter_list
        self.x_parameter_step = x_parameter_step
        self.x_parameter_check = x_parameter_check
        self.energy_step = x_parameter_step
        self.y_parameter_list = y_parameter_list
        self.delay_step = np.mean(np.diff(y_parameter_list))

    def create_map(self):
        concat_list = []
        for counter, i in enumerate(self.file_bin):
            y_parameter = self.y_parameter_list[counter]
            data_y = i.data_y[i.CPS[0]]
            data_x = i.data_x

            if self.x_parameter_check > 0.005:
                data_x = self.rounding(data_x, self.x_parameter_step)
                data_x = np.around(data_x, 2)

            if self.energy_axis == 'BE':
                x_label = "Binding energy"
                x_label_a = "Kinetic energy"
            else:
                x_label = "Kinetic energy"
                x_label_a = "Binding energy"
            x_units = 'eV'
            x_units_a = 'eV'

            coords = {x_label: ("Energy", data_x)}
            xarray = xr.DataArray(data_y,
                                  dims=["Energy"],
                                  coords=coords)
            xarray.coords['Energy'] = xarray.coords[x_label]
            xarray.attrs = {'Energy units': 'eV',
                            'Energy axis': x_label,
                            'Normalized': False}
            xarray = xarray.expand_dims(dim={"Sequence": [float(y_parameter)]},
                                        axis=0)
            xarray = xarray.drop_duplicates('Energy', keep='first')
            concat_list.append(xarray)

        # concat_list_mod = []
        # for i in concat_list:
            # print(i)
            # new_a = i.loc[:, 800:200]
            # concat_list_mod.append(new_a)
            # print(new_a.coords['Energy'].values[:100])
        # concat_list = concat_list_mod
        total_map = xr.concat(concat_list, dim='Sequence', coords="minimal",
                              compat='override')
        total_map = total_map.sortby(total_map.Sequence)
        total_map = total_map.dropna(dim='Energy')
        data_x = total_map.coords['Energy'].values
        data_x_a = np.mean(self.hv_list) - total_map.coords['Energy'].values# - 4.5

        if self.energy_axis == 'BE' and self.x_parameter_check > 0.005:
            image_data_x = data_x_a[::-1]
            image_data_x_a = data_x[::-1]
            image_data_y = total_map.coords['Sequence'].values[::-1]
            image_data = total_map.values[::-1, ::-1]
        else:
            if self.energy_axis == 'BE':
                image_data_x = data_x_a
                image_data_x_a = data_x
            else:
                image_data_x = data_x
                image_data_x_a = data_x_a
            image_data_y = total_map.coords['Sequence'].values[::-1]
            image_data = total_map.values[::-1]

        x_label = 'Kinetic energy'
        y_label = 'Sequence'
        x_units = 'eV'
        y_units = 'units'
        x_label_a = 'Binding energy'
        y_label_a = 'Sequence'
        x_units_a = 'eV'
        y_units_a = 'units'
        x_order_rec = False
        y_order_rec = True

        coords = {x_label: ("Dim_x", image_data_x),
                  y_label: ("Dim_y", image_data_y)}

        Map_2D = xr.DataArray(np.array(image_data),
                              dims=["Dim_y", "Dim_x"],
                              coords=coords)
        Map_2D.coords["Dim_y"] = Map_2D.coords[y_label]
        Map_2D.coords["Dim_x"] = Map_2D.coords[x_label]

        try:
            Map_2D.coords[x_label_a] = ('Dim_x', image_data_x_a)
        except:
            pass
        try:
            Map_2D.coords[y_label_a] = ('Dim_y', image_data_y_a)
        except:
            pass

        Map_2D.attrs = {'x_label': x_label,
                        'x_units': x_units,
                        'x_order_rec': x_order_rec,
                        'y_label': y_label,
                        'y_units': y_units,
                        'y_order_rec': y_order_rec,
                        'x_label_a': x_label_a,
                        'x_units_a': x_units_a,
                        'x_order_rec_a': not x_order_rec,
                        'y_label_a': y_label_a,
                        'y_units_a': y_units_a,
                        'y_order_rec_a': not y_order_rec,
                        'x_alt': False,
                        'y_alt': False,
                        'Normalized': False}
        self.Map_2D = Map_2D
        self.Map_2D_plot = Map_2D
        self.y_step = self.Map_2D.coords['Dim_y'].values
        self.y_step = np.min(np.abs(np.gradient(self.y_step)))
        self.x_step = self.Map_2D.coords['Dim_x'].values
        self.x_step = np.gradient(self.x_step)[0]
        self.ordinate = 'delay'
        self.en_calib = '-'


class create_batch_CasaXPS(create_batch_WESPE):
    '''
    The object for storing data of combined runs.
    '''

    def __init__(self, file_dir, run_list, DLD='DLD4Q', bg_sub='off'):
        '''
        This initialization happens on 'Upload runs'.
        '''
        self.file_dir = file_dir
        self.batch_dir, self.batch_list = [], []
        for run_number in run_list:
            file_full = file_dir + os.sep + run_number
            try:
                self.batch_list.append(read_file_CasaXPS(file_full,
                                                         bg_sub=bg_sub))
                self.batch_dir.append(file_full)
            except:
                listdir_static = os.listdir(file_dir)
                listdir = []
                for i in listdir_static:
                    if run_number in i:
                        listdir.append(i)
                run_number = listdir[0]
                file_full = file_dir + os.sep + run_number
                self.batch_list.append(read_file_CasaXPS(file_full,
                                                         bg_sub=bg_sub))
                self.batch_dir.append(file_full)

        full_info = []
        for i in self.batch_list:
            full_info.append(i.info)
        self.full_info = 'DETAILED INFO:\n\n' + '\n\n'.join(full_info)

        title = 'SHORT SUMMARY:\n'
        run_num, is_static, KE, mono = [], [], [], []
        for i in self.batch_list:
            # run_num.append(str(i.run_num))
            run_num.append(str(i.file_name))
            # is_static.append(i.is_static)
            # KE.append(i.KE)
            # mono.append(i.mono_mean)
        # Run numbers
        self.run_num = ', '.join(run_num)
        try:
            run_num = [int(i) for i in run_num]
            run_num.sort()
        except ValueError:
            run_num = run_num
        if len(run_num) == 1:
            run_num = f'Uploaded run: {run_num[0]}'
        elif len(run_num) > 6:
            temp_list = [int(i.split('-run')[-1]) for i in run_num]
            run_num = f'Uploaded runs: {np.min(temp_list)}-{np.max(temp_list)}'
        else:
            run_num = [str(i) for i in run_num]
            run_num = ', '.join(run_num)
            run_num = 'Uploaded runs: ' + run_num
        self.run_num_o = run_num.replace('Uploaded runs: ', '')
        # Static scan check
        if all(is_static):
            is_static_s = 'Static check: All runs are static (+)'
        elif not any(is_static):
            is_static_s = 'Static check: All runs are delay scans (+)'
        else:
            is_static_s = 'Static check: Delay scans are mixed with static scans (!!!)'
        static_cut_list = []
        for counter, i in enumerate(is_static):
            if i is True:
                static_cut = np.mean(self.batch_list[counter].DLD_delay)
                static_cut_list.append(static_cut)
        self.static_cut_list = static_cut_list
        # short_info = [title, run_num, is_static_s, KE_s, mono_s]
        short_info = [title, run_num]
        self.short_info = '\n'.join(short_info) + '\n\n'


class read_file_ibw(create_batch_WESPE):
    def __init__(self, file_full, map_max=1, Fermi_lim=[0.4]):
        self.path = f'{os.sep}'.join(file_full.split(os.sep)[:-1])
        self.file_name = file_full.split(os.sep)[-1]
        self.Fermi_lim = Fermi_lim
        self.map_max = map_max
        data = loadibw(file_full)
        self.data = data
        self.file_full = file_full
        self.info = ''

        data_a = data['wave']['wData']
        title = data['wave']['wave_header']['bname'].decode('UTF-8')
        label = data['wave']['dimension_units'].decode('UTF-8')

        self.Seq = False
        if label.count(']') == 2:
            x_label = label[label.find(']') + 1:]
            y_label = label[:label.find(']') + 1]
        else:
            x_label = label[:label.find(']') + 1]
            y_label = label[label.find(']',label.find(']')+1) + 1:]
            self.Seq = True

        note = data['wave']['note'].decode('UTF-8')
        note_list = note.split('\r')

        self.Fermi = False
        for i in note_list:
            if '\x0b' in i:
                self.Fermi = True
            if 'Comments' in i:
                comments = i
                hv = comments[comments.find('hv'):].split(' ')[-1]
                hv = hv.replace('eV;', '')
                try:
                    self.hv = float(hv)
                except:
                    self.hv = 1000

        line_n = None
        if self.Fermi is True:
            for counter, i in enumerate(note_list):
                if 'Point' in i:
                    line_n_0 = counter + 1
                if len(i) == 0:
                    line_n_n = counter
                    break
            manip_info = note_list[line_n_0:line_n_n]
            manip_info_a = []
            for i in manip_info:
                manip_info_a.append(i.split('\x0b'))
            manip_info_a  = np.array(manip_info_a)
            polar = manip_info_a[:,1].astype(float)
            azimuth = np.mean(manip_info_a[:,1].astype(float))
            height = np.mean(manip_info_a[:,1].astype(float))
            self.image_data_z = polar
        elif self.Seq is True:
            z_0 = data['wave']['wave_header']['sfB'][2]
            z_step = data['wave']['wave_header']['sfA'][2]
            z_points = data_a.shape[2]
            z_n = z_0 + z_step*z_points
            self.image_data_z = np.arange(z_0, z_n, z_step)
        else:
            for i in note_list:
                if 'P-Axis' in i:
                    polar = float(i.split('=')[1])
                if 'R-Axis' in i:
                    azimuth = float(i.split('=')[1])
                if 'Z-Axis' in i:
                    height = float(i.split('=')[1])


        extent = [data['wave']['wave_header']['sfB'][1],
                  data['wave']['wave_header']['sfB'][1] +
                  data['wave']['wave_header']['sfA'][1]*data_a.shape[1],
                  data['wave']['wave_header']['sfB'][0],
                  data['wave']['wave_header']['sfB'][0] +
                  data['wave']['wave_header']['sfA'][0]*data_a.shape[0]]

        x_0 = data['wave']['wave_header']['sfB'][1]
        x_step = data['wave']['wave_header']['sfA'][1]
        x_points = data_a.shape[1]
        x_n = x_0 + x_step*x_points
        self.image_data_x = np.arange(x_0, x_n, x_step)

        y_0 = data['wave']['wave_header']['sfB'][0]
        y_step = data['wave']['wave_header']['sfA'][0]
        y_points = data_a.shape[0]
        y_n = y_0 + y_step*y_points
        self.image_data_y = np.arange(y_0, y_n, y_step)

        self.image_data = data['wave']['wData']


    def create_map(self):
        if len(self.image_data_y) != self.image_data.shape[0]:
            self.image_data_y = self.image_data_y[:-1]
        if self.Fermi is True:
            coords = {"Emission angle": ("Momentum", self.image_data_x),
                      "Kinetic energy": ("Energy", self.image_data_y),
                      "Polar angle": ("Polar", self.image_data_z)}
            Map_2D = xr.DataArray(np.array(self.image_data),
                                  dims=["Energy", "Momentum", "Polar"],
                                  coords=coords)
            Map_2D.coords['Polar'] = Map_2D.coords['Polar angle']
            Map_2D.coords['Energy'] = Map_2D.coords['Kinetic energy']
            Map_2D.coords['Momentum'] = Map_2D.coords['Emission angle']
            extent = [np.min(self.image_data_x), np.max(self.image_data_x),
                      np.min(self.image_data_z), np.max(self.image_data_z)]
            y_label = 'Emission angle (deg)'
            x_label = 'Polar angle (deg)'

            Map_2D.attrs = {'Momentum units': 'deg',
                            'Energy units': 'eV',
                            'Polar units': 'deg',
                            'Momentum axis': 'Emission angle',
                            'Energy axis': 'Kinetic energy',
                            'Polar axis': 'Polar angle',
                            'Normalized': False}
        elif self.Seq is True:
            coords = {"Emission angle": ("Momentum", self.image_data_x),
                      "Kinetic energy": ("Energy", self.image_data_y),
                      "Sequence": ("Step", self.image_data_z)}
            Map_2D = xr.DataArray(np.array(self.image_data),
                                  dims=["Energy", "Momentum", "Step"],
                                  coords=coords)
            Map_2D.coords['Step'] = Map_2D.coords['Sequence']
            Map_2D.coords['Energy'] = Map_2D.coords['Kinetic energy']
            Map_2D.coords['Momentum'] = Map_2D.coords['Emission angle']
            extent = [np.min(self.image_data_x), np.max(self.image_data_x),
                      np.min(self.image_data_z), np.max(self.image_data_z)]
            y_label = 'Sequence (units)'
            x_label = 'Kinetic energy (eV)'

            Map_2D.attrs = {'Momentum units': 'deg',
                            'Energy units': 'eV',
                            'Sequence units': 'units',
                            'Momentum axis': 'Emission angle',
                            'Energy axis': 'Kinetic energy',
                            'Sequence axis': 'Step',
                            'Normalized': False}
        else:
            coords = {"Emission angle": ("Momentum", self.image_data_x),
                      "Kinetic energy": ("Energy", self.image_data_y)}
            if len(self.image_data.shape) > 2:
                self.image_data = np.sum(self.image_data, axis=2)
            Map_2D = xr.DataArray(np.array(self.image_data),
                                  dims=["Energy", "Momentum"],
                                  coords=coords)
            x_label = 'Emission angle (deg)'
            y_label = 'Kinetic energy(eV)'
            Map_2D.coords['Energy'] = Map_2D.coords['Kinetic energy']
            Map_2D.coords['Momentum'] = Map_2D.coords['Emission angle']

            Map_2D.attrs = {'Momentum units': 'deg',
                            'Energy units': 'eV',
                            'Momentum axis': 'Emission angle',
                            'Dim_x': 'Kinetic energy',
                            'Normalized': False}

        self.Map_2D = Map_2D
        if len(self.Map_2D.shape) == 3:
            if self.Fermi is True:
                '''
                if self.Fermi_lim is not None:
                    y_n = np.max(self.Map_2D.coords['Energy'].values)
                    Fermi_l = np.min(self.Fermi_lim)
                    Fermi_h = np.max(self.Fermi_lim)
                    if len(self.Fermi_lim) < 2:
                        Fermi_h = y_n
                        Fermi_l = Fermi_h - self.Fermi_lim[0]
                    elif np.mean(self.Fermi_lim) < 10:
                        Fermi_l = y_n - self.Fermi_lim[0] - abs(self.Fermi_lim[1])/2
                        Fermi_h = y_n - self.Fermi_lim[0] + abs(self.Fermi_lim[1])/2
                    self.Map_2D = self.Map_2D.loc[Fermi_l:Fermi_h, :, :]
                self.Map_2D = self.Map_2D.sum(dim='Energy')
                '''
                image_data = self.Map_2D.values.transpose(1,2,0)
                image_data = image_data[::-1]
                image_data_x = self.Map_2D.coords['Polar'].values
                image_data_y = self.Map_2D.coords['Momentum'].values
                image_data_y = image_data_y[::-1]
                
                image_data_z = self.Map_2D.coords['Energy'].values
                
                extent = [np.min(image_data_x), np.max(image_data_x),
                          np.min(image_data_y), np.max(image_data_y)]
                x_label = 'Polar angle'
                y_label = 'Emission angle'
                z_label = 'Kinetic energy'
                
                x_units = 'deg'
                y_units = 'deg'
                z_units = 'eV'
                
                x_label_a = 'Polar angle'
                y_label_a = 'Emission angle'
                z_label_a = 'Kinetic energy'
                
                x_units_a = 'deg'
                y_units_a = 'deg'
                z_units_a = 'eV'
                
                x_order_rec = False
                y_order_rec = True
                z_order_rec = False
                
                coords = {x_label: ("Dim_x", image_data_x),
                          y_label: ("Dim_y", image_data_y),
                          z_label: ("Dim_z", image_data_z)}

                Map_2D = xr.DataArray(np.array(image_data),
                                      dims=["Dim_y", "Dim_x", "Dim_z"],
                                      coords=coords)
                Map_2D.coords["Dim_z"] = Map_2D.coords[z_label]

                Map_2D.name = self.file_name
    
                Map_2D.coords['Dim_x'] = Map_2D.coords[x_label]
                Map_2D.coords['Dim_y'] = Map_2D.coords[y_label]
    
                Map_2D.attrs = {'x_label': x_label,
                                'x_units': x_units,
                                'x_order_rec': x_order_rec,
                                'y_label': y_label,
                                'y_units': y_units,
                                'y_order_rec': y_order_rec,
                                'x_label_a': x_label_a,
                                'x_units_a': x_units_a,
                                'x_order_rec_a': not x_order_rec,
                                'y_label_a': y_label_a,
                                'y_units_a': y_units_a,
                                'y_order_rec_a': not y_order_rec,
                                'x_alt': False,
                                'y_alt': False,
                                'Normalized': False,
                                'z_label': z_label,
                                'z_units': z_units,
                                'z_order_rec': z_order_rec,
                                'z_label_a': z_label_a,
                                'z_units_a': z_units_a,
                                'z_order_rec_a': not z_order_rec,
                                'z_alt': False}
            elif self.Seq is True:
                image_data = self.Map_2D.values.transpose(1,0,2)
                image_data = image_data[::-1]
                image_data_x = self.Map_2D.coords['Energy'].values
                image_data_y = self.Map_2D.coords['Momentum'].values
                image_data_y = image_data_y[::-1]
                
                image_data_z = self.Map_2D.coords['Step'].values
                
                extent = [np.min(image_data_x), np.max(image_data_x),
                          np.min(image_data_y), np.max(image_data_y)]
                x_label = 'Kinetic energy'
                y_label = 'Emission angle'
                z_label = 'Sequence'
                
                x_units = 'eV'
                y_units = 'deg'
                z_units = 'a.u.'
                
                x_label_a = 'Kinetic energy'
                y_label_a = 'Emission angle'
                z_label_a = 'Sequence'
                
                x_units_a = 'eV'
                y_units_a = 'deg'
                z_units_a = 'a.u.'
                
                x_order_rec = False
                y_order_rec = True
                z_order_rec = False
                
                coords = {x_label: ("Dim_x", image_data_x),
                          y_label: ("Dim_y", image_data_y),
                          z_label: ("Dim_z", image_data_z)}

                Map_2D = xr.DataArray(np.array(image_data),
                                      dims=["Dim_y", "Dim_x", "Dim_z"],
                                      coords=coords)
                Map_2D.coords["Dim_z"] = Map_2D.coords[z_label]

                Map_2D.name = self.file_name
    
                Map_2D.coords['Dim_x'] = Map_2D.coords[x_label]
                Map_2D.coords['Dim_y'] = Map_2D.coords[y_label]
    
                Map_2D.attrs = {'x_label': x_label,
                                'x_units': x_units,
                                'x_order_rec': x_order_rec,
                                'y_label': y_label,
                                'y_units': y_units,
                                'y_order_rec': y_order_rec,
                                'x_label_a': x_label_a,
                                'x_units_a': x_units_a,
                                'x_order_rec_a': not x_order_rec,
                                'y_label_a': y_label_a,
                                'y_units_a': y_units_a,
                                'y_order_rec_a': not y_order_rec,
                                'x_alt': False,
                                'y_alt': False,
                                'Normalized': False,
                                'z_label': z_label,
                                'z_units': z_units,
                                'z_order_rec': z_order_rec,
                                'z_label_a': z_label_a,
                                'z_units_a': z_units_a,
                                'z_order_rec_a': not z_order_rec,
                                'z_alt': False}
                attrs = Map_2D.attrs
                # Map_2D = Map_2D.sum('Dim_z')
                Map_2D.attrs = attrs
                # self.Map_2D = self.Map_2D.sum(dim='Momentum')
        else:
            if len(self.Map_2D.shape) == 2:
                if self.Fermi is True:
                    image_data = self.Map_2D.values
                    image_data = image_data[::-1]
                    image_data_x = self.Map_2D.coords['Polar'].values
                    image_data_y = self.Map_2D.coords['Momentum'].values
                    image_data_y = image_data_y[::-1]
                    extent = [np.min(image_data_x), np.max(image_data_x),
                              np.min(image_data_y), np.max(image_data_y)]
                    x_label = 'Polar angle'
                    y_label = 'Emission angle'
                    x_units = 'deg'
                    y_units = 'deg'
                    x_label_a = 'Polar angle'
                    y_label_a = 'Emission angle'
                    x_units_a = 'deg'
                    y_units_a = 'deg'
                    x_order_rec = False
                    y_order_rec = True
                elif self.Seq is True:
                    image_data = self.Map_2D.values
                    image_data = np.rot90(image_data, 3)[::, ::-1]
                    image_data = image_data[::-1]
                    image_data_x = self.Map_2D.coords['Energy'].values
                    image_data_y = self.Map_2D.coords['Step'].values
                    image_data_y = image_data_y[::-1]
                    extent = [np.min(image_data_x), np.max(image_data_x),
                              np.min(image_data_y), np.max(image_data_y)]
                    x_label = 'Kinetic energy'
                    y_label = 'Sequence'
                    x_units = 'eV'
                    y_units = 'units'
                    x_label_a = 'Binding energy'
                    y_label_a = 'Sequence'
                    x_units_a = 'eV'
                    y_units_a = 'units'
                    x_order_rec = False
                    y_order_rec = True
                    image_data_x_a = self.hv - image_data_x - 4.5
                else:
                    image_data = self.Map_2D.values
                    image_data = image_data[::-1]
                    image_data_x = self.Map_2D.coords['Momentum'].values
                    image_data_y = self.Map_2D.coords['Energy'].values
                    image_data_y = image_data_y[::-1]
                    extent = [np.min(image_data_x), np.max(image_data_x),
                              np.min(image_data_y), np.max(image_data_y)]
                    x_label = 'Emission angle'
                    y_label = 'Kinetic energy'
                    x_units = 'deg'
                    y_units = 'eV'
                    x_label_a = 'Emission angle'
                    y_label_a = 'Kinetic energy'
                    x_units_a = 'deg'
                    y_units_a = 'eV'
                    x_order_rec = False
                    y_order_rec = False
    
            coords = {x_label: ("Dim_x", image_data_x),
                      y_label: ("Dim_y", image_data_y)}
            if len(image_data.shape) > 2:
                image_data = np.sum(image_data, axis=2)
            Map_2D = xr.DataArray(np.array(image_data),
                                    dims=["Dim_y", "Dim_x"],
                                    coords=coords)
            Map_2D.coords["Dim_y"] = Map_2D.coords[y_label]
            Map_2D.coords["Dim_x"] = Map_2D.coords[x_label]
    
            try:
                Map_2D.coords[x_label_a] = ('Dim_x', image_data_x_a)
            except:
                pass
            try:
                Map_2D.coords[y_label_a] = ('Dim_y', image_data_y_a)
            except:
                pass
    
            Map_2D.attrs = {'x_label': x_label,
                            'x_units': x_units,
                            'x_order_rec': x_order_rec,
                            'y_label': y_label,
                            'y_units': y_units,
                            'y_order_rec': y_order_rec,
                            'x_label_a': x_label_a,
                            'x_units_a': x_units_a,
                            'x_order_rec_a': not x_order_rec,
                            'y_label_a': y_label_a,
                            'y_units_a': y_units_a,
                            'y_order_rec_a': not y_order_rec,
                            'x_alt': False,
                            'y_alt': False,
                            'Normalized': False}
        self.Map_2D = Map_2D
        self.Map_2D_plot = Map_2D
        self.y_step = self.Map_2D.coords['Dim_y'].values
        self.y_step = np.min(np.abs(np.gradient(self.y_step)))
        self.x_step = self.Map_2D.coords['Dim_x'].values
        self.x_step = np.gradient(self.x_step)[0]
        self.ordinate = 'delay'
        self.en_calib = '-'


class create_batch_ibw(create_batch_WESPE):
    '''
    The object for storing data of combined runs.
    '''

    def __init__(self, file_dir, run_list, DLD='DLD4Q'):
        '''
        This initialization happens on 'Upload runs'.
        '''
        self.file_dir = file_dir
        self.batch_dir, self.batch_list = [], []
        for run_number in run_list:
            file_full = file_dir + os.sep + run_number + '.ibw'
            try:
                self.batch_list.append(read_file_ibw(file_full))
                self.batch_dir.append(file_full)
            except:
                listdir_static = os.listdir(file_dir)
                listdir = []
                for i in listdir_static:
                    if run_number in i:
                        listdir.append(i)
                run_number = listdir[0]
                file_full = file_dir + os.sep + run_number
                self.batch_list.append(read_file_ibw(file_full))
                self.batch_dir.append(file_full)

        full_info = []
        for i in self.batch_list:
            full_info.append(i.info)
        self.full_info = 'DETAILED INFO:\n\n' + '\n\n'.join(full_info)

        title = 'SHORT SUMMARY:\n'
        run_num, is_static, KE, mono = [], [], [], []
        for i in self.batch_list:
            # run_num.append(str(i.run_num))
            run_num.append(str(i.file_name))
            # is_static.append(i.is_static)
            # KE.append(i.KE)
            # mono.append(i.mono_mean)
        # Run numbers
        self.run_num = ', '.join(run_num)
        try:
            run_num = [int(i) for i in run_num]
            run_num.sort()
        except ValueError:
            run_num = run_num
        if len(run_num) == 1:
            run_num = f'Uploaded run: {run_num[0]}'
        elif len(run_num) > 6:
            temp_list = [int(i.split('-run')[-1]) for i in run_num]
            run_num = f'Uploaded runs: {np.min(temp_list)}-{np.max(temp_list)}'
        else:
            run_num = [str(i) for i in run_num]
            run_num = ', '.join(run_num)
            run_num = 'Uploaded runs: ' + run_num
        self.run_num_o = run_num.replace('Uploaded runs: ', '')
        # Static scan check
        if all(is_static):
            is_static_s = 'Static check: All runs are static (+)'
        elif not any(is_static):
            is_static_s = 'Static check: All runs are delay scans (+)'
        else:
            is_static_s = 'Static check: Delay scans are mixed with static scans (!!!)'
        static_cut_list = []
        for counter, i in enumerate(is_static):
            if i is True:
                static_cut = np.mean(self.batch_list[counter].DLD_delay)
                static_cut_list.append(static_cut)
        self.static_cut_list = static_cut_list
        # short_info = [title, run_num, is_static_s, KE_s, mono_s]
        short_info = [title, run_num]
        self.short_info = '\n'.join(short_info) + '\n\n'

    def create_map(self):
        '''
        This method sums delay-energy maps of individual runs
        uploaded to the batch.
        '''
        self.x_step = self.batch_list[0].x_step
        self.y_step = self.batch_list[0].y_step
        self.ordinate = self.batch_list[0].ordinate
        attrs = self.batch_list[0].Map_2D.attrs
        for counter, i in enumerate(self.batch_list):
            if counter == 0:
                total_map = i.Map_2D
            else:
                total_map = total_map + i.Map_2D
        total_map.attrs = attrs
        if np.min(total_map.values.shape) == 0:
            concat_list = []
            for counter, i in enumerate(self.batch_list):
                concat_list.append(i.Map_2D)
            total_map = xr.concat(concat_list, dim='Dim_y', coords="minimal",
                                  compat='override')
        if np.min(total_map.values.shape) == 0:
            total_map.attrs['Merge successful'] = False
        self.Map_2D = total_map
        self.Map_2D_plot = self.Map_2D


class map_cut:
    '''
    The object for creating and storing slices of the delay-energy map.
    '''

    def __init__(self, obj, positions, deltas,
                 axis='Dim_y', approach='mean'):
        '''
        Initialization of the object when the creation of
        the slices is performed
        obj - create_batch_WESPE object
        positions - a list of values where one wants to make a slice
        deltas - a list of slice widths
        axis - 'Dim_y' or 'Dim_x' of the delay-energy map
        approach - 'mean' or 'sum' of individual lines within a slice
        '''
        self.run_num_o = obj.run_num_o
        self.x_step = obj.x_step
        self.y_step = obj.y_step
        self.file_dir = obj.file_dir
        self.ordinate = obj.ordinate
        self.plot_dif = False
        self.Map_2D_plot = obj.Map_2D_plot
        self.cor = False
        self.en_calib = obj.batch_list[0].en_calib
        self.file_full = obj.batch_list[0].file_full
        try:
            self.bunch_sel = obj.batch_list[0].bunch_sel
        except:
            self.bunch_sel = None
        self.obj = obj
        try:
            self.varied_y_step = obj.varied_y_step
        except AttributeError:
            pass
        if isinstance(positions, list) is False:
            positions = [positions]
        if isinstance(deltas, list) is False:
            deltas = [deltas]

        if len(deltas) < len(positions):  # filling in missing delta values
            for i in range(len(positions)):
                try:
                    deltas[i]
                except IndexError:
                    try:
                        deltas.append(deltas[i-1])
                    except IndexError:
                        deltas.append(0.5)
        self.axis = axis
        if obj.Map_2D_plot.attrs['Normalized'] is True:
            self.arb_u = True
        else:
            self.arb_u = False
        self.positions = []
        self.deltas = []
        self.cuts = []
        self.map_show = []
        self.fit = False
        for counter, position in enumerate(positions):
            self.positions.append(position)
            self.deltas.append(deltas[counter])
            limit_1 = position - deltas[counter]/2
            limit_2 = position + deltas[counter]/2
            limits = [limit_1, limit_2]
            cut = obj.ROI(limits, axis, mod_map=False)
            if counter == 0:
                self.cut_info = cut
            if axis == 'Dim_y':
                self.var_n = 'Y'
                self.var_n_r = 'X'
                self.coords = cut.coords['Dim_x'].values
                if approach == 'sum':
                    cut = cut.sum('Dim_y')
                else:
                    cut = cut.mean('Dim_y')
            else:
                self.var_n = 'X'
                self.var_n_r = 'Y'
                self.coords = cut.coords['Dim_y'].values
                if approach == 'sum':
                    cut = cut.sum('Dim_x')
                else:
                    cut = cut.mean('Dim_x')
            if cut.isnull().values.all():
                dummy = np.zeros(self.coords.shape)
                self.cuts.append(dummy)
                self.map_show.append(False)
            else:
                self.cuts.append(cut.values)
                self.map_show.append(True)

    def correlate_i(self, step=None, c_f=0.1):
        data = self.Map_2D_plot.values
        data_step = np.gradient(self.Map_2D_plot.coords['Dim_x'].values)[0]
        delay = self.Map_2D_plot.coords['Dim_y'].values

        if step is None:
            for i in range(10):
                rnd = abs(np.around(data_step, i))
                if rnd > 0:
                    check = np.floor(abs(data_step)/(0.8*rnd))
                    if check > 0:
                        if i == 0 or i == 1:
                            d = '1e-' + str(i+2)
                        else:
                            d = '1e-' + str(i+1)
                        step = float(f'{d}')
                        break
        if step is None:
            step = 0.0001

        order_y = 'y_order_rec'
        if self.Map_2D_plot.attrs['y_alt'] is True:
            order_y += '_a'

        if self.Map_2D_plot.attrs[order_y] is False:
            cutoff = (np.max(delay) - np.min(delay))*c_f
            cutoff = np.max(delay) - cutoff
            EDC = self.Map_2D_plot.loc[cutoff:].mean('Dim_y')
        else:
            t_axis_step = abs(np.median(np.gradient(delay)))
            t_axis_step = int(-2.5*t_axis_step)
            EDC = self.Map_2D_plot.loc[t_axis_step:].mean('Dim_y')
            if int(EDC.fillna(0).mean()) == 0:
                EDC = self.Map_2D_plot[0:2].mean('Dim_y')

        EDC_x = EDC.coords['Dim_x'].values
        EDC_y = EDC.values

        EDC_x_step = np.arange(read_file_WESPE.rounding(EDC_x.min(), step),
                               read_file_WESPE.rounding(EDC_x.max(), step), step)
        if EDC_x[-1] < EDC_x[0]:
            EDC_y_step = np.interp(EDC_x_step, EDC_x[::-1], EDC_y[::-1])
            BE = True
        else:
            EDC_y_step = np.interp(EDC_x_step, EDC_x, EDC_y)
            BE = False
        shift = []
        for i in data:
            data_y = i
            if BE:
                data_y_step = np.interp(EDC_x_step, EDC_x[::-1], data_y[::-1])
            else:
                data_y_step = np.interp(EDC_x_step, EDC_x, data_y)

            EDC_y_step = EDC_y_step - np.min(EDC_y_step)
            data_y_step = data_y_step - np.min(data_y_step)

            corr = scipy.signal.correlate(EDC_y_step, data_y_step, mode='full')
            lags = scipy.signal.correlation_lags(EDC_y_step.shape[0],
                                                  data_y_step.shape[0], mode="full")

            lag = (lags[np.argmax(corr)] / (data_step/step))*-data_step
            shift.append(lag)
        # if np.median(shift) < 0:
        #     shift = -np.array(shift)
        self.coords = delay
        self.cuts = [shift]
        self.axis = 'Dim_x'
        self.cor = True

    def correlate_b(self, step=0.001, c_f=0.1, b_sel=None):
        data = self.Map_2D_plot.values
        data_step = np.gradient(self.Map_2D_plot.coords['Dim_x'].values)[0]
        delay = self.Map_2D_plot.coords['Dim_y'].values
        file = read_file_ALS(self.file_full)
        
        try:
            DLD_t_res = self.obj.Map_2D_plot.attrs['DLD_t_res']
        except:
            DLD_t_res = None

        if b_sel is None:
            file.create_map(ordinate='delay',
                            bunch_sel=self.bunch_sel-1,
                            DLD_t_res=DLD_t_res)
        elif type(b_sel) == int or type(b_sel) == str:
            if b_sel == 'prev':
                file.create_map(ordinate='delay',
                                bunch_sel=f'1,{self.bunch_sel-1}',
                                DLD_t_res=DLD_t_res)
            else:
                file.create_map(ordinate='delay',
                                bunch_sel=b_sel,
                                DLD_t_res=DLD_t_res)
        elif type(b_sel) == list:
            file.create_map(ordinate='delay',
                            bunch_sel=f'{min(b_sel)},{max(b_sel)}',
                            DLD_t_res=DLD_t_res)
        else:
            raise ValueError('Check input values!')

        if self.Map_2D_plot.attrs['x_alt'] is True:
            file.set_BE()
        lims = self.Map_2D_plot.coords['Dim_x'].values
        lims = [np.min(lims), np.max(lims)]
        file.ROI(lims, axis='Dim_x')

        EDC = file.Map_2D_plot.values
        EDC_x = self.Map_2D_plot.coords['Dim_x'].values
        EDC_y = EDC
        EDC_x_step = np.arange(read_file_ALS.rounding(EDC_x.min(), step),
                          read_file_ALS.rounding(EDC_x.max(), step), step)

        shift = []
        for i in np.arange(data.shape[0]):
            data_y = data[i]
            data_x = EDC[i]

            if EDC_x[-1] < EDC_x[0]:
                EDC_y_step = np.interp(EDC_x_step, EDC_x[::-1], data_x[::-1])
                BE = True
            else:
                EDC_y_step = np.interp(EDC_x_step, EDC_x, data_x)
                BE = False

            if BE:
                data_y_step = np.interp(EDC_x_step, EDC_x[::-1], data_y[::-1])
            else:
                data_y_step = np.interp(EDC_x_step, EDC_x, data_y)

            EDC_y_step = EDC_y_step - np.min(EDC_y_step)
            data_y_step = data_y_step - np.min(data_y_step)
            corr = scipy.signal.correlate(EDC_y_step, data_y_step, mode='full')
            lags = scipy.signal.correlation_lags(EDC_y_step.shape[0],
                                                  data_y_step.shape[0], mode="full")

            lag = (lags[np.argmax(corr)] / (data_step/step))*-data_step
            shift.append(lag)
        # if np.median(shift) < 0:
        #     shift = -np.array(shift)
        self.coords = delay
        self.cuts = [shift]
        self.axis = 'Dim_x'
        self.cor = True

    def correlate_total(self, step=0.01, c_f=0.1, b_sel=None,
                        real_time=True, period = 656168):
        try:
            DLD_t_res = self.obj.Map_2D_plot.attrs['DLD_t_res']
        except:
            DLD_t_res = None

        file = read_file_ALS(self.file_full)
        file.create_map(ordinate='MB_ID',
                        DLD_t_res=DLD_t_res)
        b_ID = file.Map_2D_plot.coords['Dim_y'].values
        if b_sel is None:
            file.create_map(ordinate='delay',
                            bunch_sel=self.bunch_sel-1,
                            DLD_t_res=DLD_t_res)
        elif type(b_sel) == int or type(b_sel) == str:
            if b_sel == 'prev':
                file.create_map(ordinate='delay',
                                bunch_sel=f'1,{self.bunch_sel-1}',
                                DLD_t_res=DLD_t_res)
            else:
                file.create_map(ordinate='delay',
                                bunch_sel=b_sel,
                                DLD_t_res=DLD_t_res)
        elif type(b_sel) == list:
            file.create_map(ordinate='delay',
                            bunch_sel=f'{min(b_sel)},{max(b_sel)}',
                            DLD_t_res=DLD_t_res)
        else:
            raise ValueError('Check input values!')

        if self.Map_2D_plot.attrs['x_alt'] is True:
            file.set_BE()
        lims = self.Map_2D_plot.coords['Dim_x'].values
        lims = [np.min(lims), np.max(lims)]
        file.ROI(lims, axis='Dim_x')

        EDC = file.Map_2D_plot.values
        EDC_x = self.Map_2D_plot.coords['Dim_x'].values
        PS_values = file.Map_2D_plot.coords['Dim_y'].values[::-1]
        EDC_y = EDC
        EDC_x_step = np.arange(read_file_ALS.rounding(EDC_x.min(), step),
                          read_file_ALS.rounding(EDC_x.max(), step), step)

        data_file = read_file_ALS(self.file_full)
        shift_total = []
        PS_values_total = []
        for j in b_ID:
            data_file.create_map(ordinate='delay',
                                 bunch_sel=j,
                                 DLD_t_res=DLD_t_res)
            if self.Map_2D_plot.attrs['x_alt'] is True:
                data_file.set_BE()
            data_file.ROI(lims, axis='Dim_x')
            data = data_file.Map_2D_plot.values
            data_step = np.gradient(data_file.Map_2D_plot.coords['Dim_x'].values)[0]
            delay = data_file.Map_2D_plot.coords['Dim_y'].values
            shift = []
            for i in np.arange(data.shape[0]):
                data_y = data[i]
                data_x = EDC[i]

                if EDC_x[-1] < EDC_x[0]:
                    EDC_y_step = np.interp(EDC_x_step, EDC_x[::-1], data_x[::-1])
                    BE = True
                else:
                    EDC_y_step = np.interp(EDC_x_step, EDC_x, data_x)
                    BE = False

                if BE:
                    data_y_step = np.interp(EDC_x_step, EDC_x[::-1], data_y[::-1])
                else:
                    data_y_step = np.interp(EDC_x_step, EDC_x, data_y)

                EDC_y_step = EDC_y_step - np.min(EDC_y_step)
                data_y_step = data_y_step - np.min(data_y_step)
                corr = scipy.signal.correlate(EDC_y_step, data_y_step, mode='full')
                lags = scipy.signal.correlation_lags(EDC_y_step.shape[0],
                                                      data_y_step.shape[0], mode="full")

                lag = (lags[np.argmax(corr)] / (data_step/step))*-data_step
                shift.append(lag)
            # if np.median(shift) < 0:
            #     shift = -np.array(shift)
            # if j == self.bunch_sel:
            #     # value = np.max(shift)
            #     value = np.array(shift).flat[np.abs(shift).argmax()]
            # else:
            #     value = np.mean(shift)
            shift_total.append(shift[::-1])
            PS_values_total.append(PS_values+period*(np.max(b_ID)-j))
        shift_total = np.array(shift_total).flatten()
        PS_values_total = np.array(PS_values_total).flatten()

        self.cuts = [list(shift_total)]
        self.axis = 'Dim_x'
        self.t_axis = 'Delay relative t0'
        self.cor = True
        self.ordinate = 'MB_ID'

        if real_time is True:
            self.coords = list(PS_values_total/1000000)
            self.cut_info.attrs['y_label'] = 'Phase shifter values'
            self.cut_info.attrs['y_units'] = 'μs'
            self.cut_info.attrs['y_label_a'] = 'Delay'
            self.cut_info.attrs['y_units_a'] = 'μs'
        else:
            self.coords = np.arange(shift_total.shape[0])
            self.cut_info.attrs['y_order_rec'] = True
            self.cut_info.attrs['y_order_rec_a'] = False
            self.cut_info.attrs['y_label'] = 'Index'
            self.cut_info.attrs['y_units'] = 'units'
            self.cut_info.attrs['y_label_a'] = 'Index'
            self.cut_info.attrs['y_units_a'] = 'units'

    def make_fit(self):
        '''
        Method for fitting of the very first slice with singular Voigt curve.
        It is supposed to be used for finding time zero.
        '''
        x = self.coords
        y = self.cuts[0]
        if np.median(y) < 0:
            y = -np.array(y)
            self.cuts[0] = y
        axis_step = np.gradient(x).mean()

        '''Voigt fit'''
        model = VoigtModel() + ConstantModel()

        amplitude_g = np.max(y) - np.min(y)
        center_g = x[np.argmax(y)]
        c_g = np.median(y)

        # create parameters with initial values
        params = model.make_params(amplitude=amplitude_g, center=center_g,
                                   sigma=abs(axis_step),
                                   gamma=abs(axis_step), c=c_g)

        # maybe place bounds on some parameters
        params['center'].min = np.min(x)
        params['center'].max = np.max(x)
        params['sigma'].min = 0.5*abs(axis_step)
        params['sigma'].max = abs(axis_step)*200
        params['gamma'].min = 0.5*abs(axis_step)
        params['gamma'].max = abs(axis_step)*200
        if amplitude_g != 0:
            params['amplitude'].min = amplitude_g/10
            params['amplitude'].max = amplitude_g*100
        if np.min(y) + np.max(y) != 0:
            params['c'].min = np.min(y)
            params['c'].max = np.max(y)

        # do the fit, print out report with results
        result = model.fit(y, params, x=x)
        print(result.fit_report())

        self.center = np.around(result.params['center'].value, 2)
        sigma = result.params['sigma'].value
        gamma = result.params['gamma'].value
        amplitude = result.params['amplitude'].value
        c = result.params['amplitude'].value
        self.fwhm = np.around(result.params['fwhm'].value, 2)

        center_std = result.params['center'].stderr
        sigma_std = result.params['sigma'].stderr
        gamma_std = result.params['gamma'].stderr
        amplitude_std = result.params['amplitude'].stderr
        c_std = result.params['amplitude'].stderr
        fwhm_std = result.params['fwhm'].stderr

        self.x_fit = np.arange(x[0], x[-1], np.diff(x)[0]/10)
        self.y_fit = result.eval(x=self.x_fit)
        RMS_1 = y - result.eval(x=x)
        RMS_1 = np.sqrt(np.sum(RMS_1**2)/RMS_1.shape[0])

        '''Biexp fit'''
        try:
            if axis_step > 0:
                x = -x
                center_g = x[np.argmax(y)]
            model = Model(biexponential_decay, independent_vars=['x'])
            params = model.make_params(tzero=center_g,
                                       sigma=abs(axis_step)*2,
                                       tconst1=abs(np.max(x)-center_g),
                                       tconst2=abs((np.max(x)-center_g)*10),
                                       amp1=amplitude_g/2,
                                       amp2=amplitude_g/2,
                                       offset=c_g)
            if abs(axis_step) > 0.04 and abs(axis_step) < 0.21:
                params['tconst1'].min = 0.8
                params['tconst2'].min = 1.8
            else:
                params['tconst1'].min = abs(axis_step)
                params['tconst2'].min = abs(axis_step)
            params['tzero'].min = np.min(x)
            params['tzero'].max = np.max(x)
            params['sigma'].min = abs(axis_step)*0.5
            params['sigma'].max = abs(axis_step)*200
            params['amp1'].min = 0
            params['amp2'].min = 0
            if amplitude_g != 0:
                params['amp1'].max = amplitude_g*100
                params['amp2'].max = amplitude_g*100
            if np.min(y) + np.max(y) != 0:
                params['offset'].min = np.min(y)
                params['offset'].max = np.max(y)
            result = model.fit(y, params, x=x)
            print(result.fit_report())

            tzero = np.around(result.params['tzero'].value, 4)
            sigma = np.around(result.params['sigma'].value, 4)
            tconst1 = np.around(result.params['tconst1'].value, 4)
            tconst2 = np.around(result.params['tconst2'].value, 4)
            self.amp1 = np.around(result.params['amp1'].value, 4)
            self.amp2 = np.around(result.params['amp2'].value, 4)
            offset = np.around(result.params['offset'].value, 4)

            center = np.around(tzero, 2)
            fwhm = np.around(sigma, 2)
            tconst1 = np.around(tconst1, 2)
            tconst2 = np.around(tconst2, 2)

            x_fit = np.arange(x[0], x[-1], np.diff(x)[0]/10)
            y_fit = result.eval(x=x_fit)
            RMS_2 = y - result.eval(x=x)
            RMS_2 = np.sqrt(np.sum(RMS_2**2)/RMS_2.shape[0])
            if axis_step > 0:
                x_fit = -x_fit
                center = -center

            check = True
            if self.amp1 > self.amp2:
                if tconst1 < abs(np.median(x)*0.2):
                    if self.amp2/amplitude_g < 0.001:
                        check = False
                    elif self.amp2/self.amp1 < 1/50:
                        check = False
            else:
                if tconst2 < abs(np.median(x)*0.2):
                    if self.amp1/amplitude_g < 0.001:
                        check = False
                    elif self.amp1/self.amp2 < 1/50:
                        check = False

            if RMS_2 < RMS_1 and check is True and self.axis == 'Dim_x':
                self.x_fit = x_fit
                self.y_fit = y_fit
                self.center = center
                self.fwhm = fwhm
                self.tconst1 = tconst1
                self.tconst2 = tconst2
        except:
            pass
        self.fit = True

    def dif_plot(self):
        try:
            with open('config.json', 'r') as json_file:
                config = json.load(json_file)
        except FileNotFoundError:
            with open('packages/config.json', 'r') as json_file:
                config = json.load(json_file)
        config = json.dumps(config)
        config = json.loads(config,
                            object_hook=lambda d: SimpleNamespace(**d))
        self.plot_dif = True
        magn = config.t_dif_magn
        dif_labels = []
        t_cut_plot = self.cuts
        counter = 0
        t_cut_plot_dif = []
        for i in range(len(t_cut_plot)):
            if counter == 0:
                reference = t_cut_plot[i]
            else:
                dif_line = [a - b for a, b in zip(t_cut_plot[i], reference)]
                if magn > 1 or magn < 1:
                    dif_line = [a*magn for a in dif_line]
                    dif_labels.append(f'Difference {self.var_n}$_{counter+1}$-{self.var_n}$_1$ x {magn}')
                else:
                    dif_labels.append(f'Difference {self.var_n}$_{counter+1}$-{self.var_n}$_1$')
                t_cut_plot_dif.append(np.array(dif_line))
            counter += 1
        self.dif_cuts = t_cut_plot_dif
        self.dif_labels = dif_labels

    def align_cuts(self):
        shift = []
        for c_i, i in enumerate(self.cuts):
            dump = []
            EDC_y_step = np.array(i) - np.min(i)
            for c_j, j in enumerate(self.cuts):
                data_y_step = np.array(j) - np.min(j)
                corr = scipy.signal.correlate(EDC_y_step, data_y_step,
                                              mode='full')
                lags = scipy.signal.correlation_lags(EDC_y_step.shape[0],
                                                     data_y_step.shape[0],
                                                     mode="full")
                lag = (lags[np.argmax(corr)])
                dump.append(lag)
            shift.append(np.min(lag))
        shift = np.array(shift) - np.max(shift)
        for counter, i in enumerate(self.cuts):
            self.cuts[counter] = np.roll(i, np.abs(shift[counter]))

    def waterfall(self):
        try:
            with open('config.json', 'r') as json_file:
                config = json.load(json_file)
        except FileNotFoundError:
            with open('packages/config.json', 'r') as json_file:
                config = json.load(json_file)
        config = json.dumps(config)
        config = json.loads(config,
                            object_hook=lambda d: SimpleNamespace(**d))
        cut_y_max = np.nanmax(self.cuts)
        cut_y_min = np.nanmin(self.cuts)
        offset = (cut_y_max - cut_y_min)*config.t_wat_offset
        t_cut_plot = self.cuts
        for i in range(len(t_cut_plot)-1):
            t_cut_plot_wf_1 = np.delete(np.array(t_cut_plot), -1, axis=0)
            t_cut_plot_wf_2 = np.delete(np.array(t_cut_plot), 0, axis=0)
            t_cut_plot_wf_delta = t_cut_plot_wf_2 - t_cut_plot_wf_1
            t_cut_plot_wf_delta = np.abs(np.min(t_cut_plot_wf_delta, axis=1))
            t_cut_plot_wf_delta = list(t_cut_plot_wf_delta)
            t_cut_plot_wf = []
            counter = 0
            for i in t_cut_plot:
                if counter == 0:
                    t_cut_plot_wf.append(i)
                else:
                    line = [a + t_cut_plot_wf_delta[counter - 1] for a in i]
                    t_cut_plot_wf.append(line)
                counter += 1
            t_cut_plot = t_cut_plot_wf

        t_cut_plot_wf = []
        if offset > 0:
            counter = 0
            for i in t_cut_plot:
                if counter == 0:
                    t_cut_plot_wf.append(i)
                else:
                    line = [a + offset*counter for a in i]
                    t_cut_plot_wf.append(line)
                counter += 1
            t_cut_plot = t_cut_plot_wf
        self.cuts = list(t_cut_plot)

    def savgol_smooth(self, window_length=3, polyorder=1, cycles=1):
        '''
        Method which applies Savitzky–Golay filter to all the stored slices.
        '''
        cuts = []
        for i in self.cuts:
            cut = i
            for j in range(cycles):
                cut = savgol_filter(cut, window_length, polyorder,
                                    mode='nearest')
            cuts.append(cut)
        self.cuts = cuts

    def derivative(self, cycles=3):
        '''
        Method which converts curves of slices to their derivatives.
        It can help to find time zero for slices with exponential behavior.
        '''
        cuts = []
        for i in self.cuts:
            cut = i
            cut = np.abs(np.gradient(cut))
            cuts.append(cut)
        self.cuts = cuts
        self.arb_u = True

    def norm_01(self):
        '''
        Method for normalization of slices to zero to one intensity.
        '''
        cuts = []
        for i in self.cuts:
            cut = i
            norm = np.min(cut)
            cut = cut - norm
            norm = np.max(cut)
            cut = cut/norm
            cuts.append(cut)
        self.cuts = cuts
        self.arb_u = True

    def norm_11(self):
        '''
        Method for normalization of delay-energy map to minus one to one
        intensity range. Either high or low limit absolute value is one.
        The other limit is scaled accordingly.
        It suits well for the difference plot.
        '''
        pos_norm = np.max(self.cuts, axis=0)
        pos_norm = np.max(pos_norm)
        neg_norm = np.min(self.cuts, axis=0)
        neg_norm = np.min(neg_norm)
        norm = [neg_norm, pos_norm]
        norm = np.max(np.abs(norm))
        cuts = []
        for i in self.cuts:
            cut = i
            cut = cut/norm
            cuts.append(cut)
        self.cuts = cuts
        self.arb_u = True

    def axs_plot(self, axs, dif_3D=False):
        '''
        Method for creating matplotlib axes for map_cut slices.
        '''
        # Loading configs from json file.
        try:
            with open('config.json', 'r') as json_file:
                config = json.load(json_file)
        except FileNotFoundError:
            with open('packages/config.json', 'r') as json_file:
                config = json.load(json_file)
        config = json.dumps(config)
        config = json.loads(config,
                            object_hook=lambda d: SimpleNamespace(**d))
        var_n = self.var_n
        var_n_r = self.var_n_r
        if self.plot_dif is True:
            self.cut_y_max = np.nanmax(self.cuts+self.dif_cuts)
            self.cut_y_min = np.nanmin(self.cuts+self.dif_cuts)
        else:
            self.cut_y_max = np.nanmax(self.cuts)
            self.cut_y_min = np.nanmin(self.cuts)

        self.cut_y_tick = (self.cut_y_max - self.cut_y_min)/config.t_n_ticks_y
        
        if self.cut_y_max - self.cut_y_min > 5:
            self.cut_y_tick = math.ceil(self.cut_y_tick)
            if self.cut_y_tick == 0:
                self.cut_y_tick = 1
        else:
            for option in [1, 0.5, 0.2, 0.1, 0.05, 0.01, 0.001, 0.0001]:
                if read_file_WESPE.rounding(self.cut_y_tick, option) > 0:
                    self.cut_y_tick = read_file_WESPE.rounding(self.cut_y_tick,
                                                               option)
                    self.cut_y_tick_decimal = read_file_WESPE.decimal_n(option)
                    self.cut_y_tick = round(self.cut_y_tick,
                                            self.cut_y_tick_decimal)
                    break
                else:
                    if option == 0.0001:
                        self.cut_y_tick = 0.0001

        self.cut_x_max = np.nanmax(self.coords)
        self.cut_x_min = np.nanmin(self.coords)
        self.cut_x_tick = (self.cut_x_max - self.cut_x_min)/config.t_n_ticks_x            

        if self.cut_x_max - self.cut_x_min > 5:
            self.cut_x_tick = math.ceil(self.cut_x_tick)
            if self.cut_x_tick == 0:
                self.cut_x_tick = 1
        else:
            for option in [1, 0.5, 0.2, 0.1, 0.05, 0.01, 0.001, 0.0001]:
                if read_file_WESPE.rounding(self.cut_x_tick, option) > 0:
                    self.cut_x_tick = read_file_WESPE.rounding(self.cut_x_tick,
                                                               option)
                    self.cut_x_tick_decimal = read_file_WESPE.decimal_n(option)
                    self.cut_x_tick = round(self.cut_x_tick,
                                            self.cut_x_tick_decimal)
                    break
                else:
                    if option == 0.0001:
                        self.cut_x_tick = 0.0001

        if self.cut_info.attrs['x_alt'] is True:
            x_label = self.cut_info.attrs['x_label_a']
            x_units = self.cut_info.attrs['x_units_a']
            x_order = self.cut_info.attrs['x_order_rec_a']
        else:
            x_label = self.cut_info.attrs['x_label']
            x_units = self.cut_info.attrs['x_units']
            x_order = self.cut_info.attrs['x_order_rec']

        if self.cut_info.attrs['y_alt'] is True:
            y_label = self.cut_info.attrs['y_label_a']
            y_units = self.cut_info.attrs['y_units_a']
            y_order = self.cut_info.attrs['y_order_rec_a']
        else:
            y_label = self.cut_info.attrs['y_label']
            y_units = self.cut_info.attrs['y_units']
            y_order = self.cut_info.attrs['y_order_rec']

        if self.axis == 'Dim_x':
            axs.set_xlabel(f'{y_label} ({y_units})', labelpad=10,
                           fontsize=config.font_size_axis)
            self.units = x_units
            self.units_r = y_units
        else:
            axs.set_xlabel(f'{x_label} ({x_units})', labelpad=10,
                           fontsize=config.font_size_axis)
            self.units = y_units
            self.units_r = x_units

        if self.fit is False:
            for i, cut in enumerate(self.cuts):
                label = f'{var_n}$_{i+1}$ = {self.positions[i]} {self.units}, '
                label = label + f'd{var_n}$_{i+1}$ = {self.deltas[i]} {self.units}'
                axs.plot(self.coords, cut, config.line_type_d,
                         color=color_dict[i],
                         label=label,
                         markersize=config.marker_size_d,
                         linewidth=config.line_width_d,
                         alpha=config.line_op_d/100)
        else:
            label = f'{var_n} = {self.positions[0]} {self.units}, '
            label = label + f'd{var_n} = {self.deltas[0]} {self.units}'
            axs.plot(self.coords, self.cuts[0], 'o',
                     markerfacecolor='none',
                     markeredgewidth=config.line_width_d*2,
                     color=color_dict[0],
                     label=label,
                     markersize=config.marker_size_d*2,
                     alpha=config.line_op_d/100)

            label = f'Fit: {self.var_n_r} = {self.center} {self.units_r}, '
            label = label + f'FWHM = {self.fwhm} {self.units_r}'
            try:
                if self.tconst1 is not None:
                    label = label + ',\n'
                    if self.amp1/(np.median(self.y_fit)) > 0.001:
                        label = label + r'$\tau_{1}$'
                        label = label + f' = {self.tconst1} {self.units_r}'
                        if self.amp2/(np.median(self.y_fit)) > 0.001:
                            label = label + r', $\tau_{2}$'
                            label = label + f' = {self.tconst2} {self.units_r}'
                    elif self.amp2/(np.median(self.y_fit)) > 0.001:
                        label = label + r'$\tau_{2}$'
                        label = label + f' = {self.tconst2} {self.units_r}'
            except:
                pass
            axs.plot(self.x_fit, self.y_fit, '-',
                     color=color_dict[1],
                     label=label,
                     linewidth=config.line_width_d*4,
                     alpha=config.line_op_d/100)
            self.fit = False
            self.tconst1 = None
            self.tconst2 = None
            self.amp1 = None
            self.amp2 = None

        if self.plot_dif is True:
            for i, cut in enumerate(self.dif_cuts):
                label = self.dif_labels[i]
                if len(self.dif_cuts) >= 2:
                    axs.plot(self.coords, cut, config.line_type_d,
                              color=color_dict[i],
                              label=label,
                              markersize=config.marker_size_d,
                              linewidth=config.line_width_d,
                              alpha=config.line_op_d/100)
                    axs.fill_between(self.coords, cut,
                                      alpha=0.5,
                                      color=color_dict[i],
                                    )
                else:
                    for j in range(2):
                        if j == 0:
                            fill = np.array(cut)
                            fill[np.where(fill<0)] = 0
                            axs.fill_between(self.coords, fill,
                                             alpha=0.5,
                                             color='red',
                                             label=label
                                            )
                        else:
                            fill = np.array(cut)
                            fill[np.where(fill>=0)] = 0
                            axs.fill_between(self.coords, fill,
                                             alpha=0.5,
                                             color='blue'
                                            )

        if self.cor is True:
            if self.en_calib == '-':
                axs.set_ylabel('Shift (arb. units)', labelpad=10,
                               fontsize=config.font_size_axis)
            else:
                axs.set_ylabel('Shift (eV)', labelpad=10,
                               fontsize=config.font_size_axis)
        else:
            if self.arb_u is True:
                axs.set_ylabel('Intensity (arb. units)', labelpad=10,
                               fontsize=config.font_size_axis)
            else:
                axs.set_ylabel('Intensity (counts)', labelpad=10,
                               fontsize=config.font_size_axis)
        # y axis
        axs.yaxis.set_major_locator(MultipleLocator(self.cut_y_tick))
        axs.set_ylim(self.cut_y_min-self.cut_y_tick/2,
                     self.cut_y_max + self.cut_y_tick)
        axs.yaxis.set_minor_locator(MultipleLocator(self.cut_y_tick /
                                                    config.t_n_ticks_minor))
        # x axis
        axs.xaxis.set_major_locator(MultipleLocator(self.cut_x_tick))
        axs.xaxis.set_minor_locator(MultipleLocator(self.cut_x_tick /
                                                    config.t_n_ticks_minor))
        axs.set_xlim(self.cut_x_min, self.cut_x_max)

        axs.grid(which='major', axis='both', color='lightgrey',
                 linestyle=config.line_type_grid_d,
                 alpha=config.line_op_grid_d/100,
                 linewidth=config.line_width_grid_d)
        axs.tick_params(axis='both', which='major',
                        length=config.t_tick_length,
                        width=config.t_tick_length/4)
        axs.tick_params(axis='both', which='minor',
                        length=config.t_tick_length/1.5,
                        width=config.t_tick_length/4)
        if x_order is True and self.axis == 'Dim_y':
            axs.invert_xaxis()
        if y_order is False and self.axis == 'Dim_x':
            axs.invert_xaxis()

    def save_cut_dat(self):
        '''
        Method for saving the delay-energy map cuts from visualization
        to ASCII format.
        One can find the saved result in the 'ASCII_output' folder.
        '''
        arr = np.array(self.cuts)
        length = arr.shape[0]
        ts = calendar.timegm(gmtime())
        date_time = datetime.fromtimestamp(ts)
        str_date_time = date_time.strftime("%d.%m.%Y_%H-%M-%S")
        path = self.file_dir + os.sep + 'ASCII_output'
        if os.path.isdir(path) is False:
            os.mkdir(path)
        path = path + os.sep + 'Cuts'
        if os.path.isdir(path) is False:
            os.mkdir(path)
        path = path + os.sep + str_date_time + os.sep
        if os.path.isdir(path) is False:
            os.mkdir(path)

        if self.cut_info.attrs['x_alt'] is True:
            x_label = self.cut_info.attrs['x_label_a']
            x_units = self.cut_info.attrs['x_units_a']
            x_order = self.cut_info.attrs['x_order_rec_a']
        else:
            x_label = self.cut_info.attrs['x_label']
            x_units = self.cut_info.attrs['x_units']
            x_order = self.cut_info.attrs['x_order_rec']

        if self.cut_info.attrs['y_alt'] is True:
            y_label = self.cut_info.attrs['y_label_a']
            y_units = self.cut_info.attrs['y_units_a']
            y_order = self.cut_info.attrs['y_order_rec_a']
        else:
            y_label = self.cut_info.attrs['y_label']
            y_units = self.cut_info.attrs['y_units']
            y_order = self.cut_info.attrs['y_order_rec']

        with open(path+"Summary.txt", "w", encoding="utf-8") as text_file:
            text_file.write(f'Loaded runs: {self.run_num_o}\n')
            text_file.write(f'Cuts across: {self.axis}\n')
            positions_str = [str(i) for i in self.positions]
            positions_str = ', '.join(positions_str)
            deltas_str = [str(i) for i in self.deltas]
            deltas_str = ', '.join(deltas_str)
            if self.axis == 'Dim_x':
                text_file.write(f'Cut positions: {positions_str} {x_units}\n')
                text_file.write(f'Cut widths: {deltas_str} {x_units}\n')
            else:
                text_file.write(f'Cut positions: {positions_str} {y_units}\n')
                text_file.write(f'Cut widths: {deltas_str} {y_units}\n')
            text_file.write('Delay-energy map parameters:\n')
            text_file.write(f'Dim_x step: {self.x_step} {x_units}\n')
            text_file.write(f'Dim_y step: {self.y_step} {y_units}\n')
            text_file.write(f'Dim_x: {x_label}\n')
            text_file.write(f'Dim_y: {y_label}\n')
            try:
                text_file.write(f'Time zero: {self.obj.t0} {y_units}\n')
            except:
                text_file.write('Time zero: None\n')

        for i in range(length):
            x = self.coords
            x = list(x)
            if self.cor is False:
                if self.axis == 'Dim_x':
                    x = [read_file_WESPE.rounding(i, self.y_step) for i in x]
                else:
                    x = [read_file_WESPE.rounding(i, self.x_step) for i in x]
            x = np.array(x)
            x = np.expand_dims(x, axis=0)
            y = arr[i]
            y = np.expand_dims(y, axis=0)
            out = np.append(x, y, axis=0)
            out = np.rot90(out)

            file_full = path
            cut_pos = np.around(self.positions[i], 2)
            cut_delta = np.around(self.deltas[i], 2)
            if x_order is True and self.axis == 'Dim_x':
                order = length - 1 - i
            elif y_order is True and self.axis == 'Dim_y':
                order = length - 1 - i
            else:
                order = i
            if len(str(order)) == len(str(length)):
                order = str(order)
            else:
                order = str(order)
                for j in range(len(str(length))-len(str(order))):
                    order = '0' + order
            file_full = file_full + f'{order}_{cut_pos} (d{cut_delta})'
            if self.axis == 'Dim_y':
                file_full = file_full + f' {y_units}.dat'
            elif self.axis == 'Dim_x':
                file_full = file_full + f' {x_units}.dat'
            np.savetxt(file_full, out, delimiter='    ')
            print(f"Saved as {file_full}")


class plot_files:
    '''
    The class for creating matplotlib plots from a list of objects.
    Uses axs_plot methods of the objects.
    '''

    def __init__(self, objects, direction='down', dpi=300,
                 fig_width=7, fig_height=5, dif_3D=False):
        # Loading configs from json file.
        try:
            with open('config.json', 'r') as json_file:
                config = json.load(json_file)
        except FileNotFoundError:
            with open('packages/config.json', 'r') as json_file:
                config = json.load(json_file)
        config = json.dumps(config)
        config = json.loads(config,
                            object_hook=lambda d: SimpleNamespace(**d))
        self.direction = direction

        if not isinstance(objects, list):
            objects = [objects]

        try:
            fig_number = len(objects)
        except TypeError:
            fig_number = 1

        matplotlib.rcParams.update({'font.size': config.font_size,
                                    'font.family': config.font_family,
                                    'axes.linewidth': config.axes_linewidth})

        fig, axs = plt.subplots(nrows=fig_number, ncols=1, sharex=False,
                                figsize=(fig_width,
                                         fig_height*fig_number),
                                dpi=dpi,
                                gridspec_kw={'hspace': 0.5*5/fig_height}
                                )

        for fig_p, object_i in enumerate(objects):
            fig_p_real = fig_p
            if direction == 'up':
                fig_p += 1
                fig_p = -fig_p

            if fig_number == 1:
                object_i.axs_plot(axs, dif_3D=dif_3D)
            else:
                object_i.axs_plot(axs[fig_p], dif_3D=dif_3D)

        self.axs = axs
        self.fig = fig

    def span_plot(self, cut_obj):
        self.Map_2D_plot = cut_obj.Map_2D_plot
        try:
            self.varied_y_step = cut_obj.varied_y_step
        except AttributeError:
            self.varied_y_step = 'ND'
        '''
        Method which highlights regions on delay-energy map related to
        slices of map_cut objects.
        '''
        for i in self.fig.axes:
            if i.yaxis.get_label()._text.split(' ')[0] != 'Intensity':
                for counter, position in enumerate(cut_obj.positions):
                    if cut_obj.map_show[counter]:
                        limit_1 = position - cut_obj.deltas[counter]/2
                        limit_2 = position + cut_obj.deltas[counter]/2
                        if cut_obj.axis == 'Dim_x':
                            i.axvspan(limit_1, limit_2,
                                      facecolor=color_dict[counter],
                                      alpha=0.25)
                            for j in [position, limit_1, limit_2]:
                                i.axvline(x=j, color=color_dict[counter],
                                          linewidth=2, zorder=10, alpha=0.4,
                                          linestyle='--')
                        else:
                            if self.varied_y_step != 'ND':
                                if self.varied_y_step is True:
                                    coord = self.Map_2D_plot.coords['Dim_y']
                                    position = coord.sel(Dim_y=position, method="nearest")
                                    limit_1 = coord.sel(Dim_y=limit_1, method="nearest")
                                    limit_2 = coord.sel(Dim_y=limit_2, method="nearest")
                                    position = coord.where(coord==position,drop=True)
                                    limit_1 = coord.where(coord==limit_1,drop=True)
                                    limit_2 = coord.where(coord==limit_2,drop=True)
                                    position = position['Delay index'].values
                                    limit_1 = limit_1['Delay index'].values
                                    limit_2 = limit_2['Delay index'].values
                            i.axhspan(limit_1, limit_2,
                                      facecolor=color_dict[counter],
                                      alpha=0.25)
                            for j in [position, limit_1, limit_2]:
                                i.axhline(y=j, color=color_dict[counter],
                                          linewidth=2, zorder=10, alpha=0.4,
                                          linestyle='--')

    def legend_plot(self):
        '''
        Method for adding a legend to the figure.
        '''
        for i in self.fig.axes:
            if i.yaxis.get_label()._text.split(' ')[0] == 'Intensity':
                if i.xaxis.get_label()._text != '':
                    i.legend(loc='best',
                             fontsize=config.font_size-2, markerscale=2)
            elif i.yaxis.get_label()._text.split(' ')[0] == 'Shift':
                if i.xaxis.get_label()._text != '':
                    i.legend(loc='best',
                             fontsize=config.font_size-2, markerscale=2)

    def set_y_tick(self, major, minor):
        for i in self.fig.axes:
            if i.yaxis.get_label()._text.split(' ')[0] == 'Intensity':
                if i.xaxis.get_label()._text != '':
                    i.yaxis.set_major_locator(MultipleLocator(major))
                    i.yaxis.set_minor_locator(MultipleLocator(minor))
            elif i.yaxis.get_label()._text.split(' ')[0] == 'Shift':
                if i.xaxis.get_label()._text != '':
                    i.yaxis.set_major_locator(MultipleLocator(major))
                    i.yaxis.set_minor_locator(MultipleLocator(minor))

    def set_x_tick(self, major, minor):
        for i in self.fig.axes:
            if i.yaxis.get_label()._text.split(' ')[0] == 'Intensity':
                if i.xaxis.get_label()._text != '':
                    i.xaxis.set_major_locator(MultipleLocator(major))
                    i.xaxis.set_minor_locator(MultipleLocator(minor))
            elif i.yaxis.get_label()._text.split(' ')[0] == 'Shift':
                if i.xaxis.get_label()._text != '':
                    i.xaxis.set_major_locator(MultipleLocator(major))
                    i.xaxis.set_minor_locator(MultipleLocator(minor))

    def set_y_lim(self, lim_list):
        for i in self.fig.axes:
            if i.yaxis.get_label()._text.split(' ')[0] == 'Intensity':
                if i.xaxis.get_label()._text != '':
                    i.set_ylim(min(lim_list), max(lim_list))
            elif i.yaxis.get_label()._text.split(' ')[0] == 'Shift':
                if i.xaxis.get_label()._text != '':
                    i.set_ylim(min(lim_list), max(lim_list))

    def set_x_lim(self, lim_list):
        for i in self.fig.axes:
            if i.yaxis.get_label()._text.split(' ')[0] == 'Intensity':
                if i.xaxis.get_label()._text != '':
                    i.set_xlim(min(lim_list), max(lim_list))
            elif i.yaxis.get_label()._text.split(' ')[0] == 'Shift':
                if i.xaxis.get_label()._text != '':
                    i.set_xlim(min(lim_list), max(lim_list))


if __name__ == "__main__":
    # Loading configs from json file.
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)
    try:
        with open('config.json', 'r') as json_file:
            config = json.load(json_file)
    except FileNotFoundError:
        with open('packages/config.json', 'r') as json_file:
            config = json.load(json_file)
    config = json.dumps(config)
    config = json.loads(config,
                        object_hook=lambda d: SimpleNamespace(**d))

    file_dir = r'D:\Data\MM December 23'
    file_dir = r'D:\Data\HEXTOF'
    # file_dir = r'D:\Data\SXP'
    file_dir = r'D:\Data\ALS 24'
    file_dir = r'D:\Data\Fink Sep.2021'
    # file_dir = r'D:\Data\10_BESSY_Fink'
    run_numbers = ['run_50032_50033_50041']
    # run_numbers = [44824,44825,44826,44827]
    run_numbers = ['run_51663']
    run_numbers = ['PS_Scan_240417-run104']
    run_numbers = ['W_0058w058_Fermi_surface_Na']
    run_numbers = ['K_0008_008']
    
    listdir = sorted(os.listdir(file_dir))
    listdir_static = listdir.copy()
    for i in listdir_static:
        if '.ibw' not in i:
            listdir.remove(i)
        # if '078' not in i:
        #     listdir.remove(i)
            
    issues=[]
    
    for j in listdir:
        try:
            b = create_batch_ibw(file_dir, [j.replace('.ibw', '')])
            for i in b.batch_list:
                i.create_map()
            b.create_map()
            # c = map_cut(b, 6.5, [10], axis='Dim_y', approach='mean')
            p = plot_files([b], dif_3D=False)
            plt.show()
            # p.span_plot(c)
        except:
            issues.append(j)

else:
    # Loading configs from json file.
    try:
        with open('config.json', 'r') as json_file:
            config = json.load(json_file)
    except FileNotFoundError:
        with open('packages/config.json', 'r') as json_file:
            config = json.load(json_file)
    config = json.dumps(config)
    config = json.loads(config,
                        object_hook=lambda d: SimpleNamespace(**d))
