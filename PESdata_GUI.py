# -*- coding: utf-8 -*-
"""
Created on Thu Jul 7 18:51:36 2022

author: Dr. Dmitrii Potorochin
email:  dmitrii.potorochin@desy.de
        dmitrii.potorochin@physik.tu-freiberg.de
        dm.potorochin@gmail.com
"""
# Switching off internal Kivy logger to exclude spamming
import os
os.environ["KIVY_NO_CONSOLELOG"] = "1"
# Configs must be set up before loading the app
from kivy.config import Config
Config.set('graphics', 'resizable', True)
Config.set('graphics', 'fullscreen', 'False')
Config.set('graphics', 'window_state', 'maximized')
Config.write()
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
import json
import numpy as np
from IPython import get_ipython
import calendar
from time import gmtime
from datetime import datetime
from types import SimpleNamespace
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
from packages.PESdata_OOP import map_cut
from packages.PESdata_OOP import plot_files

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

# Loading the active module
if config.file_type == 'CasaXPS':
    try:
        from packages.PESdata_OOP import create_batch_CasaXPS as create_batch
    except:
        from packages.PESdata_OOP import create_batch_WESPE as create_batch
elif config.file_type == 'ibw':
    try:
        from packages.PESdata_OOP import create_batch_ibw as create_batch
    except:
        from packages.PESdata_OOP import create_batch_WESPE as create_batch
elif config.file_type == 'ALS':
    from packages.PESdata_OOP import create_batch_ALS as create_batch
elif config.file_type == 'MM':
    from packages.PESdata_OOP import create_batch_MM as create_batch
else:
    from packages.PESdata_OOP import create_batch_WESPE as create_batch

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


def scan_dir(file_dir):
    '''
    This function determines what module must be loaded for the input data.

    '''
    dir_dict = {}
    count_dict = {}
    for k in ['.ibw', '.txt', '.h5', '.mat', '.4q', '.1q', '.hextof', '.sxp']:
        dir_dump = []
        counter = 0
        for i in os.walk(file_dir):
            for j in i[2]:
                if k in j.lower():
                    dir_dump.append(i[0])
                    counter += 1
        dir_dict[k] = dir_dump
        count_dict[k] = counter
    if len(dir_dict['.mat']) > 0:
        file_type = 'WESPE'
    elif len(dir_dict['.4q']) > 0:
        file_type = 'WESPE'
    elif len(dir_dict['.1q']) > 0:
        file_type = 'WESPE'
    elif len(dir_dict['.hextof']) > 0 or len(dir_dict['.sxp']) > 0:
        file_type = 'MM'
    else:
        if len(dir_dict['.h5']) > 0:
            file_type = 'MM'
            if count_dict['.h5']/len(set(dir_dict['.h5'])) > 5:
                file_type = 'ALS'
        else:
            if len(dir_dict['.ibw']) > 0:
                file_type = 'ibw'
            else:
                if len(dir_dict['.txt']) > 0:
                    file_type = 'CasaXPS'
                else:
                    file_type = 'UNKNOWN'
    return file_type


class MainApp(App):
    def build(self):
        # returns a window object with all it's widgets
        self.title = 'PESdata'
        self.window = BoxLayout(orientation='vertical',
                                spacing=10, padding=10)
        self.top = BoxLayout(orientation='horizontal',
                             size_hint=(1, 0.5),
                             pos_hint={"center_x": 0.55, "center_y": 0.575})
        self.box1 = BoxLayout(orientation='horizontal',
                              size_hint=(1, 0.5))
        self.box2 = BoxLayout(orientation='horizontal',
                              size_hint=(1, 0.5))
        self.box3 = BoxLayout(orientation='horizontal',
                              size_hint=(1, 0.5))
        self.box4 = BoxLayout(orientation='horizontal',
                              size_hint=(1, 0.5))
        self.box5 = BoxLayout(orientation='horizontal',
                              size_hint=(1, 0.5))
        self.box6 = BoxLayout(orientation='horizontal',
                              size_hint=(1, 0.5))
        self.box7 = BoxLayout(orientation='horizontal',
                              size_hint=(1, 0.5))
        self.box8 = BoxLayout(orientation='horizontal',
                              size_hint=(1, 0.5))
        self.box9 = BoxLayout(orientation='horizontal',
                              size_hint=(1, 0.5))
        self.box10 = BoxLayout(orientation='horizontal',
                               size_hint=(1, 0.5))
        self.box11 = BoxLayout(orientation='horizontal',
                               size_hint=(1, 0.5))
        self.box12 = BoxLayout(orientation='horizontal',
                               size_hint=(1, 0.5))
        self.box13 = BoxLayout(orientation='horizontal',
                               size_hint=(1, 0.5))
        self.box14 = BoxLayout(orientation='horizontal',
                               size_hint=(1, 0.5))
        self.box15 = BoxLayout(orientation='horizontal',
                               size_hint=(1, 0.5))
        self.box16 = BoxLayout(orientation='horizontal',
                               size_hint=(1, 0.1))

        self.window.size_hint = (0.85, 0.95)
        self.window.pos_hint = {"center_x": 0.5, "center_y": 0.5}

        '''
        TITLE
        '''
        self.title_text = Label(
                            text="PESdata",
                            font_size=config.kivy_font_size_title*2,
                            color=config.kivy_color_title,
                            font_name=config.kivy_font
                            )
        self.top.add_widget(self.title_text)

        '''
        DIRECTORY INPUT
        '''
        # label
        self.directory_label = Label(
                            text="File directory",
                            font_size=config.kivy_font_size_title,
                            color=config.kivy_color_title,
                            bold=True,
                            pos_hint={"center_x": 0.5, "center_y": 0.5},
                            size_hint=(0.5, 1),
                            font_name=config.kivy_font
                            )
        self.box1.add_widget(self.directory_label)

        # input
        self.directory_input = TextInput(
                        text=config.file_dir,
                        multiline=True,
                        pos_hint={"center_x": 0.5, "center_y": 0.5},
                        size_hint=(0.8, 0.8),
                        font_name=config.kivy_font,
                        font_size=config.kivy_font_size
                        )
        self.box1.add_widget(self.directory_input)
        self.directory_input.bind(on_double_tap=self.suggestion_dir)

        self.DLD_label = Label(
                            text="Detector:",
                            font_size=config.kivy_font_size_title,
                            color=config.kivy_color_title,
                            bold=True,
                            pos_hint={"center_x": 0.5, "center_y": 0.5},
                            size_hint=(0.2, 1),
                            font_name=config.kivy_font
                            )
        if config.file_type == 'WESPE':
            self.box1.add_widget(self.DLD_label)
            self.directory_input.size_hint = (0.8, 0.8)
        else:
            self.directory_input.size_hint = (1, 0.8)

        '''
        SCAN NAME INPUT
        '''
        # label
        self.run_numbers_label = Label(
                            text="Run numbers",
                            font_size=config.kivy_font_size_title,
                            color=config.kivy_color_title,
                            bold=True,
                            pos_hint={"center_x": 0.5, "center_y": 0.5},
                            size_hint=(0.5, 1),
                            font_name=config.kivy_font
                            )
        self.box2.add_widget(self.run_numbers_label)
        # input
        self.run_numbers_input = TextInput(
                        text=config.run_numbers,
                        multiline=True,
                        size_hint=(0.8, 0.8),
                        pos_hint={"center_x": 0.5, "center_y": 0.5},
                        font_name=config.kivy_font,
                        font_size=config.kivy_font_size
                        )
        self.box2.add_widget(self.run_numbers_input)
        self.run_numbers_input.bind(on_double_tap=self.suggestion)

        self.DLD_toggle = ToggleButton(text='DLD4Q',
                                       state='down',
                                       font_name=config.kivy_font,
                                       font_size=config.kivy_font_size,
                                       pos_hint={"center_x": 0.5,
                                                 "center_y": 0.49},
                                       size_hint=(0.2, 0.8)
                                       )
        self.DLD_toggle.bind(on_press=self.callback_DLD)
        if config.file_type == 'WESPE':
            self.box2.add_widget(self.DLD_toggle)
            self.run_numbers_input.size_hint = (0.8, 0.8)
        else:
            self.run_numbers_input.size_hint = (1, 0.8)
        '''UPLOAD RUNS BUTTON'''
        self.upload_runs = Button(
                          text="Upload runs",
                          bold=True,
                          background_color=config.kivy_color_button,
                          font_name=config.kivy_font,
                          font_size=config.kivy_font_size_title,
                          color=config.kivy_color_white,
                          size_hint=(0.8, 1),
                          pos_hint={"center_x": 0.5, "center_y": 0.5}
                          )

        self.upload_runs.bind(on_press=self.callback_1_0)
        self.upload_runs.bind(on_release=self.callback_1)
        self.box3.add_widget(self.upload_runs)

        settings_popup = Button(text='App settings',
                                bold=True,
                                background_color='#09ff00',
                                font_name=config.kivy_font,
                                font_size=config.kivy_font_size_title,
                                color=config.kivy_color_white,
                                size_hint=(0.2, 1),
                                pos_hint={"center_x": 0.5, "center_y": 0.5}
                                )
        settings_popup.bind(on_press=self.settings_popup_callback)
        self.box3.add_widget(settings_popup)
        '''
        CREATE MAP PARAMETERS
        '''
        # label
        d1 = Label(
                    text="2D Histogram\nparameters",
                    font_size=config.kivy_font_size_title,
                    color=config.kivy_color_title,
                    bold=True,
                    font_name=config.kivy_font,
                    size_hint=(0.5, 1),
                    halign='center'
                    )
        # t0 toggle
        self.d2 = ToggleButton(text='T0: OFF',
                               state='normal',
                               font_name=config.kivy_font,
                               font_size=config.kivy_font_size,
                               pos_hint={"center_x": 0.5, "center_y": 0.5},
                               size_hint=(0.275, 1)
                               )
        self.d2.bind(on_press=self.callback_d2)
        # t0 input
        self.d3 = TextInput(text=config.t_zero,
                            multiline=False,
                            pos_hint={"center_x": 0.5, "center_y": 0.5},
                            size_hint=(0.15, 0.7),
                            font_name=config.kivy_font,
                            font_size=config.kivy_font_size
                            )

        d3_l = Label(text='ps',
                     font_size=config.kivy_font_size_title,
                     color=config.kivy_color_white,
                     font_name=config.kivy_font,
                     pos_hint={"center_x": 0.5, "center_y": 0.5},
                     size_hint=(0.075, 1),
                     halign='center'
                     )
        # KE button
        self.d4 = ToggleButton(text='Kinetic energy',
                               group='energy_axis_input',
                               state='down',
                               pos_hint={"center_x": 0.5, "center_y": 0.5},
                               size_hint=(0.25, 1),
                               font_name=config.kivy_font,
                               font_size=config.kivy_font_size,
                               allow_no_selection=False
                               )
        # BE button
        self.d5 = ToggleButton(text='Binding energy',
                               group='energy_axis_input',
                               pos_hint={"center_x": 0.5, "center_y": 0.5},
                               size_hint=(0.25, 1),
                               font_name=config.kivy_font,
                               font_size=config.kivy_font_size,
                               allow_no_selection=False
                               )
        # add widgets to the line
        self.box4.add_widget(d1)
        self.box4.add_widget(self.d2)
        self.box4.add_widget(self.d3)
        self.box4.add_widget(d3_l)
        self.box4.add_widget(self.d4)
        self.box4.add_widget(self.d5)
        '''
        Binning
        '''
        # label
        self.e1 = Label(
                        text="Binning",
                        font_size=config.kivy_font_size_title,
                        color=config.kivy_color_title,
                        bold=True,
                        font_name=config.kivy_font,
                        size_hint=(0.5, 1)
                        )
        # E step label
        self.e2 = Label(text='Energy step',
                        font_size=config.kivy_font_size_title,
                        color=config.kivy_color_white,
                        font_name=config.kivy_font,
                        pos_hint={"center_x": 0.5, "center_y": 0.5},
                        size_hint=(0.275, 1),
                        halign='center'
                        )
        # E step input
        self.e3 = TextInput(text='0.05',
                            multiline=False,
                            pos_hint={"center_x": 0.5, "center_y": 0.5},
                            size_hint=(0.15, 0.7),
                            font_name=config.kivy_font,
                            font_size=config.kivy_font_size
                            )
        self.e4 = Label(text='eV',
                        font_size=config.kivy_font_size_title,
                        color=config.kivy_color_white,
                        font_name=config.kivy_font,
                        pos_hint={"center_x": 0.5, "center_y": 0.5},
                        size_hint=(0.075, 1),
                        halign='center'
                        )
        # T step label
        self.e5 = Label(text='Time step',
                        font_size=config.kivy_font_size_title,
                        color=config.kivy_color_white,
                        font_name=config.kivy_font,
                        pos_hint={"center_x": 0.5, "center_y": 0.5},
                        size_hint=(0.275, 1),
                        halign='center'
                        )
        # T step input
        self.e6 = TextInput(text='0.1',
                            multiline=False,
                            pos_hint={"center_x": 0.5, "center_y": 0.5},
                            size_hint=(0.15, 0.7),
                            font_name=config.kivy_font,
                            font_size=config.kivy_font_size
                            )
        self.e7 = Label(text='ps',
                        font_size=config.kivy_font_size_title,
                        color=config.kivy_color_white,
                        font_name=config.kivy_font,
                        pos_hint={"center_x": 0.5, "center_y": 0.5},
                        size_hint=(0.075, 1),
                        halign='center'
                        )

        # add widgets to the line
        self.box5.add_widget(self.e1)
        self.box5.add_widget(self.e2)
        self.box5.add_widget(self.e3)
        self.box5.add_widget(self.e4)
        self.box5.add_widget(self.e5)
        self.box5.add_widget(self.e6)
        self.box5.add_widget(self.e7)
        '''
        BUNCH FILTERING
        '''
        # label
        f1 = Label(
                    text="Bunch filtering",
                    font_size=config.kivy_font_size_title,
                    color=config.kivy_color_title,
                    bold=True,
                    font_name=config.kivy_font,
                    size_hint=(0.5, 1),
                    halign='center'
                    )
        # MacroB toggle
        self.f2 = ToggleButton(text='MacroB: OFF',
                               state='normal',
                               font_name=config.kivy_font,
                               font_size=config.kivy_font_size,
                               pos_hint={"center_x": 0.5, "center_y": 0.5},
                               size_hint=(0.275, 1)
                               )
        self.f2.bind(on_press=self.callback_f2)
        # MacroB input
        self.f3 = TextInput(text='0,80',
                            multiline=False,
                            pos_hint={"center_x": 0.5, "center_y": 0.5},
                            size_hint=(0.15, 0.7),
                            font_name=config.kivy_font,
                            font_size=config.kivy_font_size
                            )
        f4 = Label(text='%',
                   font_size=config.kivy_font_size_title,
                   color=config.kivy_color_white,
                   font_name=config.kivy_font,
                   pos_hint={"center_x": 0.5, "center_y": 0.5},
                   size_hint=(0.075, 1),
                   halign='center'
                   )
        # MicroB toggle
        self.f5 = ToggleButton(text='MicroB: OFF',
                               state='normal',
                               font_name=config.kivy_font,
                               font_size=config.kivy_font_size,
                               pos_hint={"center_x": 0.5, "center_y": 0.5},
                               size_hint=(0.275, 1)
                               )
        self.f5.bind(on_press=self.callback_f5)
        # MicroB input
        self.f6 = TextInput(text='0,400',
                            multiline=False,
                            pos_hint={"center_x": 0.5, "center_y": 0.5},
                            size_hint=(0.15, 0.7),
                            font_name=config.kivy_font,
                            font_size=config.kivy_font_size
                            )
        f7 = Label(text='u.',
                   font_size=config.kivy_font_size_title,
                   color=config.kivy_color_white,
                   font_name=config.kivy_font,
                   pos_hint={"center_x": 0.5, "center_y": 0.5},
                   size_hint=(0.075, 1),
                   halign='center'
                   )
        # add widgets to the line
        self.box6.add_widget(f1)
        self.box6.add_widget(self.f2)
        self.box6.add_widget(self.f3)
        self.box6.add_widget(f4)
        self.box6.add_widget(self.f5)
        self.box6.add_widget(self.f6)
        self.box6.add_widget(f7)
        '''CREATE MAP BUTTON'''
        self.create_map = Button(text="Compute 2D Histogram",
                                 bold=True,
                                 background_color=config.kivy_color_button,
                                 font_name=config.kivy_font,
                                 font_size=config.kivy_font_size_title,
                                 color=config.kivy_color_white,
                                 size_hint=(0.8, 1),
                                 pos_hint={"center_x": 0.5, "center_y": 0.5}
                                 )
        self.create_map.bind(on_press=self.callback_2_0)
        self.create_map.bind(on_release=self.callback_2)
        self.box7.add_widget(self.create_map)

        self.map_mode = ToggleButton(
                                     text='Mode: Time delay',
                                     bold=True,
                                     background_color=config.kivy_color_button,
                                     font_name=config.kivy_font,
                                     font_size=config.kivy_font_size_title,
                                     color=config.kivy_color_white,
                                     size_hint=(0.2, 1),
                                     pos_hint={"center_x": 0.5,
                                               "center_y": 0.5},
                                     halign='center'
                                     )
        self.map_mode.bind(on_press=self.create_plot_mode_callback)

        self.hist_mode_label = Button(text="Select dims:",
                                      bold=True,
                                      background_color=config.kivy_color_button,
                                      font_name=config.kivy_font,
                                      font_size=config.kivy_font_size_title,
                                      color=config.kivy_color_white,
                                      size_hint=(0.15, 1),
                                      pos_hint={"center_x": 0.5,
                                                "center_y": 0.5}
                                      )

        self.hist_mode = TextInput(text='td',
                                   multiline=False,
                                   pos_hint={"center_x": 0.5,
                                             "center_y": 0.5165},
                                   size_hint=(0.05, 0.974),
                                   font_name=config.kivy_font,
                                   font_size=config.kivy_font_size_title,
                                   background_color='#09ff00',
                                   foreground_color=config.kivy_color_white,
                                   padding=[0, 22, 0, 0],
                                   background_normal='packages/normal.png',
                                   background_active='packages/normal.png',
                                   halign='center',
                                   )
        if config.file_type == 'MM':
            self.box7.add_widget(self.hist_mode_label)
            self.box7.add_widget(self.hist_mode)
        elif config.file_type == 'WESPE' or config.file_type == 'ALS':
            self.box7.add_widget(self.map_mode)

        '''
        DELAY-ENERGY PLOT PARAMETERS
        '''
        # label
        h1 = Label(
                    text="Plot parameters",
                    font_size=config.kivy_font_size_title,
                    color=config.kivy_color_title,
                    bold=True,
                    font_name=config.kivy_font,
                    size_hint=(0.5, 1),
                    halign='center'
                    )
        # t0 toggle
        self.h2 = ToggleButton(text='T0: OFF',
                               state='normal',
                               font_name=config.kivy_font,
                               font_size=config.kivy_font_size,
                               pos_hint={"center_x": 0.5, "center_y": 0.5},
                               size_hint=(0.2, 1)
                               )
        self.h2.bind(on_press=self.callback_h2)
        # Dif map toggle
        self.h3 = ToggleButton(text='Dif map: OFF',
                               state='normal',
                               font_name=config.kivy_font,
                               font_size=config.kivy_font_size,
                               pos_hint={"center_x": 0.5, "center_y": 0.5},
                               size_hint=(0.3, 1)
                               )
        self.h3.bind(on_press=self.callback_h3)
        # KE
        self.h4 = ToggleButton(text='Kinetic energy',
                               state='down',
                               font_name=config.kivy_font,
                               font_size=config.kivy_font_size,
                               pos_hint={"center_x": 0.5, "center_y": 0.5},
                               size_hint=(0.25, 1),
                               group='KE/BE plot',
                               allow_no_selection=False
                               )
        # BE
        self.h5 = ToggleButton(text='Binding energy',
                               state='normal',
                               font_name=config.kivy_font,
                               font_size=config.kivy_font_size,
                               pos_hint={"center_x": 0.5, "center_y": 0.5},
                               size_hint=(0.25, 1),
                               group='KE/BE plot',
                               allow_no_selection=False
                               )
        # add widgets to the line
        self.box8.add_widget(h1)
        self.box8.add_widget(self.h2)
        self.box8.add_widget(self.h3)
        self.box8.add_widget(self.h4)
        self.box8.add_widget(self.h5)
        '''
        ROI PLOT PARAMETERS
        '''
        # label
        self.i1 = Label(
                    text="Delay/Energy ROI",
                    font_size=config.kivy_font_size_title,
                    color=config.kivy_color_title,
                    bold=True,
                    font_name=config.kivy_font,
                    size_hint=(0.5, 1),
                    halign='center'
                    )
        # Energy ROI toggle
        self.i2 = ToggleButton(text='Energy: OFF',
                               state='normal',
                               font_name=config.kivy_font,
                               font_size=config.kivy_font_size,
                               pos_hint={"center_x": 0.5, "center_y": 0.5},
                               size_hint=(0.275, 1)
                               )
        # Energy ROI input
        self.i3 = TextInput(text=config.en_ROI,
                            multiline=False,
                            pos_hint={"center_x": 0.5, "center_y": 0.5},
                            size_hint=(0.15, 0.7),
                            font_name=config.kivy_font,
                            font_size=config.kivy_font_size
                            )
        self.i4 = Label(text='eV',
                        font_size=config.kivy_font_size_title,
                        color=config.kivy_color_white,
                        font_name=config.kivy_font,
                        pos_hint={"center_x": 0.5, "center_y": 0.5},
                        size_hint=(0.075, 1),
                        halign='center'
                        )
        # Delay ROI toggle
        self.i5 = ToggleButton(text='Delay: OFF',
                               state='normal',
                               font_name=config.kivy_font,
                               font_size=config.kivy_font_size,
                               pos_hint={"center_x": 0.5, "center_y": 0.5},
                               size_hint=(0.275, 1)
                               )
        # Delay ROI input
        self.i6 = TextInput(text=config.t_ROI,
                            multiline=False,
                            pos_hint={"center_x": 0.5, "center_y": 0.5},
                            size_hint=(0.15, 0.7),
                            font_name=config.kivy_font,
                            font_size=config.kivy_font_size
                            )
        self.i7 = Label(text='ps',
                        font_size=config.kivy_font_size_title,
                        color=config.kivy_color_white,
                        font_name=config.kivy_font,
                        pos_hint={"center_x": 0.5, "center_y": 0.5},
                        size_hint=(0.075, 1),
                        halign='center'
                        )

        # add widgets to the line
        self.box9.add_widget(self.i1)
        self.box9.add_widget(self.i2)
        self.box9.add_widget(self.i3)
        self.box9.add_widget(self.i4)
        self.box9.add_widget(self.i5)
        self.box9.add_widget(self.i6)
        self.box9.add_widget(self.i7)

        '''
        NORMALIZING PARAMETERS
        '''
        # label
        j1 = Label(
                    text="Normalizing",
                    font_size=config.kivy_font_size_title,
                    color=config.kivy_color_title,
                    bold=True,
                    font_name=config.kivy_font,
                    size_hint=(0.5, 1),
                    halign='center'
                    )
        # Total electrons norm
        self.j2 = ToggleButton(text='Total electrons: ON',
                               state='down',
                               font_name=config.kivy_font,
                               font_size=config.kivy_font_size,
                               pos_hint={"center_x": 0.5, "center_y": 0.5},
                               size_hint=(0.333, 1)
                               )
        self.j2.bind(on_press=self.callback_j2)
        # [0,1] norm
        self.j3 = ToggleButton(text='[0,1]: OFF',
                               state='normal',
                               font_name=config.kivy_font,
                               font_size=config.kivy_font_size,
                               pos_hint={"center_x": 0.5, "center_y": 0.5},
                               size_hint=(0.333, 1)
                               )
        self.j3.bind(on_press=self.callback_j3)
        # [-1,1] norm
        self.j4 = ToggleButton(text='[-1,1]: OFF',
                               state='normal',
                               font_name=config.kivy_font,
                               font_size=config.kivy_font_size,
                               pos_hint={"center_x": 0.5, "center_y": 0.5},
                               size_hint=(0.333, 1)
                               )
        self.j4.bind(on_press=self.callback_j4)

        # add widgets to the line
        self.box10.add_widget(j1)
        self.box10.add_widget(self.j2)
        self.box10.add_widget(self.j3)
        self.box10.add_widget(self.j4)

        '''CREATE PLOT'''
        self.create_plot = Button(text="Create 2D Histogram visualization",
                                  bold=True,
                                  background_color=config.kivy_color_button,
                                  font_name=config.kivy_font,
                                  font_size=config.kivy_font_size_title,
                                  color=config.kivy_color_white,
                                  size_hint=(0.8, 1),
                                  pos_hint={"center_x": 0.5, "center_y": 0.5}
                                  )
        self.create_plot.bind(on_press=self.callback_3_0)
        self.create_plot.bind(on_release=self.callback_3)
        self.box11.add_widget(self.create_plot)

        save_map_dat = Button(text="Save ASCII",
                              bold=True,
                              background_color=config.kivy_color_button,
                              font_name=config.kivy_font,
                              font_size=config.kivy_font_size_title,
                              color=config.kivy_color_white,
                              size_hint=(0.2, 1),
                              pos_hint={"center_x": 0.5, "center_y": 0.5}
                              )
        save_map_dat.bind(on_press=self.callback_save_map_dat)
        self.box11.add_widget(save_map_dat)

        self.create_plot_mode = ToggleButton(
                                     text='Mode: PyQt',
                                     bold=True,
                                     background_color=config.kivy_color_button,
                                     font_name=config.kivy_font,
                                     font_size=config.kivy_font_size_title,
                                     color=config.kivy_color_white,
                                     size_hint=(0.2, 1),
                                     pos_hint={"center_x": 0.5,
                                               "center_y": 0.5},
                                     halign='center'
                                     )
        self.create_plot_mode.bind(on_press=self.create_plot_mode_callback)
        self.box11.add_widget(self.create_plot_mode)

        '''
        CREATE MAP CUT PARAMETERS
        '''
        # label
        k1 = Label(
                    text="Slice mode",
                    font_size=config.kivy_font_size_title,
                    color=config.kivy_color_title,
                    bold=True,
                    font_name=config.kivy_font,
                    size_hint=(0.5, 1),
                    halign='center'
                    )

        # Time axis toggle
        self.k2 = ToggleButton(text='Time axis',
                               state='down',
                               pos_hint={"center_x": 0.5, "center_y": 0.5},
                               size_hint=(0.34, 1),
                               background_normal='packages/normal.png',
                               background_down='packages/down.png',
                               font_name=config.kivy_font,
                               font_size=config.kivy_font_size_title,
                               opacity=0.85,
                               )
        # Add waterfall toggle
        self.k4 = ToggleButton(text='Waterfall: OFF',
                               state='normal',
                               pos_hint={"center_x": 0.5, "center_y": 0.5},
                               size_hint=(0.22, 1),
                               font_name=config.kivy_font,
                               font_size=config.kivy_font_size,
                               )
        self.k4.bind(on_press=self.callback_k4)
        # Add map
        self.k5 = ToggleButton(text='Add map: ON',
                               state='down',
                               font_name=config.kivy_font,
                               font_size=config.kivy_font_size,
                               pos_hint={"center_x": 0.5, "center_y": 0.5},
                               size_hint=(0.22, 1)
                               )
        self.k5.bind(on_press=self.callback_k5)
        # Add legend
        self.k6 = ToggleButton(text='Legend: ON',
                               state='down',
                               font_name=config.kivy_font,
                               font_size=config.kivy_font_size,
                               pos_hint={"center_x": 0.5, "center_y": 0.5},
                               size_hint=(0.22, 1)
                               )
        self.k6.bind(on_press=self.callback_k6)
        # add widgets to the line
        self.box12.add_widget(k1)
        self.box12.add_widget(self.k2)
        self.box12.add_widget(self.k4)
        self.box12.add_widget(self.k5)
        self.box12.add_widget(self.k6)
        '''
        Cut parameters
        '''
        # label
        l1 = Label(
                    text="Slice parameters",
                    font_size=config.kivy_font_size_title,
                    color=config.kivy_color_title,
                    bold=True,
                    font_name=config.kivy_font,
                    size_hint=(0.5, 1)
                    )
        # E step label
        l2 = Label(text='Position',
                   font_size=config.kivy_font_size_title,
                   color=config.kivy_color_white,
                   font_name=config.kivy_font,
                   pos_hint={"center_x": 0.5, "center_y": 0.5},
                   size_hint=(0.19, 1),
                   halign='center'
                   )
        # E step input
        self.l3 = TextInput(text=config.cut_pos,
                            multiline=False,
                            pos_hint={"center_x": 0.5, "center_y": 0.5},
                            size_hint=(0.12, 0.7),
                            font_name=config.kivy_font,
                            font_size=config.kivy_font_size
                            )
        l4 = Label(text='ps/eV',
                   font_size=config.kivy_font_size_title,
                   color=config.kivy_color_white,
                   font_name=config.kivy_font,
                   pos_hint={"center_x": 0.5, "center_y": 0.5},
                   size_hint=(0.11, 1),
                   halign='center'
                   )
        # T step label
        l5 = Label(text='Width',
                   font_size=config.kivy_font_size_title,
                   color=config.kivy_color_white,
                   font_name=config.kivy_font,
                   pos_hint={"center_x": 0.5, "center_y": 0.5},
                   size_hint=(0.13, 1),
                   halign='center'
                   )
        # T step input
        self.l6 = TextInput(text=config.cut_width,
                            multiline=False,
                            pos_hint={"center_x": 0.5, "center_y": 0.5},
                            size_hint=(0.11, 0.7),
                            font_name=config.kivy_font,
                            font_size=config.kivy_font_size
                            )
        l7 = Label(text='ps/eV',
                   font_size=config.kivy_font_size_title,
                   color=config.kivy_color_white,
                   font_name=config.kivy_font,
                   pos_hint={"center_x": 0.5, "center_y": 0.5},
                   size_hint=(0.12, 1),
                   halign='center'
                   )
        # Add difference toggle
        self.k3 = ToggleButton(text='Difference: OFF',
                               state='normal',
                               pos_hint={"center_x": 0.5, "center_y": 0.5},
                               size_hint=(0.22, 1),
                               font_name=config.kivy_font,
                               font_size=config.kivy_font_size,
                               )
        self.k3.bind(on_press=self.callback_k3)

        # add widgets to the line
        self.box13.add_widget(l1)
        self.box13.add_widget(l2)
        self.box13.add_widget(self.l3)
        self.box13.add_widget(l4)
        self.box13.add_widget(l5)
        self.box13.add_widget(self.l6)
        self.box13.add_widget(l7)
        self.box13.add_widget(self.k3)

        '''
        NORMALIZING PARAMETERS
        '''
        # label
        m1 = Label(
                    text="Data treatment",
                    font_size=config.kivy_font_size_title,
                    color=config.kivy_color_title,
                    bold=True,
                    font_name=config.kivy_font,
                    size_hint=(0.5, 1),
                    halign='center'
                    )
        # [0,1] norm
        self.m2 = ToggleButton(text='Norm [0,1]: OFF',
                               state='normal',
                               font_name=config.kivy_font,
                               font_size=config.kivy_font_size,
                               pos_hint={"center_x": 0.5, "center_y": 0.5},
                               size_hint=(0.2, 1)
                               )
        self.m2.bind(on_press=self.callback_m2)
        # [-1,1] norm
        self.m3 = ToggleButton(text='Norm [-1,1]: OFF',
                               state='normal',
                               font_name=config.kivy_font,
                               font_size=config.kivy_font_size,
                               pos_hint={"center_x": 0.5, "center_y": 0.5},
                               size_hint=(0.2, 1)
                               )
        self.m3.bind(on_press=self.callback_m3)
        # Smoothing
        self.m4 = ToggleButton(text='Smoothing: OFF',
                               state='normal',
                               font_name=config.kivy_font,
                               font_size=config.kivy_font_size,
                               pos_hint={"center_x": 0.5, "center_y": 0.5},
                               size_hint=(0.2, 1)
                               )
        self.m4.bind(on_press=self.callback_m4)
        # Derivative
        self.m5 = ToggleButton(text='Derivative: OFF',
                               state='normal',
                               font_name=config.kivy_font,
                               font_size=config.kivy_font_size,
                               pos_hint={"center_x": 0.5, "center_y": 0.5},
                               size_hint=(0.2, 1)
                               )
        self.m5.bind(on_press=self.callback_m5)

        self.create_cut_fit_plot = Button(text="Fit",
                                          bold=True,
                                          background_color=config.kivy_color_button,
                                          font_name=config.kivy_font,
                                          font_size=config.kivy_font_size_title,
                                          color=config.kivy_color_white,
                                          size_hint=(0.2, 1),
                                          pos_hint={"center_x": 0.5,
                                                    "center_y": 0.5}
                                          )
        self.create_cut_fit_plot.bind(on_press=self.callback_5_0)
        self.create_cut_fit_plot.bind(on_release=self.callback_5)

        # add widgets to the line
        self.box14.add_widget(m1)
        self.box14.add_widget(self.m2)
        self.box14.add_widget(self.m3)
        self.box14.add_widget(self.m4)
        self.box14.add_widget(self.m5)
        self.box14.add_widget(self.create_cut_fit_plot)

        '''CREATE PLOT'''
        self.create_cut_plot = Button(text="Slice 2D Histogram",
                                      bold=True,
                                      background_color=config.kivy_color_button,
                                      font_name=config.kivy_font,
                                      font_size=config.kivy_font_size_title,
                                      color=config.kivy_color_white,
                                      size_hint=(0.8, 1),
                                      pos_hint={"center_x": 0.5,
                                                "center_y": 0.5}
                                      )
        self.create_cut_plot.bind(on_press=self.callback_4_0)
        self.create_cut_plot.bind(on_release=self.callback_4)
        self.box15.add_widget(self.create_cut_plot)

        save_cut_dat = Button(text="Save ASCII",
                              bold=True,
                              background_color=config.kivy_color_button,
                              font_name=config.kivy_font,
                              font_size=config.kivy_font_size_title,
                              color=config.kivy_color_white,
                              size_hint=(0.2, 1),
                              pos_hint={"center_x": 0.5, "center_y": 0.5}
                              )
        save_cut_dat.bind(on_press=self.callback_save_cut_dat)
        self.box15.add_widget(save_cut_dat)

        self.create_cut_plot_mode = ToggleButton(
                                     text='Mode: PyQt',
                                     bold=True,
                                     background_color=config.kivy_color_button,
                                     font_name=config.kivy_font,
                                     font_size=config.kivy_font_size_title,
                                     color=config.kivy_color_white,
                                     size_hint=(0.2, 1),
                                     pos_hint={"center_x": 0.5,
                                               "center_y": 0.5},
                                     halign='center'
                                     )
        self.create_cut_plot_mode.bind(on_press=self.create_plot_mode_callback)
        self.box15.add_widget(self.create_cut_plot_mode)

        contact_info = Label(text='Author: Dr. Dmitrii Potorochin   Email: dmitrii.potorochin@desy.de',
                             font_size=config.kivy_font_size*0.75,
                             color=config.kivy_color_white,
                             font_name=config.kivy_font,
                             size_hint=(1, 1),
                             halign='left', opacity=0.3
                             )
        self.box16.add_widget(contact_info)

        if config.file_type == 'WESPE':
            self.e1.text = "Binning"
            self.e2.text = 'Energy step'
            self.e3.text = '0.05'
            self.e4.text = 'eV'
            self.e5.text = 'Time step'
            self.e6.text = '0.1'
            self.e7.text = 'ps'

            self.i1.text = "Delay/Energy ROI"
            self.i2.text = 'Energy: OFF'
            self.i4.text = 'eV'
            self.i5.text = 'Delay: OFF'
            self.i7.text = 'ps'

            self.k2.text = 'Time axis'

            self.i2.bind(on_press=self.callback_i2)
            self.i5.bind(on_press=self.callback_i5)
            self.k2.bind(on_press=self.callback_k2)
        if config.file_type == 'ALS':
            self.e1.text = "Load parameters"
            self.e2.text = 'DLD bins'
            self.e3.text = '1500'
            self.e4.text = 'a.u.'
            self.e5.text = 'Plot bunch'
            self.e6.text = '4'
            self.e7.text = 'a.u.'

            self.i1.text = "Delay/Energy ROI"
            self.i2.text = 'Energy: OFF'
            self.i4.text = 'eV'
            self.i5.text = 'Delay: OFF'
            self.i7.text = 'ps'

            self.k2.text = 'Time axis'

            self.i2.bind(on_press=self.callback_i2)
            self.i5.bind(on_press=self.callback_i5)
            self.k2.bind(on_press=self.callback_k2)
        elif config.file_type == 'MM':
            self.e1.text = "Binning"
            self.e2.text = 'Dim X step'
            self.e3.text = '0.05'
            self.e4.text = 'a.u.'
            self.e5.text = 'Dim Y step'
            self.e6.text = '0.1'
            self.e7.text = 'a.u.'

            self.i1.text = "Dim X/Dim Y ROI"
            self.i2.text = 'Dim X: OFF'
            self.i4.text = 'a.u.'
            self.i5.text = 'Dim Y: OFF'
            self.i7.text = 'a.u.'

            self.k2.text = 'Dim Y'

            self.i2.bind(on_press=self.callback_i2_a)
            self.i5.bind(on_press=self.callback_i5_a)
            self.k2.bind(on_press=self.callback_k2_a)
        else:
            self.e1.text = "Binning"
            self.e2.text = 'Energy step'
            self.e3.text = '0.05'
            self.e4.text = 'eV'
            self.e5.text = 'Time step'
            self.e6.text = '0.1'
            self.e7.text = 'ps'

            self.i1.text = "Dim X/Dim Y ROI"
            self.i2.text = 'Dim X: OFF'
            self.i4.text = 'a.u.'
            self.i5.text = 'Dim Y: OFF'
            self.i7.text = 'a.u.'

            self.k2.text = 'Dim Y'

            self.i2.bind(on_press=self.callback_i2_a)
            self.i5.bind(on_press=self.callback_i5_a)
            self.k2.bind(on_press=self.callback_k2_a)

        self.window.add_widget(self.top)
        self.window.add_widget(self.box1)
        self.window.add_widget(self.box2)
        self.window.add_widget(self.box3)
        self.window.add_widget(self.box4)
        if config.file_type in ['WESPE', 'MM', 'ALS']:
            self.window.add_widget(self.box5)
        if config.file_type in ['WESPE', 'MM']:
            self.window.add_widget(self.box6)
        self.window.add_widget(self.box7)
        self.window.add_widget(self.box8)
        self.window.add_widget(self.box9)
        self.window.add_widget(self.box10)
        self.window.add_widget(self.box11)
        self.window.add_widget(self.box12)
        self.window.add_widget(self.box13)
        self.window.add_widget(self.box14)
        self.window.add_widget(self.box15)
        self.window.add_widget(self.box16)
        print('***PESdata is running***')

        return self.window

    def suggestion(self, instance):
        try:
            items = self.run_numbers_input.text.split(',')
            item = items[-1]
            listdir = sorted(os.listdir(self.directory_input.text))
            listdir = [i.split('.')[0] for i in listdir]
            for i in ['fig_output', 'netCDF_maps', 'ASCII_output']:
                try:
                    listdir.remove(i)
                except:
                    pass
            output = []
            for i in listdir:
                if item.lower() in i.lower():
                    output.append(i)
            try:
                items[-1] = output[-1]
            except:
                items[-1] = item
            self.run_numbers_input.text = ','.join(items)
        except Exception as err:
            print('Unable to autocomplete!')
            print('Full error message:')
            print(err)

    def suggestion_dir(self, instance):
        try:
            items = self.directory_input.text.split(os.sep)
            item = items[-1]
            if len(items[:-1]) > 1:
                listdir = sorted(os.listdir(os.sep.join(items[:-1])))
            else:
                listdir = sorted(os.listdir(items[0] + os.sep))
            listdir = [i.split('.')[0] for i in listdir]
            for i in ['fig_output', 'netCDF_maps', 'ASCII_output']:
                try:
                    listdir.remove(i)
                except:
                    pass
            output = []
            for i in listdir:
                if item.lower() in i.lower():
                    output.append(i)
            try:
                items[-1] = output[0]
            except:
                items[-1] = item
            self.directory_input.text = os.sep.join(items)
        except Exception as err:
            print('Unable to autocomplete!')
            print('Full error message:')
            print(err)

    def callback_1_0(self, instance):
        self.upload_runs.text = 'Loading...'

    def callback_1(self, instance):
        try:
            with open('packages/config.json', 'r') as json_file:
                self.config = json.load(json_file)
            file_dir = self.directory_input.text
            file_type = scan_dir(file_dir)
            if file_type != self.config['file_type']:
                if file_type != 'UNKNOWN':
                    self.config['file_type'] = file_type
                    with open('packages/config.json', 'w') as json_file:
                        json.dump(self.config, json_file)
                    self.i2.unbind(on_press=self.callback_i2)
                    self.i2.unbind(on_press=self.callback_i2_a)
                    self.i5.unbind(on_press=self.callback_i5)
                    self.i5.unbind(on_press=self.callback_i5_a)
                    self.k2.unbind(on_press=self.callback_k2)
                    self.k2.unbind(on_press=self.callback_k2_a)
                    if file_type == 'CasaXPS':
                        from packages.PESdata_OOP import create_batch_CasaXPS as create_batch
                        self.i1.text = "Dim X/Dim Y ROI"
                        self.i2.text = 'Dim X: OFF'
                        self.i4.text = 'a.u.'
                        self.i5.text = 'Dim Y: OFF'
                        self.i7.text = 'a.u.'

                        self.k2.text = 'Dim Y'

                        self.window.clear_widgets()

                        self.box1.clear_widgets()
                        self.box1.add_widget(self.directory_label)
                        self.directory_input.size_hint = (1, 0.8)
                        self.box1.add_widget(self.directory_input)

                        self.box2.clear_widgets()
                        self.box2.add_widget(self.run_numbers_label)
                        self.run_numbers_input.size_hint = (1, 0.8)
                        self.box2.add_widget(self.run_numbers_input)

                        self.box7.clear_widgets()
                        self.box7.add_widget(self.create_map)

                        self.box9.clear_widgets()
                        self.box9.add_widget(self.i1)
                        self.box9.add_widget(self.i2)
                        self.box9.add_widget(self.i3)
                        self.box9.add_widget(self.i4)
                        self.box9.add_widget(self.i5)
                        self.box9.add_widget(self.i6)
                        self.box9.add_widget(self.i7)

                        self.i2.bind(on_press=self.callback_i2_a)
                        self.i5.bind(on_press=self.callback_i5_a)
                        self.k2.bind(on_press=self.callback_k2_a)

                        for i in [self.top, self.box1, self.box2, self.box3,
                                  self.box4, self.box7, self.box8, self.box9,
                                  self.box10, self.box11, self.box12,
                                  self.box13, self.box14, self.box15,
                                  self.box16]:
                            self.window.add_widget(i)
                    elif file_type == 'ibw':
                        from packages.PESdata_OOP import create_batch_ibw as create_batch
                        self.i1.text = "Dim X/Dim Y ROI"
                        self.i2.text = 'Dim X: OFF'
                        self.i4.text = 'a.u.'
                        self.i5.text = 'Dim Y: OFF'
                        self.i7.text = 'a.u.'

                        self.k2.text = 'Dim Y'

                        self.window.clear_widgets()

                        self.box1.clear_widgets()
                        self.box1.add_widget(self.directory_label)
                        self.directory_input.size_hint = (1, 0.8)
                        self.box1.add_widget(self.directory_input)

                        self.box2.clear_widgets()
                        self.box2.add_widget(self.run_numbers_label)
                        self.run_numbers_input.size_hint = (1, 0.8)
                        self.box2.add_widget(self.run_numbers_input)

                        self.box7.clear_widgets()
                        self.box7.add_widget(self.create_map)

                        self.box9.clear_widgets()
                        self.box9.add_widget(self.i1)
                        self.box9.add_widget(self.i2)
                        self.box9.add_widget(self.i3)
                        self.box9.add_widget(self.i4)
                        self.box9.add_widget(self.i5)
                        self.box9.add_widget(self.i6)
                        self.box9.add_widget(self.i7)

                        self.i2.bind(on_press=self.callback_i2_a)
                        self.i5.bind(on_press=self.callback_i5_a)
                        self.k2.bind(on_press=self.callback_k2_a)

                        for i in [self.top, self.box1, self.box2, self.box3,
                                  self.box4, self.box7, self.box8, self.box9,
                                  self.box10, self.box11, self.box12,
                                  self.box13,  self.box14, self.box15,
                                  self.box16]:
                            self.window.add_widget(i)
                    elif file_type == 'ALS':
                        from packages.PESdata_OOP import create_batch_ALS as create_batch
                        self.e1.text = "Load parameters"
                        self.e2.text = 'DLD bins'
                        self.e3.text = '1500'
                        self.e4.text = 'a.u.'
                        self.e5.text = 'Plot bunch'
                        self.e6.text = '4'
                        self.e7.text = 'a.u.'

                        self.i1.text = "Delay/Energy ROI"
                        self.i2.text = 'Energy: OFF'
                        self.i4.text = 'eV'
                        self.i5.text = 'Delay: OFF'
                        self.i7.text = 'ps'

                        self.k2.text = 'Time axis'

                        self.window.clear_widgets()

                        self.box1.clear_widgets()
                        self.box1.add_widget(self.directory_label)
                        self.directory_input.size_hint = (1, 0.8)
                        self.box1.add_widget(self.directory_input)

                        self.box2.clear_widgets()
                        self.box2.add_widget(self.run_numbers_label)
                        self.run_numbers_input.size_hint = (1, 0.8)
                        self.box2.add_widget(self.run_numbers_input)

                        self.box7.clear_widgets()
                        self.box7.add_widget(self.create_map)
                        self.box7.add_widget(self.map_mode)

                        self.box9.clear_widgets()
                        self.box9.add_widget(self.i1)
                        self.box9.add_widget(self.i2)
                        self.box9.add_widget(self.i3)
                        self.box9.add_widget(self.i4)
                        self.box9.add_widget(self.i5)
                        self.box9.add_widget(self.i6)
                        self.box9.add_widget(self.i7)

                        self.i2.bind(on_press=self.callback_i2)
                        self.i5.bind(on_press=self.callback_i5)
                        self.k2.bind(on_press=self.callback_k2)

                        for i in [self.top, self.box1, self.box2, self.box3,
                                  self.box4, self.box5, self.box7, self.box8,
                                  self.box9, self.box10, self.box11,
                                  self.box12, self.box13,
                                  self.box14, self.box15, self.box16]:
                            self.window.add_widget(i)
                    elif file_type == 'MM':
                        from packages.PESdata_OOP import create_batch_MM as create_batch
                        self.e1.text = "Binning"
                        self.e2.text = 'Dim X step'
                        self.e3.text = '0.05'
                        self.e4.text = 'a.u.'
                        self.e5.text = 'Dim Y step'
                        self.e6.text = '0.1'
                        self.e7.text = 'a.u.'

                        self.i1.text = "Dim X/Dim Y ROI"
                        self.i2.text = 'Dim X: OFF'
                        self.i4.text = 'a.u.'
                        self.i5.text = 'Dim Y: OFF'
                        self.i7.text = 'a.u.'

                        self.k2.text = 'Dim Y'

                        self.window.clear_widgets()

                        self.box1.clear_widgets()
                        self.box1.add_widget(self.directory_label)
                        self.directory_input.size_hint = (1, 0.8)
                        self.box1.add_widget(self.directory_input)

                        self.box2.clear_widgets()
                        self.box2.add_widget(self.run_numbers_label)
                        self.run_numbers_input.size_hint = (1, 0.8)
                        self.box2.add_widget(self.run_numbers_input)

                        self.box7.clear_widgets()
                        self.box7.add_widget(self.create_map)
                        self.box7.add_widget(self.hist_mode_label)
                        self.box7.add_widget(self.hist_mode)

                        self.box9.clear_widgets()
                        self.box9.add_widget(self.i1)
                        self.box9.add_widget(self.i2)
                        self.box9.add_widget(self.i3)
                        self.box9.add_widget(self.i4)
                        self.box9.add_widget(self.i5)
                        self.box9.add_widget(self.i6)
                        self.box9.add_widget(self.i7)

                        self.i2.bind(on_press=self.callback_i2_a)
                        self.i5.bind(on_press=self.callback_i5_a)
                        self.k2.bind(on_press=self.callback_k2_a)

                        for i in [self.top, self.box1, self.box2, self.box3,
                                  self.box4, self.box5, self.box6, self.box7,
                                  self.box8, self.box9, self.box10,
                                  self.box11, self.box12, self.box13,
                                  self.box14, self.box15, self.box16]:
                            self.window.add_widget(i)
                    else:
                        from packages.PESdata_OOP import create_batch_WESPE as create_batch
                        self.e1.text = "Binning"
                        self.e2.text = 'Energy step'
                        self.e3.text = '0.05'
                        self.e4.text = 'eV'
                        self.e5.text = 'Time step'
                        self.e6.text = '0.1'
                        self.e7.text = 'ps'

                        self.i1.text = "Delay/Energy ROI"
                        self.i2.text = 'Energy: OFF'
                        self.i4.text = 'eV'
                        self.i5.text = 'Delay: OFF'
                        self.i7.text = 'ps'

                        self.k2.text = 'Time axis'

                        self.window.clear_widgets()

                        self.box1.clear_widgets()
                        self.box1.add_widget(self.directory_label)
                        self.directory_input.size_hint = (0.8, 0.8)
                        self.box1.add_widget(self.directory_input)
                        self.box1.add_widget(self.DLD_label)

                        self.box2.clear_widgets()
                        self.box2.add_widget(self.run_numbers_label)
                        self.run_numbers_input.size_hint = (0.8, 0.8)
                        self.box2.add_widget(self.run_numbers_input)
                        self.box2.add_widget(self.DLD_toggle)

                        self.box7.clear_widgets()
                        self.box7.add_widget(self.create_map)
                        self.box7.add_widget(self.map_mode)

                        self.box9.clear_widgets()
                        self.box9.add_widget(self.i1)
                        self.box9.add_widget(self.i2)
                        self.box9.add_widget(self.i3)
                        self.box9.add_widget(self.i4)
                        self.box9.add_widget(self.i5)
                        self.box9.add_widget(self.i6)
                        self.box9.add_widget(self.i7)

                        self.i2.bind(on_press=self.callback_i2)
                        self.i5.bind(on_press=self.callback_i5)
                        self.k2.bind(on_press=self.callback_k2)

                        for i in [self.top, self.box1, self.box2, self.box3,
                                  self.box4, self.box5, self.box6, self.box7,
                                  self.box8, self.box9, self.box10,
                                  self.box11, self.box12, self.box13,
                                  self.box14, self.box15, self.box16]:
                            self.window.add_widget(i)
                    global create_batch

            run_numbers = self.run_numbers_input.text.split(',')
            run_numbers = [i.strip() for i in run_numbers]

            self.config['file_dir'] = self.directory_input.text
            self.config['run_numbers'] = self.run_numbers_input.text
            with open('packages/config.json', 'w') as json_file:
                json.dump(self.config, json_file)

            if self.DLD_toggle.state == 'down':
                self.DLD = 'DLD4Q'
            else:
                self.DLD = 'DLD1Q'
            batch = create_batch(file_dir, run_numbers, DLD=self.DLD)

            Popup_run_info = BoxLayout(orientation='vertical', spacing=1)
            Popup_output = batch.short_info + batch.full_info
            length = Popup_output.count('\n')/7

            Run_info = Label(text=Popup_output,
                             font_size=config.kivy_font_size_title,
                             color=config.kivy_color_white,
                             font_name=config.kivy_font,
                             size_hint_y=0.25*length
                             )

            Run_info_scroll = ScrollView()
            Run_info_scroll.add_widget(Run_info)
            Popup_run_info.add_widget(Run_info_scroll)

            close_button = Button(
                              text="Close",
                              bold=True,
                              background_color=config.kivy_color_button,
                              font_name=config.kivy_font,
                              font_size=config.kivy_font_size_title,
                              color=config.kivy_color_white,
                              size_hint=(1, 0.1),
                              pos_hint={"center_x": 0.5, "center_y": 0.5}
                              )
            Popup_run_info.add_widget(close_button)

            popupWindow = Popup(title="Run info",
                                content=Popup_run_info)
            close_button.bind(on_press=lambda x: popupWindow.dismiss())
            popupWindow.open()
            self.batch = batch
            self.upload_runs.text = "Upload runs"
        except Exception as err:
            print('Unable to open file(s)!')
            print('Full error message:')
            print(err)
            self.upload_runs.text = "Upload runs"

    def callback_2_0(self, instance):
        self.create_map.text = 'Loading...'

    def callback_2(self, instance):
        try:
            config = json.dumps(self.config)
            config = json.loads(config,
                                object_hook=lambda d: SimpleNamespace(**d))
        except:
            pass
        try:
            if self.f2.state == 'down':
                for i in self.batch.batch_list:
                    B_range = self.f3.text.split(',')
                    B_range = [float(i) for i in B_range]
                    i.Bunch_filter(B_range, B_type='MacroBunch')

            if self.f5.state == 'down':
                for i in self.batch.batch_list:
                    B_range = self.f6.text.split(',')
                    B_range = [float(i) for i in B_range]
                    i.Bunch_filter(B_range, B_type='MicroBunch')

            # Input for WESPE
            try:
                energy_step = float(self.e3.text)
                delay_step = float(self.e6.text)
            except:
                pass
            t0 = float(self.d3.text)

            # Input for ALS
            # Plot bunch
            try:
                bunch_sel = int(self.e6.text)
            except ValueError:
                bunch_sel = self.e6.text
            try:
                DLD_t_res = int(self.e3.text)
            except ValueError:
                DLD_t_res = self.e3.text

            if self.map_mode.state == 'down':
                ordinate = 'MB_ID'
                delay_step = 1
            else:
                ordinate = 'delay'

            if config.file_type == 'MM':
                ordinate = self.hist_mode.text

            for i in self.batch.batch_list:
                try:
                    i.create_map(energy_step, delay_step, ordinate=ordinate,
                                 save=config.save_nc)
                except:
                    try:
                        i.create_map(ordinate=ordinate, bunch_sel=bunch_sel,
                                     DLD_t_res=DLD_t_res, save=config.save_nc)
                    except:
                        i.create_map()
                if self.d5.state == 'down':
                    i.set_BE()

            self.batch.create_map()
            if self.d2.state == 'down':
                self.batch.time_zero(t0)
            if self.d2.state == 'down':
                self.batch.create_dif_map()

            if config.file_type == 'WESPE':
                self.batch.ROI([0, self.batch.en_threshold], 'Dim_x')
            if config.matplotlib == 'qt':
                get_ipython().run_line_magic('matplotlib', 'qt')
                self.dpi = config.dpi/config.dpi_scale
                self.fig_width = config.fig_width*1.5
                self.fig_height = config.fig_height*1.5
            else:
                self.dpi = config.dpi
                self.fig_width = config.fig_width
                self.fig_height = config.fig_height
            if len(self.batch.static_cut_list) > 0:
                try:
                    self.cut = map_cut(self.batch, self.batch.static_cut_list,
                                       [0.5], 'Dim_y')
                    if np.mean(self.cut.cuts) != 0:
                        self.cut.norm_01()
                    plot = plot_files([self.batch, self.cut], dpi=self.dpi,
                                      fig_width=self.fig_width,
                                      fig_height=self.fig_height)
                    plot.span_plot(self.cut)
                    plot.legend_plot()
                    self.batch.static_cut_list_s = [str(i) for i in self.batch.static_cut_list]
                    self.batch.static_cut_list_s = ','.join(self.batch.static_cut_list_s)
                    self.l3.text = self.batch.static_cut_list_s
                except KeyError:
                    plot_files(self.batch, dpi=self.dpi,
                               fig_width=self.fig_width,
                               fig_height=self.fig_height)
            else:
                self.batch.Map_2D_plot = self.batch.Map_2D
                plot_files(self.batch, dpi=self.dpi,
                           fig_width=self.fig_width,
                           fig_height=self.fig_height)
            if config.matplotlib != 'qt':
                plt.show()
            self.create_map.text = "Compute 2D Histogram"

            with open('packages/config.json', 'r') as json_file:
                self.config = json.load(json_file)
            self.config['t_zero'] = self.d3.text
            with open('packages/config.json', 'w') as json_file:
                json.dump(self.config, json_file)
        except Exception as err:
            print('Unable to open file(s)!')
            print('Full error message:')
            print(err)
            self.create_map.text = "Compute 2D Histogram"

    def callback_3_0(self, instance):
        self.create_plot.text = 'Loading...'

    def callback_3(self, instance):
        try:
            config = json.dumps(self.config)
            config = json.loads(config,
                                object_hook=lambda d: SimpleNamespace(**d))
        except:
            pass
        try:
            self.dpi = config.dpi
            self.fig_width = config.fig_width
            self.fig_height = config.fig_height
            if self.create_plot_mode.state == 'down':
                if config.matplotlib == 'qt':
                    get_ipython().run_line_magic('matplotlib', 'inline')
            else:
                if config.matplotlib == 'qt':
                    get_ipython().run_line_magic('matplotlib', 'qt')
                    self.dpi = config.dpi/config.dpi_scale
                    self.fig_width = config.fig_width*1.5
                    self.fig_height = config.fig_height*1.5

            self.batch.Map_2D_plot = self.batch.Map_2D

            if self.j2.state == 'down':
                self.batch.norm_total_e()

            if self.h2.state == 'down':
                try:
                    self.batch.set_T0()
                except KeyError:
                    pass
            else:
                self.batch.set_Tds()

            if self.h4.state == 'down':
                self.batch.set_KE()
            else:
                self.batch.set_BE()

            if self.h3.state == 'down':
                self.batch.create_dif_map()
                self.batch.set_dif_map()

            if self.j3.state == 'down':
                self.batch.norm_01()

            if self.j4.state == 'down':
                self.batch.norm_11()

            if self.i2.state == 'down':
                ROI_E = self.i3.text.split(',')
                try:
                    ROI_E = [float(i) for i in ROI_E]
                except ValueError:
                    ROI_E = self.batch.Map_2D.coords['Dim_x']
                    ROI_E = ROI_E.values
                    ROI_E = [np.min(ROI_E), np.max(ROI_E)]
                self.batch.ROI(ROI_E, 'Dim_x')

            if self.i5.state == 'down':
                ROI_D = self.i6.text.split(',')
                try:
                    ROI_D = [float(i) for i in ROI_D]
                except ValueError:
                    ROI_D = self.batch.Map_2D.coords['Delay']
                    ROI_D = ROI_D.values
                    ROI_D = [np.min(ROI_D), np.max(ROI_D)]
                    print(ROI_D)
                self.batch.ROI(ROI_D, 'Dim_y')

            plot_files(self.batch, dpi=self.dpi,
                       fig_width=self.fig_width,
                       fig_height=self.fig_height)

            if self.create_plot_mode.state == 'down':
                path = self.directory_input.text.strip() + os.sep
                path = path + 'fig_output' + os.sep
                if os.path.isdir(path) is False:
                    os.mkdir(path)
                ts = calendar.timegm(gmtime())
                date_time = datetime.fromtimestamp(ts)
                str_date_time = date_time.strftime("%d.%m.%Y_%H-%M-%S")
                plt.tight_layout()
                plt.savefig(f'{path}Fig_{str_date_time}.png', dpi=self.dpi,
                            bbox_inches="tight")
                if config.matplotlib != 'qt':
                    plt.show()
            else:
                if config.matplotlib != 'qt':
                    plt.show()

            self.create_plot.text = "Create 2D Histogram visualization"

            with open('packages/config.json', 'r') as json_file:
                self.config = json.load(json_file)
            self.config['en_ROI'] = self.i3.text
            self.config['t_ROI'] = self.i6.text
            with open('packages/config.json', 'w') as json_file:
                json.dump(self.config, json_file)
        except Exception as err:
            print('Probably no data loaded!')
            print('Full error message:')
            print(err)
            self.create_plot.text = "Create 2D Histogram visualization"

    def callback_4_0(self, instance):
        self.create_cut_plot.text = 'Loading...'

    def callback_4(self, instance):
        try:
            config = json.dumps(self.config)
            config = json.loads(config,
                                object_hook=lambda d: SimpleNamespace(**d))
        except:
            pass
        try:
            self.dpi = config.dpi
            self.fig_width = config.fig_width
            self.fig_height = config.fig_height
            if self.create_cut_plot_mode.state == 'down':
                if config.matplotlib == 'qt':
                    get_ipython().run_line_magic('matplotlib', 'inline')
            else:
                if config.matplotlib == 'qt':
                    get_ipython().run_line_magic('matplotlib', 'qt')
                    self.dpi = config.dpi/config.dpi_scale
                    self.fig_width = config.fig_width*1.5
                    self.fig_height = config.fig_height*1.5

            if self.k2.state == 'down':
                axis = 'Dim_y'
            else:
                axis = 'Dim_x'

            if self.k5.state == 'down':
                add_map = True
            else:
                add_map = False

            positions = self.l3.text.split(',')

            # correlation part 1
            cc_total_trigger = False
            if positions[0].lower() == 'cc':
                cc_trigger = True
                try:
                    positions[1]
                    try:
                        b_sel = [int(i) for i in positions[1:]]
                        if len(b_sel) == 1:
                            b_sel = b_sel[0]
                    except ValueError:
                        b_sel = positions[1]
                except IndexError:
                    b_sel = None
                positions = 'Main'
            elif positions[0].lower() == 'ccp':
                cc_trigger = True
                positions = 'Main'
                b_sel = 'prev'
            elif positions[0].lower() == 'cci':
                cc_trigger = True
                positions = 'Main'
                b_sel = 'cci'
            elif positions[0].lower() == 'cct':
                cc_trigger = True
                try:
                    positions[1]
                    try:
                        b_sel = [int(i) for i in positions[1:]]
                        if len(b_sel) == 1:
                            b_sel = b_sel[0]
                    except ValueError:
                        b_sel = positions[1]
                except IndexError:
                    b_sel = None
                positions = 'Main'
                cc_total_trigger = True
            else:
                cc_trigger = False

            try:
                positions = [float(i) for i in positions]
            except ValueError:
                positions = ','.join(positions)
                positions = self.batch.t0_cut(positions, axis=axis)

            deltas = self.l6.text.split(',')
            try:
                deltas = [float(i) for i in deltas]
            except ValueError:
                deltas = []

            self.cut = map_cut(self.batch, positions, deltas, axis)
            # correlation part 2
            if cc_trigger is True:
                if cc_total_trigger is True:
                    self.cut.correlate_total(b_sel=b_sel)
                else:
                    if b_sel == 'cci':
                        self.cut.correlate_i()
                    else:
                        self.cut.correlate_b(b_sel=b_sel)
            if self.m2.state == 'down':
                self.cut.norm_01()
            if self.m3.state == 'down':
                self.cut.norm_11()
            if self.m4.state == 'down':
                self.cut.savgol_smooth()
            if self.m5.state == 'down':
                self.cut.derivative()

            if self.k3.state == 'down':
                self.cut.dif_plot()
            if self.k4.state == 'down':
                self.cut.waterfall()

            if add_map is True and cc_trigger is False:
                plot = plot_files([self.batch, self.cut], dpi=self.dpi,
                                  fig_width=self.fig_width,
                                  fig_height=self.fig_height)
                plot.span_plot(self.cut)
            else:
                plot = plot_files(self.cut, dpi=self.dpi,
                                  fig_width=self.fig_width,
                                  fig_height=self.fig_height)

            if self.k6.state == 'down':
                plot.legend_plot()

            if self.create_cut_plot_mode.state == 'down':
                path = self.directory_input.text.strip() + os.sep
                path = path + 'fig_output' + os.sep
                if os.path.isdir(path) is False:
                    os.mkdir(path)
                ts = calendar.timegm(gmtime())
                date_time = datetime.fromtimestamp(ts)
                str_date_time = date_time.strftime("%d.%m.%Y_%H-%M-%S")
                plt.tight_layout()
                plt.savefig(f'{path}Fig_{str_date_time}.png', dpi=self.dpi,
                            bbox_inches="tight")
                if config.matplotlib != 'qt':
                    plt.show()
            else:
                if config.matplotlib != 'qt':
                    plt.show()

            self.create_cut_plot.text = "Slice 2D Histogram"

            with open('packages/config.json', 'r') as json_file:
                self.config = json.load(json_file)
            self.config['cut_pos'] = self.l3.text
            self.config['cut_width'] = self.l6.text
            with open('packages/config.json', 'w') as json_file:
                json.dump(self.config, json_file)
        except Exception as err:
            print('Probably no data loaded!')
            print('Full error message:')
            print(err)
            self.create_cut_plot.text = "Slice 2D Histogram"

    def callback_5_0(self, instance):
        self.create_cut_fit_plot.text = 'Loading...'

    def callback_5(self, instance):
        try:
            config = json.dumps(self.config)
            config = json.loads(config,
                                object_hook=lambda d: SimpleNamespace(**d))
        except:
            pass
        try:
            self.dpi = config.dpi
            self.fig_width = config.fig_width
            self.fig_height = config.fig_height
            if self.create_cut_plot_mode.state == 'down':
                if config.matplotlib == 'qt':
                    get_ipython().run_line_magic('matplotlib', 'inline')
            else:
                if config.matplotlib == 'qt':
                    get_ipython().run_line_magic('matplotlib', 'qt')
                    self.dpi = config.dpi/config.dpi_scale
                    self.fig_width = config.fig_width*1.5
                    self.fig_height = config.fig_height*1.5

            if self.k2.state == 'down':
                axis = 'Dim_y'
            else:
                axis = 'Dim_x'

            if self.k5.state == 'down':
                add_map = True
            else:
                add_map = False

            positions = self.l3.text.split(',')

            # correlation part 1
            cc_total_trigger = False
            if positions[0].lower() == 'cc':
                cc_trigger = True
                try:
                    positions[1]
                    try:
                        b_sel = [int(i) for i in positions[1:]]
                        if len(b_sel) == 1:
                            b_sel = b_sel[0]
                    except ValueError:
                        b_sel = positions[1]
                except IndexError:
                    b_sel = None
                positions = 'Main'
            elif positions[0].lower() == 'ccp':
                cc_trigger = True
                positions = 'Main'
                b_sel = 'prev'
            elif positions[0].lower() == 'cci':
                cc_trigger = True
                positions = 'Main'
                b_sel = 'cci'
            elif positions[0].lower() == 'cct':
                cc_trigger = True
                try:
                    positions[1]
                    try:
                        b_sel = [int(i) for i in positions[1:]]
                        if len(b_sel) == 1:
                            b_sel = b_sel[0]
                    except ValueError:
                        b_sel = positions[1]
                except IndexError:
                    b_sel = None
                positions = 'Main'
                cc_total_trigger = True
            else:
                cc_trigger = False

            try:
                positions = [float(i) for i in positions]
            except ValueError:
                positions = ','.join(positions)
                positions = self.batch.t0_cut(positions, axis=axis)

            deltas = self.l6.text.split(',')
            try:
                deltas = [float(i) for i in deltas]
            except ValueError:
                deltas = []

            self.cut = map_cut(self.batch, positions, deltas, axis)
            # correlation part 2
            if cc_trigger is True:
                if cc_total_trigger is True:
                    self.cut.correlate_total(b_sel=b_sel)
                else:
                    if b_sel == 'cci':
                        self.cut.correlate_i()
                    else:
                        self.cut.correlate_b(b_sel=b_sel)
            if self.m2.state == 'down':
                self.cut.norm_01()
            if self.m3.state == 'down':
                self.cut.norm_11()
            if self.m4.state == 'down':
                self.cut.savgol_smooth()
            if self.m5.state == 'down':
                self.cut.derivative()
            self.cut.make_fit()

            if add_map is True and cc_trigger is False:
                plot = plot_files([self.batch, self.cut], dpi=self.dpi,
                                  fig_width=self.fig_width,
                                  fig_height=self.fig_height)
                plot.legend_plot()
                plot.span_plot(self.cut)
            else:
                plot = plot_files(self.cut, dpi=self.dpi,
                                  fig_width=self.fig_width,
                                  fig_height=self.fig_height)
                plot.legend_plot()

            if self.create_cut_plot_mode.state == 'down':
                path = self.directory_input.text.strip() + os.sep
                path = path + 'fig_output' + os.sep
                if os.path.isdir(path) is False:
                    os.mkdir(path)
                ts = calendar.timegm(gmtime())
                date_time = datetime.fromtimestamp(ts)
                str_date_time = date_time.strftime("%d.%m.%Y_%H-%M-%S")
                plt.tight_layout()
                plt.savefig(f'{path}Fig_{str_date_time}.png', dpi=self.dpi,
                            bbox_inches="tight")
                if config.matplotlib != 'qt':
                    plt.show()
            else:
                if config.matplotlib != 'qt':
                    plt.show()

            self.create_cut_fit_plot.text = "Fit"
        except Exception as err:
            print('Probably no data loaded!')
            print('Full error message:')
            print(err)
            self.create_cut_fit_plot.text = "Fit"

    def callback_save_map_dat(self, instance):
        try:
            self.batch.save_map_dat()
        except Exception as err:
            print('Probably no data loaded!')
            print('Full error message:')
            print(err)

    def callback_save_cut_dat(self, instance):
        try:
            self.cut.save_cut_dat()
        except Exception as err:
            print('Probably no data loaded!')
            print('Full error message:')
            print(err)

    def settings_popup_callback(self, instance):
        with open('packages/config.json', 'r') as json_file:
            self.config = json.load(json_file)
        config = json.dumps(self.config)
        config = json.loads(config, object_hook=lambda d: SimpleNamespace(**d))

        Settings_window = BoxLayout(orientation='vertical',
                                    spacing=10, padding=5,
                                    size_hint=(0.5, 1))
        box0 = BoxLayout(orientation='horizontal', size_hint=(0.8, 1),
                         pos_hint={"center_x": 0.5, "center_y": 0.5})
        box1 = BoxLayout(orientation='horizontal', size_hint=(0.8, 1),
                         pos_hint={"center_x": 0.5, "center_y": 0.5})
        box2 = BoxLayout(orientation='horizontal', size_hint=(0.8, 1),
                         pos_hint={"center_x": 0.5, "center_y": 0.5})
        box3 = BoxLayout(orientation='horizontal', size_hint=(0.8, 1),
                         pos_hint={"center_x": 0.5, "center_y": 0.5})
        box4 = BoxLayout(orientation='horizontal', size_hint=(0.8, 1),
                         pos_hint={"center_x": 0.5, "center_y": 0.5})
        box5 = BoxLayout(orientation='horizontal', size_hint=(0.8, 1),
                         pos_hint={"center_x": 0.5, "center_y": 0.5})
        box6 = BoxLayout(orientation='horizontal', size_hint=(0.8, 1),
                         pos_hint={"center_x": 0.5, "center_y": 0.5})
        box7 = BoxLayout(orientation='horizontal', size_hint=(0.8, 1),
                         pos_hint={"center_x": 0.5, "center_y": 0.5})
        box8 = BoxLayout(orientation='horizontal', size_hint=(0.8, 1),
                         pos_hint={"center_x": 0.5, "center_y": 0.5})
        box9 = BoxLayout(orientation='horizontal', size_hint=(0.8, 1),
                         pos_hint={"center_x": 0.5, "center_y": 0.5})
        box10 = BoxLayout(orientation='horizontal', size_hint=(0.8, 1),
                          pos_hint={"center_x": 0.5, "center_y": 0.5})
        box11 = BoxLayout(orientation='horizontal', size_hint=(0.8, 1),
                          pos_hint={"center_x": 0.5, "center_y": 0.5})
        box12 = BoxLayout(orientation='horizontal', size_hint=(0.8, 1),
                          pos_hint={"center_x": 0.5, "center_y": 0.5})
        box13 = BoxLayout(orientation='horizontal', size_hint=(0.8, 1),
                          pos_hint={"center_x": 0.5, "center_y": 0.5})

        bottom = BoxLayout(orientation='horizontal', size_hint=(1, 1.5))

        kivy_app_section = Label(text='App parameters',
                                 font_size=config.kivy_font_size_title*1.3,
                                 color=config.kivy_color_title,
                                 font_name=config.kivy_font,
                                 )

        graph_section = Label(text='General graph parameters',
                              font_size=config.kivy_font_size_title*1.3,
                              color=config.kivy_color_title,
                              font_name=config.kivy_font,
                              )

        map_section = Label(text='2D Histogram parameters',
                            font_size=config.kivy_font_size_title*1.3,
                            color=config.kivy_color_title,
                            font_name=config.kivy_font,
                            )

        slice_section = Label(text='Slice plot parameters',
                              font_size=config.kivy_font_size_title*1.3,
                              color=config.kivy_color_title,
                              font_name=config.kivy_font,
                              )

        slice_grid_section = Label(text='Slice plot grid parameters',
                                   font_size=config.kivy_font_size_title*1.3,
                                   color=config.kivy_color_title,
                                   font_name=config.kivy_font,
                                   )

        kivy_font = Label(
                    text="App font",
                    font_size=config.kivy_font_size_title,
                    color=config.kivy_color_title,
                    bold=True,
                    font_name=config.kivy_font,
                    size_hint=(0.5, 1),
                    halign='center'
                    )

        self.kivy_font_value = TextInput(text=str(config.kivy_font),
                                         multiline=False,
                                         pos_hint={"center_x": 0.5,
                                                   "center_y": 0.5},
                                         size_hint=(0.5, 1),
                                         font_name=config.kivy_font,
                                         font_size=config.kivy_font_size
                                         )

        kivy_font_size = Label(
                    text="App font size",
                    font_size=config.kivy_font_size_title,
                    color=config.kivy_color_title,
                    bold=True,
                    font_name=config.kivy_font,
                    size_hint=(0.5, 1),
                    halign='center'
                    )

        self.kivy_font_size_value = TextInput(text=str(config.kivy_font_size),
                                              multiline=False,
                                              pos_hint={"center_x": 0.5,
                                                        "center_y": 0.5},
                                              size_hint=(0.5, 1),
                                              font_name=config.kivy_font,
                                              font_size=config.kivy_font_size
                                              )

        kivy_font_size_title = Label(
                    text="App font size (title)",
                    font_size=config.kivy_font_size_title,
                    color=config.kivy_color_title,
                    bold=True,
                    font_name=config.kivy_font,
                    size_hint=(0.5, 1),
                    halign='center'
                    )

        self.kivy_font_size_title_value = TextInput(
                                         text=str(config.kivy_font_size_title),
                                         multiline=False,
                                         pos_hint={"center_x": 0.5,
                                                   "center_y": 0.5},
                                         size_hint=(0.5, 1),
                                         font_name=config.kivy_font,
                                         font_size=config.kivy_font_size
                                         )

        font_family = Label(
                        text="Graph font",
                        font_size=config.kivy_font_size_title,
                        color=config.kivy_color_title,
                        bold=True,
                        font_name=config.kivy_font,
                        size_hint=(0.5, 1),
                        halign='center'
                        )

        self.font_family_value = TextInput(text=str(config.font_family),
                                           multiline=False,
                                           pos_hint={"center_x": 0.5,
                                                     "center_y": 0.5},
                                           size_hint=(0.5, 1),
                                           font_name=config.kivy_font,
                                           font_size=config.kivy_font_size
                                           )

        font_size = Label(
                          text="Font size",
                          font_size=config.kivy_font_size_title,
                          color=config.kivy_color_title,
                          bold=True,
                          font_name=config.kivy_font,
                          size_hint=(0.5, 1),
                          halign='center'
                          )

        self.font_size_value = TextInput(text=str(config.font_size),
                                         multiline=False,
                                         pos_hint={"center_x": 0.5,
                                                   "center_y": 0.5},
                                         size_hint=(0.5, 1),
                                         font_name=config.kivy_font,
                                         font_size=config.kivy_font_size
                                         )

        font_size_axis = Label(
                    text="Font size axes",
                    font_size=config.kivy_font_size_title,
                    color=config.kivy_color_title,
                    bold=True,
                    font_name=config.kivy_font,
                    size_hint=(0.5, 1),
                    halign='center'
                    )

        self.font_size_axis_value = TextInput(text=str(config.font_size_axis),
                                              multiline=False,
                                              pos_hint={"center_x": 0.5,
                                                        "center_y": 0.5},
                                              size_hint=(0.5, 1),
                                              font_name=config.kivy_font,
                                              font_size=config.kivy_font_size
                                              )
        dpi = Label(
                    text="DPI",
                    font_size=config.kivy_font_size_title,
                    color=config.kivy_color_title,
                    bold=True,
                    font_name=config.kivy_font,
                    size_hint=(0.5, 1),
                    halign='center'
                    )

        self.dpi_value = TextInput(text=str(config.dpi),
                                   multiline=False,
                                   pos_hint={"center_x": 0.5,
                                             "center_y": 0.5},
                                   size_hint=(0.5, 1),
                                   font_name=config.kivy_font,
                                   font_size=config.kivy_font_size
                                   )

        fig_width = Label(
                    text="Figure width",
                    font_size=config.kivy_font_size_title,
                    color=config.kivy_color_title,
                    bold=True,
                    font_name=config.kivy_font,
                    size_hint=(0.5, 1),
                    halign='center'
                    )

        self.fig_width_value = TextInput(text=str(config.fig_width),
                                         multiline=False,
                                         pos_hint={"center_x": 0.5,
                                                   "center_y": 0.5},
                                         size_hint=(0.5, 1),
                                         font_name=config.kivy_font,
                                         font_size=config.kivy_font_size
                                         )

        fig_height = Label(
                    text="Figure height",
                    font_size=config.kivy_font_size_title,
                    color=config.kivy_color_title,
                    bold=True,
                    font_name=config.kivy_font,
                    size_hint=(0.5, 1),
                    halign='center'
                    )

        self.fig_height_value = TextInput(text=str(config.fig_height),
                                          multiline=False,
                                          pos_hint={"center_x": 0.5,
                                                    "center_y": 0.5},
                                          size_hint=(0.5, 1),
                                          font_name=config.kivy_font,
                                          font_size=config.kivy_font_size
                                          )

        axes_linewidth = Label(
                    text="Axes linewidth",
                    font_size=config.kivy_font_size_title,
                    color=config.kivy_color_title,
                    bold=True,
                    font_name=config.kivy_font,
                    size_hint=(0.5, 1),
                    halign='center'
                    )

        self.axes_linewidth_value = TextInput(text=str(config.axes_linewidth),
                                              multiline=False,
                                              pos_hint={"center_x": 0.5,
                                                        "center_y": 0.5},
                                              size_hint=(0.5, 1),
                                              font_name=config.kivy_font,
                                              font_size=config.kivy_font_size
                                              )

        cmap = Label(
                    text="Color map type",
                    font_size=config.kivy_font_size_title,
                    color=config.kivy_color_title,
                    bold=True,
                    font_name=config.kivy_font,
                    size_hint=(0.5, 1),
                    halign='center'
                    )

        self.cmap_value = TextInput(text=str(config.cmap),
                                    multiline=False,
                                    pos_hint={"center_x": 0.5,
                                              "center_y": 0.5},
                                    size_hint=(0.5, 1),
                                    font_name=config.kivy_font,
                                    font_size=config.kivy_font_size
                                    )

        map_scale = Label(
                        text="Color map scale",
                        font_size=config.kivy_font_size_title,
                        color=config.kivy_color_title,
                        bold=True,
                        font_name=config.kivy_font,
                        size_hint=(0.5, 1),
                        halign='center'
                        )

        self.map_scale_value = TextInput(text=str(config.TwoSlopeNorm),
                                         multiline=False,
                                         pos_hint={"center_x": 0.5,
                                                   "center_y": 0.5},
                                         size_hint=(0.5, 1),
                                         font_name=config.kivy_font,
                                         font_size=config.kivy_font_size
                                         )

        line_type_d = Label(
                        text="Line and marker type",
                        font_size=config.kivy_font_size_title,
                        color=config.kivy_color_title,
                        bold=True,
                        font_name=config.kivy_font,
                        size_hint=(0.5, 1),
                        halign='center'
                        )

        self.line_type_d_value = TextInput(text=str(config.line_type_d),
                                           multiline=False,
                                           pos_hint={"center_x": 0.5,
                                                     "center_y": 0.5},
                                           size_hint=(0.5, 1),
                                           font_name=config.kivy_font,
                                           font_size=config.kivy_font_size
                                           )

        save_nc = Label(
                        text="Save .nc files",
                        font_size=config.kivy_font_size_title,
                        color=config.kivy_color_title,
                        bold=True,
                        font_name=config.kivy_font,
                        size_hint=(0.5, 1),
                        halign='center'
                        )

        self.save_nc_value = TextInput(text=str(config.save_nc),
                                       multiline=False,
                                       pos_hint={"center_x": 0.5,
                                                 "center_y": 0.5},
                                       size_hint=(0.5, 1),
                                       font_name=config.kivy_font,
                                       font_size=config.kivy_font_size
                                       )

        line_width_d = Label(
                        text="Line width",
                        font_size=config.kivy_font_size_title,
                        color=config.kivy_color_title,
                        bold=True,
                        font_name=config.kivy_font,
                        size_hint=(0.5, 1),
                        halign='center'
                        )

        self.line_width_d_value = TextInput(text=str(config.line_width_d),
                                            multiline=False,
                                            pos_hint={"center_x": 0.5,
                                                      "center_y": 0.5},
                                            size_hint=(0.5, 1),
                                            font_name=config.kivy_font,
                                            font_size=config.kivy_font_size
                                            )

        marker_size_d = Label(
                            text="Marker size",
                            font_size=config.kivy_font_size_title,
                            color=config.kivy_color_title,
                            bold=True,
                            font_name=config.kivy_font,
                            size_hint=(0.5, 1),
                            halign='center'
                            )

        self.marker_size_d_value = TextInput(text=str(config.marker_size_d),
                                             multiline=False,
                                             pos_hint={"center_x": 0.5,
                                                       "center_y": 0.5},
                                             size_hint=(0.5, 1),
                                             font_name=config.kivy_font,
                                             font_size=config.kivy_font_size
                                             )

        line_op_d = Label(
                          text="Line opacity",
                          font_size=config.kivy_font_size_title,
                          color=config.kivy_color_title,
                          bold=True,
                          font_name=config.kivy_font,
                          size_hint=(0.5, 1),
                          halign='center'
                          )

        self.line_op_d_value = TextInput(text=str(config.line_op_d),
                                         multiline=False,
                                         pos_hint={"center_x": 0.5,
                                                   "center_y": 0.5},
                                         size_hint=(0.5, 1),
                                         font_name=config.kivy_font,
                                         font_size=config.kivy_font_size
                                         )

        line_type_grid_d = Label(
                              text="Grid line type",
                              font_size=config.kivy_font_size_title,
                              color=config.kivy_color_title,
                              bold=True,
                              font_name=config.kivy_font,
                              size_hint=(0.5, 1),
                              halign='center'
                              )

        self.line_type_grid_d_value = TextInput(
                                            text=str(config.line_type_grid_d),
                                            multiline=False,
                                            pos_hint={"center_x": 0.5,
                                                      "center_y": 0.5},
                                            size_hint=(0.5, 1),
                                            font_name=config.kivy_font,
                                            font_size=config.kivy_font_size
                                            )

        line_width_grid_d = Label(
                              text="Grid line width",
                              font_size=config.kivy_font_size_title,
                              color=config.kivy_color_title,
                              bold=True,
                              font_name=config.kivy_font,
                              size_hint=(0.5, 1),
                              halign='center'
                              )

        self.line_width_grid_d_value = TextInput(
                                            text=str(config.line_width_grid_d),
                                            multiline=False,
                                            pos_hint={"center_x": 0.5,
                                                      "center_y": 0.5},
                                            size_hint=(0.5, 1),
                                            font_name=config.kivy_font,
                                            font_size=config.kivy_font_size
                                            )

        line_op_grid_d = Label(
                          text="Grid line opacity",
                          font_size=config.kivy_font_size_title,
                          color=config.kivy_color_title,
                          bold=True,
                          font_name=config.kivy_font,
                          size_hint=(0.5, 1),
                          halign='center'
                          )

        self.line_op_grid_d_value = TextInput(text=str(config.line_op_grid_d),
                                              multiline=False,
                                              pos_hint={"center_x": 0.5,
                                                        "center_y": 0.5},
                                              size_hint=(0.5, 1),
                                              font_name=config.kivy_font,
                                              font_size=config.kivy_font_size
                                              )

        blur = Label(
                    text="Gaussian blur",
                    font_size=config.kivy_font_size_title,
                    color=config.kivy_color_title,
                    bold=True,
                    font_name=config.kivy_font,
                    size_hint=(0.5, 1),
                    halign='center'
                    )

        self.blur_value = TextInput(
                                    text=str(config.interpolation),
                                    multiline=False,
                                    pos_hint={"center_x": 0.5,
                                              "center_y": 0.5},
                                    size_hint=(0.5, 1),
                                    font_name=config.kivy_font,
                                    font_size=config.kivy_font_size
                                    )

        line_width_t0_line = Label(
                                  text="T0 line width",
                                  font_size=config.kivy_font_size_title,
                                  color=config.kivy_color_title,
                                  bold=True,
                                  font_name=config.kivy_font,
                                  size_hint=(0.5, 1),
                                  halign='center'
                                  )

        self.line_width_t0_line_value = TextInput(
                                        text=str(config.line_width_t0_line),
                                        multiline=False,
                                        pos_hint={"center_x": 0.5,
                                                  "center_y": 0.5},
                                        size_hint=(0.5, 1),
                                        font_name=config.kivy_font,
                                        font_size=config.kivy_font_size
                                        )

        line_op_t0_line = Label(
                                text="T0 line opacity",
                                font_size=config.kivy_font_size_title,
                                color=config.kivy_color_title,
                                bold=True,
                                font_name=config.kivy_font,
                                size_hint=(0.5, 1),
                                halign='center'
                                )

        self.line_op_t0_line_value = TextInput(
                                            text=str(config.line_op_t0_line),
                                            multiline=False,
                                            pos_hint={"center_x": 0.5,
                                                      "center_y": 0.5},
                                            size_hint=(0.5, 1),
                                            font_name=config.kivy_font,
                                            font_size=config.kivy_font_size
                                            )

        box0.add_widget(graph_section)
        box1.add_widget(font_family)
        box1.add_widget(self.font_family_value)
        box2.add_widget(font_size)
        box2.add_widget(self.font_size_value)
        box3.add_widget(font_size_axis)
        box3.add_widget(self.font_size_axis_value)
        box4.add_widget(dpi)
        box4.add_widget(self.dpi_value)
        box5.add_widget(fig_width)
        box5.add_widget(self.fig_width_value)
        box6.add_widget(fig_height)
        box6.add_widget(self.fig_height_value)
        box7.add_widget(axes_linewidth)
        box7.add_widget(self.axes_linewidth_value)

        box8.add_widget(map_section)
        box9.add_widget(cmap)
        box9.add_widget(self.cmap_value)
        box10.add_widget(map_scale)
        box10.add_widget(self.map_scale_value)

        box11.add_widget(blur)
        box11.add_widget(self.blur_value)
        box12.add_widget(line_width_t0_line)
        box12.add_widget(self.line_width_t0_line_value)
        box13.add_widget(line_op_t0_line)
        box13.add_widget(self.line_op_t0_line_value)

        box0.add_widget(kivy_app_section)
        box1.add_widget(kivy_font)
        box1.add_widget(self.kivy_font_value)
        box2.add_widget(kivy_font_size)
        box2.add_widget(self.kivy_font_size_value)
        box3.add_widget(kivy_font_size_title)
        box3.add_widget(self.kivy_font_size_title_value)
        box4.add_widget(save_nc)
        box4.add_widget(self.save_nc_value)

        box5.add_widget(slice_section)
        box6.add_widget(line_type_d)
        box6.add_widget(self.line_type_d_value)
        box7.add_widget(line_width_d)
        box7.add_widget(self.line_width_d_value)
        box8.add_widget(marker_size_d)
        box8.add_widget(self.marker_size_d_value)
        box9.add_widget(line_op_d)
        box9.add_widget(self.line_op_d_value)

        box10.add_widget(slice_grid_section)
        box11.add_widget(line_type_grid_d)
        box11.add_widget(self.line_type_grid_d_value)
        box12.add_widget(line_width_grid_d)
        box12.add_widget(self.line_width_grid_d_value)
        box13.add_widget(line_op_grid_d)
        box13.add_widget(self.line_op_grid_d_value)

        close_button = Button(
                          text="Close",
                          bold=True,
                          background_color=config.kivy_color_button,
                          font_name=config.kivy_font,
                          font_size=config.kivy_font_size_title,
                          color=config.kivy_color_white,
                          size_hint=(0.5, 1),
                          pos_hint={"center_x": 0.5, "center_y": 0.5}
                          )

        save_settings = Button(
                          text="Save settings",
                          bold=True,
                          background_color=config.kivy_color_button,
                          font_name=config.kivy_font,
                          font_size=config.kivy_font_size_title,
                          color=config.kivy_color_white,
                          size_hint=(0.3, 1),
                          pos_hint={"center_x": 0.5, "center_y": 0.5}
                          )

        load_settings = Button(
                              text="Load default settings",
                              bold=True,
                              background_color=config.kivy_color_button,
                              font_name=config.kivy_font,
                              font_size=config.kivy_font_size_title,
                              color=config.kivy_color_white,
                              size_hint=(0.2, 1),
                              pos_hint={"center_x": 0.5, "center_y": 0.5}
                              )

        load_settings.bind(on_press=self.callback_load_settings)
        save_settings.bind(on_press=self.callback_save_settings)

        bottom.add_widget(close_button)
        bottom.add_widget(save_settings)
        bottom.add_widget(load_settings)

        # Settings_window.add_widget(Settings_title)

        Settings_window.add_widget(box0)
        Settings_window.add_widget(box1)
        Settings_window.add_widget(box2)
        Settings_window.add_widget(box3)
        Settings_window.add_widget(box4)
        Settings_window.add_widget(box5)
        Settings_window.add_widget(box6)
        Settings_window.add_widget(box7)
        Settings_window.add_widget(box8)
        Settings_window.add_widget(box9)
        Settings_window.add_widget(box10)
        Settings_window.add_widget(box11)
        Settings_window.add_widget(box12)
        Settings_window.add_widget(box13)

        Settings_window.add_widget(bottom)

        popupWindow = Popup(title="Settings",
                            content=Settings_window)
        close_button.bind(on_press=lambda x: popupWindow.dismiss())
        popupWindow.open()

    def callback_save_settings(self, instance):
        self.config['kivy_font'] = self.kivy_font_value.text
        self.config['kivy_font_size'] = float(self.kivy_font_size_value.text)
        self.config['kivy_font_size_title'] = float(self.kivy_font_size_title_value.text)
        self.config['font_family'] = self.font_family_value.text
        self.config['font_size'] = float(self.font_size_value.text)
        self.config['font_size_axis'] = float(self.font_size_axis_value.text)
        self.config['dpi'] = float(self.dpi_value.text)
        self.config['fig_width'] = float(self.fig_width_value.text)
        self.config['fig_height'] = float(self.fig_height_value.text)
        self.config['axes_linewidth'] = float(self.axes_linewidth_value.text)
        self.config['cmap'] = self.cmap_value.text
        self.config['TwoSlopeNorm'] = float(self.map_scale_value.text)
        self.config['line_type_d'] = self.line_type_d_value.text
        self.config['save_nc'] = self.save_nc_value.text
        self.config['line_width_d'] = float(self.line_width_d_value.text)
        self.config['marker_size_d'] = float(self.marker_size_d_value.text)
        self.config['line_op_d'] = float(self.line_op_d_value.text)
        self.config['line_type_grid_d'] = self.line_type_grid_d_value.text
        self.config['line_width_grid_d'] = float(self.line_width_grid_d_value.text)
        self.config['line_op_grid_d'] = float(self.line_op_grid_d_value.text)
        self.config['interpolation'] = self.blur_value.text
        self.config['line_width_t0_line'] = float(self.line_width_t0_line_value.text)
        self.config['line_op_t0_line'] = float(self.line_op_t0_line_value.text)

        with open('packages/config.json', 'w') as json_file:
            json.dump(self.config, json_file)

        global config
        config = json.dumps(self.config)
        config = json.loads(config, object_hook=lambda d: SimpleNamespace(**d))

        if config.file_type == 'CasaXPS':
            from packages.PESdata_OOP import create_batch_CasaXPS as create_batch
        elif config.file_type == 'ibw':
            from packages.PESdata_OOP import create_batch_ibw as create_batch
        elif config.file_type == 'ALS':
            from packages.PESdata_OOP import create_batch_ALS as create_batch
        elif config.file_type == 'MM':
            from packages.PESdata_OOP import create_batch_MM as create_batch
        else:
            from packages.PESdata_OOP import create_batch_WESPE as create_batch
        global create_batch

        self.i2.unbind(on_press=self.callback_i2)
        self.i2.unbind(on_press=self.callback_i2_a)
        self.i5.unbind(on_press=self.callback_i5)
        self.i5.unbind(on_press=self.callback_i5_a)
        self.k2.unbind(on_press=self.callback_k2)
        self.k2.unbind(on_press=self.callback_k2_a)

        if config.file_type == 'WESPE':
            self.e1.text = "Binning"
            self.e2.text = 'Energy step'
            self.e4.text = 'eV'
            self.e5.text = 'Time step'
            self.e7.text = 'ps'

            self.i1.text = "Delay/Energy ROI"
            self.i2.text = 'Energy: OFF'
            self.i4.text = 'eV'
            self.i5.text = 'Delay: OFF'
            self.i7.text = 'ps'

            self.k2.text = 'Time axis'

            self.i2.bind(on_press=self.callback_i2)
            self.i5.bind(on_press=self.callback_i5)
            self.k2.bind(on_press=self.callback_k2)
        if config.file_type == 'ALS':
            self.e1.text = "Load parameters"
            self.e2.text = 'DLD bins'
            self.e4.text = 'a.u.'
            self.e5.text = 'Plot bunch'
            self.e7.text = 'a.u.'

            self.i1.text = "Delay/Energy ROI"
            self.i2.text = 'Energy: OFF'
            self.i4.text = 'eV'
            self.i5.text = 'Delay: OFF'
            self.i7.text = 'ps'

            self.k2.text = 'Time axis'

            self.i2.bind(on_press=self.callback_i2)
            self.i5.bind(on_press=self.callback_i5)
            self.k2.bind(on_press=self.callback_k2)
        elif config.file_type == 'MM':
            self.e1.text = "Binning"
            self.e2.text = 'Dim X step'
            self.e4.text = 'a.u.'
            self.e5.text = 'Dim Y step'
            self.e7.text = 'a.u.'

            self.i1.text = "Dim X/Dim Y ROI"
            self.i2.text = 'Dim X: OFF'
            self.i4.text = 'a.u.'
            self.i5.text = 'Dim Y: OFF'
            self.i7.text = 'a.u.'

            self.k2.text = 'Dim Y'

            self.i2.bind(on_press=self.callback_i2_a)
            self.i5.bind(on_press=self.callback_i5_a)
            self.k2.bind(on_press=self.callback_k2_a)
        else:
            self.e1.text = "Binning"
            self.e2.text = 'Energy step'
            self.e4.text = 'eV'
            self.e5.text = 'Time step'
            self.e7.text = 'ps'

            self.i1.text = "Dim X/Dim Y ROI"
            self.i2.text = 'Dim X: OFF'
            self.i4.text = 'a.u.'
            self.i5.text = 'Dim Y: OFF'
            self.i7.text = 'a.u.'

            self.k2.text = 'Dim Y'

            self.i2.bind(on_press=self.callback_i2_a)
            self.i5.bind(on_press=self.callback_i5_a)
            self.k2.bind(on_press=self.callback_k2_a)

        self.window.clear_widgets()
        if config.file_type == 'WESPE':
            self.box1.clear_widgets()
            self.box1.add_widget(self.directory_label)
            self.directory_input.size_hint = (0.8, 0.8)
            self.box1.add_widget(self.directory_input)
            self.box1.add_widget(self.DLD_label)

            self.box2.clear_widgets()
            self.box2.add_widget(self.run_numbers_label)
            self.run_numbers_input.size_hint = (0.8, 0.8)
            self.box2.add_widget(self.run_numbers_input)
            self.box2.add_widget(self.DLD_toggle)

            self.box7.clear_widgets()
            self.box7.add_widget(self.create_map)
            self.box7.add_widget(self.map_mode)
        elif config.file_type == 'ALS':
            self.box1.clear_widgets()
            self.box1.add_widget(self.directory_label)
            self.directory_input.size_hint = (1, 0.8)
            self.box1.add_widget(self.directory_input)

            self.box2.clear_widgets()
            self.box2.add_widget(self.run_numbers_label)
            self.run_numbers_input.size_hint = (1, 0.8)
            self.box2.add_widget(self.run_numbers_input)

            self.box7.clear_widgets()
            self.box7.add_widget(self.create_map)
            self.box7.add_widget(self.map_mode)
        elif config.file_type == 'MM':
            self.box1.clear_widgets()
            self.box1.add_widget(self.directory_label)
            self.directory_input.size_hint = (1, 0.8)
            self.box1.add_widget(self.directory_input)

            self.box2.clear_widgets()
            self.box2.add_widget(self.run_numbers_label)
            self.run_numbers_input.size_hint = (1, 0.8)
            self.box2.add_widget(self.run_numbers_input)

            self.box7.clear_widgets()
            self.box7.add_widget(self.create_map)
            self.box7.add_widget(self.hist_mode_label)
            self.box7.add_widget(self.hist_mode)
        else:
            self.box1.clear_widgets()
            self.box1.add_widget(self.directory_label)
            self.directory_input.size_hint = (1, 0.8)
            self.box1.add_widget(self.directory_input)

            self.box2.clear_widgets()
            self.box2.add_widget(self.run_numbers_label)
            self.run_numbers_input.size_hint = (1, 0.8)
            self.box2.add_widget(self.run_numbers_input)

            self.box7.clear_widgets()
            self.box7.add_widget(self.create_map)

        self.directory_input.text = config.file_dir
        self.run_numbers_input.text = config.run_numbers
        self.d3.text = config.t_zero
        self.i3.text = config.en_ROI
        self.i6.text = config.t_ROI
        self.l3.text = config.cut_pos
        self.l6.text = config.cut_width
        for i in [self.top, self.box1, self.box2, self.box3, self.box4,
                  self.box5, self.box6, self.box7, self.box8, self.box9,
                  self.box10, self.box11, self.box12, self.box13,
                  self.box14, self.box15, self.box16]:
            if config.file_type not in ['WESPE', 'MM', 'ALS'] and i == self.box5:
                continue
            if config.file_type not in ['WESPE', 'MM'] and i == self.box6:
                continue
            self.window.add_widget(i)

    def callback_load_settings(self, instance):
        with open('packages/default_config.json', 'r') as json_file:
            self.config = json.load(json_file)
        config = json.dumps(self.config)
        config = json.loads(config, object_hook=lambda d: SimpleNamespace(**d))

        self.kivy_font_value.text = str(config.kivy_font)
        self.kivy_font_size_value.text = str(config.kivy_font_size)
        self.kivy_font_size_title_value.text = str(config.kivy_font_size_title)
        self.font_family_value.text = str(config.font_family)
        self.font_size_value.text = str(config.font_size)
        self.font_size_axis_value.text = str(config.font_size_axis)
        self.dpi_value.text = str(config.dpi)
        self.fig_width_value.text = str(config.fig_width)
        self.fig_height_value.text = str(config.fig_height)
        self.axes_linewidth_value.text = str(config.axes_linewidth)
        self.cmap_value.text = str(config.cmap)
        self.map_scale_value.text = str(config.TwoSlopeNorm)
        self.line_type_d_value.text = str(config.line_type_d)
        self.save_nc_value.text = str(config.save_nc)
        self.line_width_d_value.text = str(config.line_width_d)
        self.marker_size_d_value.text = str(config.marker_size_d)
        self.line_op_d_value.text = str(config.line_op_d)
        self.line_type_grid_d_value.text = str(config.line_type_grid_d)
        self.line_width_grid_d_value.text = str(config.line_width_grid_d)
        self.line_op_grid_d_value.text = str(config.line_op_grid_d)
        self.blur_value.text = str(config.interpolation)
        self.line_width_t0_line_value.text = str(config.line_width_t0_line)
        self.line_op_t0_line_value.text = str(config.line_op_t0_line)

    def callback_d2(self, instance):
        if self.d2.state == 'down':
            self.d2.text = 'T0: ON'
        else:
            self.d2.text = 'T0: OFF'

    def callback_f2(self, instance):
        if self.f2.state == 'down':
            self.f2.text = 'MacroB: ON'
        else:
            self.f2.text = 'MacroB: OFF'

    def callback_f5(self, instance):
        if self.f5.state == 'down':
            self.f5.text = 'MicroB: ON'
        else:
            self.f5.text = 'MicroB: OFF'

    def callback_h2(self, instance):
        if self.h2.state == 'down':
            self.h2.text = 'T0: ON'
        else:
            self.h2.text = 'T0: OFF'

    def callback_h3(self, instance):
        if self.h3.state == 'down':
            self.h3.text = 'Dif map: ON'
        else:
            self.h3.text = 'Dif map: OFF'

    def callback_i2(self, instance):
        if self.i2.state == 'down':
            self.i2.text = 'Energy: ON'
        else:
            self.i2.text = 'Energy: OFF'

    def callback_i2_a(self, instance):
        if self.i2.state == 'down':
            self.i2.text = 'Dim X: ON'
        else:
            self.i2.text = 'Dim X: OFF'

    def callback_i5(self, instance):
        if self.i5.state == 'down':
            self.i5.text = 'Delay: ON'
        else:
            self.i5.text = 'Delay: OFF'

    def callback_i5_a(self, instance):
        if self.i5.state == 'down':
            self.i5.text = 'Dim Y: ON'
        else:
            self.i5.text = 'Dim Y: OFF'

    def callback_j2(self, instance):
        if self.j2.state == 'down':
            self.j2.text = 'Total electrons: ON'
        else:
            self.j2.text = 'Total electrons: OFF'

    def callback_j3(self, instance):
        if self.j3.state == 'down':
            self.j3.text = '[0,1]: ON'
        else:
            self.j3.text = '[0,1]: OFF'

    def callback_j4(self, instance):
        if self.j4.state == 'down':
            self.j4.text = '[-1,1]: ON'
        else:
            self.j4.text = '[-1,1]: OFF'

    def callback_k2(self, instance):
        if self.k2.state == 'down':
            self.k2.text = 'Time axis'
        else:
            self.k2.text = 'Energy axis'

    def callback_k2_a(self, instance):
        if self.k2.state == 'down':
            self.k2.text = 'Dim Y'
        else:
            self.k2.text = 'Dim X'

    def callback_k3(self, instance):
        if self.k3.state == 'down':
            self.k3.text = 'Difference: ON'
        else:
            self.k3.text = 'Difference: OFF'

    def callback_k4(self, instance):
        if self.k4.state == 'down':
            self.k4.text = 'Waterfall: ON'
        else:
            self.k4.text = 'Waterfall: OFF'

    def callback_k5(self, instance):
        if self.k5.state == 'down':
            self.k5.text = 'Add map: ON'
        else:
            self.k5.text = 'Add map: OFF'

    def callback_k6(self, instance):
        if self.k6.state == 'down':
            self.k6.text = 'Legend: ON'
        else:
            self.k6.text = 'Legend: OFF'

    def callback_m2(self, instance):
        if self.m2.state == 'down':
            self.m2.text = 'Norm [0,1]: ON'
        else:
            self.m2.text = 'Norm [0,1]: OFF'

    def callback_m3(self, instance):
        if self.m3.state == 'down':
            self.m3.text = 'Norm [-1,1]: ON'
        else:
            self.m3.text = 'Norm [-1,1]: OFF'

    def callback_m4(self, instance):
        if self.m4.state == 'down':
            self.m4.text = 'Smoothing: ON'
        else:
            self.m4.text = 'Smoothing: OFF'

    def callback_m5(self, instance):
        if self.m5.state == 'down':
            self.m5.text = 'Derivative: ON'
        else:
            self.m5.text = 'Derivative: OFF'

    def callback_DLD(self, instance):
        if self.DLD_toggle.state == 'down':
            self.DLD_toggle.text = 'DLD4Q'
        else:
            self.DLD_toggle.text = 'DLD1Q'

    def create_plot_mode_callback(self, instance):
        if self.create_plot_mode.state == 'down':
            self.create_plot_mode.text = 'Mode: Save Fig'
        else:
            self.create_plot_mode.text = 'Mode: PyQt'

        if self.create_cut_plot_mode.state == 'down':
            self.create_cut_plot_mode.text = 'Mode: Save Fig'
        else:
            self.create_cut_plot_mode.text = 'Mode: PyQt'

        if self.map_mode.state == 'down':
            self.map_mode.text = 'Mode: MicroBunch'
        else:
            self.map_mode.text = 'Mode: Time delay'


if __name__ == "__main__":
    MainApp().run()
