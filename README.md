# PESdata
Graphical user interface for processing and visualization of multidimensional photoemission spectroscopy (PES) data measured at free-electron lasers (FELs), synchrotrons, and laboratory setups


## Table of content
* [Introduction](#introduction)
* [Installation](#installation)
* [How to use](#how-to-use)
    + [Section I - Upload runs](#section-i---upload-runs)
    + [Section II - Calculate delay-energy map (computationally demanding part)](#section-ii---calculate-delay-energy-map--computationally-demanding-part-)
    + [Section III - Create delay-energy map visualization](#section-iii---create-delay-energy-map-visualization)
    + [Section IV - Slice delay-energy map data](#section-iv---slice-delay-energy-map-data)
* [Special features](#special-features)
* [Setting up Julia-enabled mode](#setting-up-julia-enabled-mode)

## Introduction
This program is developed, first of all, for convenient manipulation of data measured at large-scale facilities, which can not be handled with conventional data analysis software and requires the use of programming. Thus, PESdata makes it accessible to a larger community of scientists working on photoemission-based experiments. The graphical user interface is developed using an open-source, cross-platform Python framework [Kivy](https://kivy.org/). The application works based on a number of Python objects designed for reading out, filtering, histograming, selecting, and combining input data of different shapes but resulting in the generation of labeled data arrays of the same structure stored as [Xarray](https://xarray.dev/) DataArray objects. Thus, the same visualization approach based on the [Matplotlib](https://matplotlib.org/) library is used for various data. Besides, DataArray objects can be saved as [netCDF files](https://docs.xarray.dev/en/stable/user-guide/io.html), which allows for avoiding repeated processing of the same datasets and storing and sharing condensed preprocessed data. Currently, the following modules are available and under development:
- **[WESPE module]** Time-resolved XPS data measured with two time-of-flight analyzers of [WESPE endstation](https://uniexp.desy.de/e47432/e297712/e306033/) at the [PG2 beamline](https://photon-science.desy.de/facilities/flash/beamlines/pg_beamlines_flash1/index_eng.html) of [FLASH](https://flash.desy.de/) free-electron laser (Hamburg, Germany)
- **[ALS module]** Time-resolved XPS data measured with a hemispherical analyzer of ambient-pressure photoemission spectroscopy setup at the [11.0.2 beamline](https://als.lbl.gov/beamlines/11-0-2/) of the [Advanced Light Source](https://als.lbl.gov/about/about-the-als/) synchrotron (Berkeley, USA)
- **[Momentum Microscope module]** Time-resolved XPS, ARPES, and XPD data measured with a momentum microscope of the [HEXTOF endstation](https://photon-science.desy.de/news__events/news__highlights/archive/archive_of_2020/hextof_at_flash___merging_time_resolved_photoemission_techniques_into_a_new_instrument/index_eng.html) at the [PG2 beamline](https://photon-science.desy.de/facilities/flash/beamlines/pg_beamlines_flash1/index_eng.html) of [FLASH](https://flash.desy.de/) free-electron laser (Hamburg, Germany) and the [TR-XPES endstation]( https://www.xfel.eu/facility/instruments/sxp/instrument/index_eng.html) of [SXP instrument]( https://www.xfel.eu/facility/instruments/sxp/index_eng.html) at the SASE3 beamline of [European XFEL](https://www.xfel.eu/) free-electron laser (Hamburg, Germany)
- **[Igor Binary Wave module]** Angle-resolved photoemission spectroscopy (ARPES) data in Igor Binary Wave format (.ibw), measured at beamlines of various synchrotrons as [BESSY II](https://www.helmholtz.de/forschung/forschungsinfrastrukturen/lichtquellen/bessy-ii/) (Berlin, Germany) and [PETRA III](https://www.desy.de/research/facilities__projects/petra_iii/index_eng.html) (Hamburg, Germany) **[early access]**
- **[Casa XPS module]** Static 2D XPS data (xy curves) stacked to 3D data with labels specified by a user (e.g., plotting energy dispersive curve as a function of deposition/etching time, the temperature of annealing) **[early access]**

## Installation
To run the application, you need to execute 'PESdata_GUI.py' in an IPython console having the necessary Python packages installed. The recommended way is to install [Anaconda]( https://www.anaconda.com/) Python distribution because it includes Spyder (which has an IPython console) and most of the necessary packages. Then, after this, you need to install in Spyder IPython console several extra packages:
1) [Kivy](https://kivy.org/) (mandatory, needed for GUI).
```python
pip install kivy
```
2) [lmfit](https://pypi.org/project/lmfit/) (mandatory, needed for data fitting inside the app).
```python
pip install lmfit
```
3) [Fastparquet](https://pypi.org/project/fastparquet/) (optional for the WESPE module and mandatory for the Momentum Microscope module, needed for loading parquet files).
```python
pip install fastparquet
```
4) [PyJulia](https://pypi.org/project/julia/) (optional for the ALS module and 2D view of the Momentum Microscope module to speed up large dataset processing, obligatory for more computationally demanding 3D view of the Momentum Microscope module).
```python
pip install julia
```
If you want to use the packages outside Spyder or you face problems during installation inside Spyder, you can install any of these packages for the whole conda environment. For this, one should execute the following in the Anaconda prompt (put package name instead of **\*package\***: kivy, lmfit, fastparquet, or julia):
```conda
conda install *package* -c conda-forge
```
## How to use
The main window includes four sections followed by green execution buttons:
1. Upload runs
2. Compute histogram
3. Create visualization
4. Make slices

The first two sections vary slightly depending on the data input. The switch of GUI happens on the press of the 'Upload runs' button since a check of the 'File directory' folder content is performed. In the picture below, Section I of the WESPE data module (left panel) contains a switch between two available TOF analyzers (DLD4Q or DLD1Q). In contrast, for other modules, it is absent. In Section II, inputs partially overlap and differ partly because of different data structures.

<p align="center">
    <img align="middle" src="https://github.com/potorocd/PESdata/blob/main/packages/readme/Main_menu.png" alt="Main menu"/>
</p>

### Section I - Upload runs
<p align="center">
    <img align="middle" src="https://github.com/potorocd/PESdata/blob/main/packages/readme/Main_menu_I.png" alt="Main menu I"/>
</p>

This section serves for loading data files corresponding to specific runs.

***File directory*** – specify the directory where data is stored

***Run numbers*** – specify run numbers separated by coma

***DLD4Q/DLD1Q switch*** – select between two TOF analyzers **[only for the WESPE module]**

For the WESPE module, the data files are searched as ‘FileDirectory\RunNumber\RunNumber_energy.mat’ for Matlab files, both 4Q and 1Q data are stored in a single file. In the case of reading parquet files, data will be searched as ‘FileDirectory\RunNumber.4Q’ or ‘FileDirectory\RunNumber.1Q’ depending on the spectrometer choice because the data of two TOFs are stored in separate files.
For the ALS module, data is searched as ‘FileDirectory\RunNumber’, where the program expects multiple h5 files recorded at various phase shifter values.

The ‘App settings’ button opens a popup window with application settings.

<p align="center">
    <img align="middle" src="https://github.com/potorocd/PESdata/blob/main/packages/readme/App_settings.png" alt="App settings"/>
</p>

The settings are stored in the ‘packages/config.json’ file. To save changes to the file, press ‘Save settings’. On press of ‘Load default settings’, the ‘packages/default_config.json’ file is read, it can not be changed via the application window. However, one can rewrite ‘default_config.json’ with ‘config.json’ in the preferred state, and then one can always return to this state directly from the GUI by pressing ‘Load default settings’. Graph parameters can be changed without restarting the application; changing the app font parameters requires this.

The ‘Upload runs’ button reads out data from the specified runs and creates a summary popup window. For the WESPE module, on the top, the results of three checks are shown: 1) check if all runs are static or delay stage scans; 2) check if runs contain data from similar energy regions; 3) check if the monochromator position is the same for all runs. See below summary examples for the WESPE module (left panel) and for the ALS module (right panel).

<p align="center">
    <img align="middle" src="https://github.com/potorocd/PESdata/blob/main/packages/readme/Upload_runs.png" alt="Upload runs"/>
</p>

### Section II - Compute histogram (computationally demanding part)
This section serves for the generation of a labeled 2D or 3D array using data uploaded in **Section I**, which can be represented as a false-color plot in **Section III**. In the case of tr-XPS data, the y-axis corresponds to the time domain, and the x-axis corresponds to the electron energy domain. The Momentum Microscope module provides the freedom to put any dimension on any axis.

For tr-XPS data, the x-axis is described with two coordinates: 1) Kinetic energy – values taken from data files; 2) Binding energy calculated as ‘mono value – Kinetic energy - work function of the spectrometer (4.5 eV as default for the WESPE module)’. The y-axis is described with one coordinate as default: ‘Delay stage values’/’Phase shifter values’ taken from data files. In the WESPE module, for static runs (no delay stage value data included in files), all electrons are assigned to delay stage values equal to the run number in ps. When T0 in **Section II** is ‘on’, the second coordinate called ‘Delay relative t0’ is created. The values are calculated as ‘time zero – delay stage values/phase shifter values’.

<p align="center">
    <img align="middle" src="https://github.com/potorocd/PESdata/blob/main/packages/readme/Main_menu_II.png" alt="Main menu II"/>
</p>

#### Histogram parameters
***T0: ON/OFF*** – when the switch is on, the ‘Delay relative t0’ coordinate is created

***Kinetic energy/Binding energy*** – toggle determining energy coordinate for the first delay-energy map visualization; besides, it determines what energy coordinates are used for combining several individual runs (Binding energy coordinate is compensated for mono value change)
#### Binning [WESPE and Momentum Microscope modules]
***Energy step*** or ***Dim X step*** – determines binning in the energy domain (x-axis)

***Time step*** or ***Dim Y step*** – determines binning in the time domain (y-axis)

***Dim Z step*** – determines binning for the z-axis; the processor considers values larger or equal to 100 as a command to automatically determine the z-step resulting in the corresponding number of slices along the z-axis (i.e., '10' would request absolute step of 10, '110' would request 110 slices) **[only for the Momentum Microscope module]**

*Note: for the Momentum Microscope module, x/y/t values from DLD are divided by 1000. Otherwise, large values do not fit as tick labels. Therefore, to bin to every single pixel/TOF arb. unit, one needs to set 0.001 instead of 1 as a step.*
#### Filtering [WESPE and Momentum Microscope modules]
These filters allow the exclusion of electrons from input data not satisfying specific conditions before histogramming.

***MacroB: ON/OFF*** – switch determining if MacroBunch ID filter is applied

Limits in % are specified as two values separated by coma

***Custom: ON/OFF*** – switch determining if Custom filter is applied

Set custom filter boundaries for various dimensions by specifying the following arguments separated with a comma:
```conda
dimension_key,boundary1,boundary2
```
Semicolon separates different requests for filtering. E.g.:
```conda
x,0.4,0.9;y,0.4,0.9;t,4.46,4.48
```
This entry would filter X-Pixel from 0.4 to 0.9 (400 to 900 actual pixels), Y-pixel from 0.4 to 0.9 (400 to 900 actual pixels), and TOF from 4.46 to 4.48 (4460 to 4480 TOF arb. units).
The following dimension keys are currently available:
```conda
x - X pixel
y - Y pixel
t - Time-of-flight for MM/Kinetic energy for WESPE
d - delay stage position
b - Micro Bunch ID
mono - monochromator values
bam - beam arrival monitor values
```
#### Processor arguments **[ALS module]**
***DLD bins*** – the approximate number of bins for creating a 2D histogram (energy channels vs. DLD time) for every single h5 file recorded at a specific phase shifter value. The larger the number – the higher the DLD time resolution, but the slower the processing. Experimentally, it was determined that approximately 1500 bins are sufficient for multi-bunch data to extract camshaft bunches out of the dataset. In the case of two-bunch data, the number of bins is determined as this coefficient divided by 10 (e.g., as default 1500/10 = 150).

***Bunch selection*** – select a bunch number or a slice of bunches (start and end bunches separated with coma), which is used for visualization. A specific bunch is usually used to select the pumped bunch, synchronized with the optical pump, a slice of bunches – for summing up all available bunches in case of measuring reference laser-off data for energy scale or transmission calibration.
#### Modes [WESPE, ALS, Momentum Microscope modules]
**[WESPE and ALS modules]:** ***‘Time delay’/’MicroBunch’*** – switch to an alternative mode that creates a 2D labeled array, where MicroBunch ID/Bunch ID serves as the y-axis label instead of Time delay. It can be used for comparison of pumped and unpumped data.
**[Momentum Microscope module]:** ***Select dims*** – input field that allows the user to define the sources for the 2D/3D histogram computation. The user needs to put two/three key letters in this field for 2D/3D view (3D view requires setting up the Julie-enabled mode). The first letter stands for the X dimension, the second for the Y dimension, and the third for the Z dimension. A false-color plot will represent the XY plane, whereas the Z dimension will be bound to a manually adjusted slider. Slider position change leads to an update of the XY slice. Currently, the following options are available (key - source):
```conda
x - X pixel
y - Y pixel
t - Time-of-flight
d - delay stage position
b - Micro Bunch ID
```
**'td'** is the default, which means that the Time-of-flight will be on the X-axis, and the delay stage position will be on the Y-axis. In other words, this is tr-XPS mode.
**'xyt'** is recommended for finding the Fermi surface since it allows one to examine the kx/ky plane at various TOF (Kinetic energy) values. Then, applying the Custom filer allows the filtering out of TOF values above and below the Fermi surface (e.g., 't,4.46,4.48'). Thus, after the filtering, **'xyd'** would demonstrate the Fermi surface as a function of pump-probe delay.

### Section III - Create visualization
<p align="center">
    <img align="middle" src="https://github.com/potorocd/PESdata/blob/main/packages/readme/Main_menu_III.png" alt="Main menu III"/>
</p>

This section serves for the instant generation of visualizations using the 2D/3D labeled array generated in **Section II**.

<p align="center">
    <img align="middle" src="https://github.com/potorocd/PESdata/blob/main/packages/readme/Map_example.png" alt="Map example" width="600"/>
</p>

#### Plot parameters
***T0: ON/OFF*** – switch between ‘Delay stage values’/’Phase shifter values’ and ‘Delay relative t0’ coordinates in the time domain

***Difference plot: ON/OFF*** – switch to the difference map plot where the averaged energy dispersive curve before -0.25 ps is subtracted from the whole array row by row. It helps to emphasize minor variations of intensity as a function of time delay. It also works for 3D view by showing the difference between X/Y slices selected by two ranged sliders bound to the Z axis.

***Kinetic energy/Binding energy*** – toggle to select the coordinate for the energy axis

#### Delay/Energy ROI or Dim X/Dim Y ROI
***Energy: ON/OFF*** or ***Dim X: ON/OFF*** – when on, the cutoff for the energy axis (x-axis) is activated, where limits are determined by two values separated by coma

***Delay: ON/OFF*** or ***Dim Y: ON/OFF***  – when on, the cutoff for the time axis (y-axis) is activated, where limits are determined by two values separated by coma

#### Normalizing
***Electrons per dim: ON/OFF*** – switch on normalization to compensate for different acquisition times for every specific time delay. Every energy dispersive curve is normalized to the total sum of electrons within the whole energy window.

***[0,1]: ON/OFF*** – normalization of every energy dispersive curve to [0,1] as [min, max].

***[-1,1]: ON/OFF*** – normalization of every energy dispersive curve to the maximal value between abs(min) and abs(max). Suitable for the difference plots, which include negative values.

***Mode: PyQt/Save Fig*** – PyQt corresponds to plot visualization in a popup window, while Save Fig creates an image in the ‘FileDir\fig_output’ folder.

***Save ASCII*** – it saves the whole delay energy map from the current visualization to ‘FileDir/ASCII_output/Maps/DateTime’. Every time delay value is saved as a separate .dat file and contains the corresponding energy dispersive curve.

### Section IV - Make slices
<p align="center">
    <img align="middle" src="https://github.com/potorocd/PESdata/blob/main/packages/readme/Main_menu_IV.png" alt="Main menu IV"/>
</p>

This section allows making of 1D slices out of the last visualization state of the 2D labeled array created in **Section III**. Changing the toggle state leads to switching between the Y and X axes. In the case of the 3D view, it slices the last state of the false-color plot from **Section III** (last Dim Z slider position).

<p align="center">
    <img align="middle" src="https://github.com/potorocd/PESdata/blob/main/packages/readme/Slice_example.png" alt="Slice example" width="600"/>
</p>

#### Slice mode
***Time axis/Energy axis*** or ***Dim Y/Dim X*** – switch between slices across the y- or x-axis

***Waterfall: ON/OFF*** – ‘on’ selection introduces y offset between curves to avoid overlapping

***Add map: ON/OFF*** – ‘on’ selection adds map plot to slice plot where the regions used for generation of slices are displayed

***Legend: ON/OFF*** – show or hide the legend of the slice plot

#### Parameters
***Position*** – specification of the slice positions separated with commas; **‘Main’** instead of a number automatically finds the position of the most intense feature; **‘SB, hv’** will determine the sideband position by the addition of ‘hv’ to the position of the most intense feature; keywords **'cci', 'cc', 'ccp', 'cct'** can be specified here for performing cross-correlation

***Width*** – specification of slice widths separated with commas; if one value is specified, it will be applied to every slice

***Difference: ON/OFF*** – ‘on’ selection adds additional curves to the plot calculated as a difference of every curve starting from the second one and the first one

#### Processing
***Norm [0,1]: ON/OFF*** – ‘on’ selection normalizes the curves to [0,1] as [min, max]

***Norm [-1,1]: ON/OFF*** – ‘on’ selection normalizes the curves to the maximal value between abs(min) and abs(max).

***Smoothing: ON/OFF*** – ‘on’ selection applies the Savitzky-Golay filter to all slices

***Derivative: ON/OFF*** – ‘on’ selection applies the first derivative to all slices, which can be helpful for determination of time zero

***Fit*** – it performs fitting of the first slice with either a single Voigt lineshape or with a biexponential decay curve, which is particularly helpful for time zero determination

<p align="center">
    <img align="middle" src="https://github.com/potorocd/PESdata/blob/main/packages/readme/Voigt_fit_example.png" alt="Voigt fit example" width="600"/>
</p>

***Mode: PyQt/Save Fig*** – PyQt corresponds to plot visualization in a popup window, while Save Fig creates an image in the ‘FileDir\fig_output’ folder.

***Save ASCII*** – it saves all slices from the current visualization to ‘FileDir/ASCII_output/Cuts/DateTime’.

## Special features
### For all modules
#### Autocompletion in the ‘Upload runs’ section
Autocompletion works on a double click. For the ‘File directory’ field, it takes the folder path by the last directory separator symbol and searches for folders containing the characters after the last directory separator symbol. For the ‘Run numbers’ field, it scans the ‘File directory’ folder and searches for files containing the symbols after the last separator. The closer the specified characters are to the target folder or run name, the higher the chance that autocompletion retrieves the expected name.
#### Switching between two color mapping schemes
By default, colormap colors are mapped linearly from minimal to maximal value. However, there are cases when it hinders some fine dynamics around time zero (for example, sidebands). Therefore, there is a possibility to switch to a different mapping scheme, which is determined by two linear color scales centered relative to the user-selected center point. This point can be specified in the ‘color map scale’ field in the settings menu and is perceived by the program as a fraction of the intensity range, where 0 is the minimal value, and 1 is the maximal value. I.e., if one specifies, for example, 0.25, then 25 % of the lower range will be described by 50 % of the color range, which gives a finer resolution for lower-intensity features. In contrast, if one sets this value between 0.5 and 1, it provides higher color resolution for higher-intensity features.
As default, the scaling factor is set to 1 because values higher or equal to 1 lead to using a standard linear scale. The same linear scale can be achieved by setting 0.5, which gives the same color range for lower-intensity and higher-intensity ranges. This coefficient is stored as a 'TwoSlopeNorm' parameter in the config.json file. The program reads out this value when a false-color plot is generated.

#### Saving preprocessed data as NetCDF files
To avoid repeated histogramming of data performed with the same parameters, the result is saved as netCDF files. Besides, it allows the storing and sharing of condensed pre-processed data.
The app has three saving modes, which can be set in the app settings menu in the ‘save .nc’ field.
I (‘on’) – the app tries to read a pre-processed netCDF file, if such file is not found, it processes the raw data file from scratch and saves the result.
II (‘off’) – the app always starts processing raw data and doesn’t save the result.
III (any other string apart from ‘on’ or ‘off’, e.g., ‘w’) – the app always starts processing raw data, but it also always saves the result. If pre-processed files already exist, it overwrites them. This can be helpful, for example, during beamtimes, when you treated a run that wasn’t measured to the end, then you can rewrite the files directly from the app without deleting the ‘.nc’ files manually.

#### Keywords for finding primary features and sidebands
**‘main’** – finds the coordinate with the highest median intensity
**‘sb’** – finds the coordinate of the highest median intensity, then adds 2.407 [eV], which corresponds to the expected first-order sideband position for an optical pump wavelength of 515 nm
**‘sb,value’** – the same as ‘sb’, but ‘value’ overrides the default 2.407 eV photon energy of the optical pump (e.g., ‘sb,3.61’ would work for finding the first-order sideband position for the third harmonic of the optical pump, 343 nm)
*Note: letters can be lower- or uppercase.*

#### Peak position variation determination using cross-correlation
Cross-correlation allows one to determine the similarity of arrays as a function of their displacement. If we use energy-dispersive curves as such arrays, it allows facile determination of peak position variation relative to a user-defined reference, which is helpful in time-resolved XPS experiments. PESdata allows cross-correlation to be called by typing a keyword inside the slice position field and pressing the ‘Slice 2D Histogram’ execution button. The keyword determines what reference the user picks for the cross-correlation procedure.
##### ‘cci’ keyword [all modules]
The **‘cci’** keyword stands for cross-correlation inside the 2D histogram generated in Section III and can be used for any data input. First, it determines a reference 1D array by taking a slice from the same 2D histogram to which cross-correlation is applied. If time zero is known, it takes a cut with negative pump-probe values (values < -2.5*step along the y-axis) and computes a mean. If it is unknown, it takes a slice of 10 % of the largest delay stage values (y-axis). After this, it performs cross-correlation of every slice along the Y-axis with the reference and determines the shift relative to this reference.
### Specific for the ALS module
##### ‘cc’ keyword
The ‘cc’ keyword stands for cross-correlation. Instead of taking reference from the same 2D histogram, which is generated for bunches specified in the ‘Plot bunch’ field, it computes a second histogram using bunch numbers selected by the user, which is considered an unpumped reference. Then, it performs cross-correlation of a slice from the ‘pumped’ map and the ‘unpumped’ map at every phase shifter value. 
Examples of available reference options:
1) ‘cc,6’ – if you specify a single value separated with a comma, it generates the unpumped map using it as the unpumped bunch number (bunch number 6 in this case).
2) ‘cc,3,6’ – if you specify two values separated with a comma, it takes all bunch numbers between min and max values to generate the unpumped delay-energy map (in this case, these are bunches 3,4,5,6).
3) ‘cc’ – if you specify the plain ‘cc’ keyword, it uses the single previous bunch to generate the unpumped delay-energy map (i.e., comparison of ‘Plot bunch’ with the previous bunch; if  ‘Plot bunch’ is 7, it is compared with bunch 6).
4) ‘ccp’ – if you specify the plain ‘ccp’ keyword (‘p’ stands for previous), it will use all previous bunches to generate the unpumped delay-energy map (if ‘Plot bunch’ is 7, it will be compared with a combination of bunches 1,2,3,4,5,6)
#### ‘cct’ keyword
‘t’ stands for total. It provides a more quantitative analysis of shift as a function of bunch number. It performs the ‘cc’ approach for every single bunch, where, as the unpumped signal, the previous to  ‘Plot bunch’ bunch value is taken. Thus, for every bunch ID, we get a shift as a function of the phase shifter value. Then, the shift for the pumped bunch is determined as the max value to get the amplitude since an abrupt shift from zero to max is expected. For all other bunches, the shift value is determined as a mean. 
#### Transmission function calibration
The electron collection efficiency of hemispherical analyzers is not constant, it varies over the electron kinetic detection energy range. To compensate for this factor, a transmission function must be applied (dependency of detection efficiency over electron kinetic energy). It can be determined by measuring a spectrum of a featureless region consisting of only inelastically scattered electrons with the same spectrometer settings, which are used for core-level spectra acquisition (same voltages and apertures). To apply a transmission function to a specific run, you need to put a file called **‘Transmission function.txt’** in the run folder. This file should consist of two columns:
1) energy channel number;
2) electron counts detected by the corresponding channel for the featureless region.

Below is an example of the **‘Transmission function.txt’** file (the first and last three lines out of a total of 126 lines are shown corresponding to 126 DLD energy channels):
```conda
1.250000000000000000e+02    9.945912716572154366e+00
1.240000000000000000e+02    1.447682850967724733e+01
1.230000000000000000e+02    1.079315713316904102e+01
....................................................
2.000000000000000000e+00    2.578569963555743527e-01
1.000000000000000000e+00    2.578569963555743527e-01
0.000000000000000000e+00    2.578569963555743527e-01
```
When a user generates a delay-energy map, the app tries to open the transmission function file. If it finds one, it determines the coefficient by which counts detected by every channel are divided. This coefficient is determined as follows. The coefficient for channels with a low count rate (< 5 % from the maximum) is set to 1. For other channels, the mean number of detected counts is determined, and the coefficient is taken as the counts divided by mean counts.
The recommended way to generate **‘Transmission function.txt’** is to open the spectrum recorded for a featureless region with the *PESdata* and save it using the ‘Save ASCII’ button. Then, you can directly put the saved file to the target run folder and rename it to **‘Transmission function.txt’**.

#### Energy axis calibration
The kinetic energy of detected photoelectrons in raw data is recorded as channel numbers, which represent the electron hit position along the energy-dispersive axis of the hemispherical analyzer. Thus, it requires recalculation of energy channels into kinetic or binding energy in electronvolts. Since the dependency is linear, it requires the determination of only two coefficients, k and b, of a linear function y = k\*x + b, where y is kinetic/binding energy,  and x is the energy channel. These coefficients can be determined, for example, by measuring a set of reference core-level spectra (e.g., Au 4f core level) with the same analyzer settings used for tr-XPS spectra acquisition. The spectra in the calibration set have to be shifted along the energy axis by known user-defined values in electronvolts. This can be done either by changing the photon energy with a specific step or by applying a known bias to the reference sample. Then, variation of the peak position of reference spectra can be determined in arbitrary energy channel units and correlated with peak position variation in electronvolts. To apply energy calibration to a specific run, one has to put a file called **‘Energy calibration.txt’** into a run folder. This file must consist of five values separated by a comma. For example:
```conda
BE,-0.13709,91.93986,686,4.5
```
**Value 1**  can be either BE or KE, which tells the app what axis you calibrate using the **k** and **b** coefficients of the **y = k\*x + b** curve. BE stands for binding energy, and KE stands for kinetic energy.
**Value 2** is the **k** coefficient.
**Value 3** is the **b** coefficient.
**Value 4** is the photon energy.
**Value 5** is the work function of the spectrometer.
Note: Values 4 and 5 are used for the calculation of the second axis. For example, from the example above, we calibrate the BE axis using k and b coefficients. Then, we calculate the KE axis as **‘photon energy - BE -  WF’**.
### Specific for WESPE and Momentum Microscope modules
#### User-selected processing runs as static
If one puts the letter 's' at the end of a run number ('Upload runs' section), the program considers it a **static run** and assigns all electrons to the **delay stage value** equal to the **run number**. I.e., it will consider that for run '46615s', all electrons were detected at a delay stage value equal to 46615 ps. This allows the user to plot different input data in a single delay-energy map (pump-probe scans and static data where the optical pump is off). For example, if the user wants to compare two survey scans, one must specify '46615s,46624s' as run numbers instead of '46615,46624'.
If the user wants to compare static scans with a pump-probe scan before and after time zero, then one can do the following:  
1) Set run numbers, i.e. - '46410s, 46413' (static, pump-probe scan).  
2) Set time zero (e.g., 1381.1 ps)  
3) Slice the delay-energy map at negative time delay, positive time delay, and arbitrary time delay of the static run, calculated as 'time zero - run number' (can be visually determined from the y-axis of the delay-energy map). E.g., cuts: '-0.8,0.8,-45028'. Widths: 1,1,1000. This allows one to compare negative pump-probe delay, positive pump-probe delay, and static (laser off) cases.
## Setting up Julia-enabled mode
1) Install [Julia]( https://julialang.org/downloads/)
2) Launch Julia and execute in Julia console:
    ```julia
    using Pkg
    Pkg.add(“PyCall”)
    Pkg.add(“FHist”)
    ```
3) Make sure that you installed the PyJulia package for Python (see [Installation](#installation) section).
4) Reboot your computer.
5) Find in the PESdata folder the following two files:
**PESdata\packages\config.json**
**PESdata\packages\ default_config.json**
Note: These config files store many parameters that are used by the app. ‘config.json’ – is the actual file that is used by the app. default_config.json stores the parameters that are applied when you press load default settings inside the GUI.
**Edit config files with a text editor.** You need to change the parameter named “jpath” from "C:/Users/dmpot/AppData/Local/Programs/Julia-1.9.2/bin/julia.exe" to your path to *julia.exe*.
Note: you need to use ‘/’ or ‘\\\’ as separator, ‘\’ will raise an error.
6) If everything is set up right, you’ll see in the Spyder console during raw data processing the following:
    ```conda
    ***Start of raw data processing***
    Loading Julia...
    ***Julia-enabled mode***
    ```
    Otherwise:
    ```conda
    ***Start of raw data processing***
    Loading Julia...
    ***All-Python mode***
    ```
    Note: Julia loading can take 30-60 seconds during raw data processing of the first run after launching the app. But for all subsequent runs, it happens immediately.
    In Julia-enabled mode, data processing is at least 4-5 times faster, which is very helpful for processing large datasets.