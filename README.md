# 2dbat-PI 
	- 2DBAT-PI: 2D Bayesian Automated Tiltedringfitter
	- Version 1.0.0 (24 Jan 2024)
	- by Se-Heon Oh (Department of Physics and Astronomy, Sejong University, Seoul, Korea)
	
	2dbat-PI is a new tool for deriving galaxy rotation curve based on Bayesian nested sampling. 

# Prerequisite

	- Python3.10
	- Python3 virtual environment module: venv (normally, venv is installed along with python3)
	- The latest version of dynesty 2.0.3 will be installed for Bayesian analysis utilizing nested sampling.
	- Tested for Ubuntu 18.04 LT and macOS Monterey 12.6 on Apple M2

# Installation

1. Make a directory for the python3.10 virtual environment for 2dbat-PI. For example, 

		[seheon@sejong00] makedir /home/seheon/research/2dbat-PI
	

2. Set up a 'python3.10 virtual environment' in the '2dbat' directory created.

		[seheon@sejong00] python3.10 -m venv /home/seheon/research/2dbat-PI
		
		--> Then, activate the virtual environment.
		--> If you are using CSH or TCSH
		[seheon@sejong00] source /home/seheon/research/2dbat-PI/bin/activate.csh
		
		--> If you are using BASH
		[seheon@sejong00] source /home/seheon/research/2dbat-PI/bin/activate
		
		--> Now, you enter the python3.10 virtual environment, named '2dbat-PI'
		(2dbat-PI) [seheon@sejong00]
		
		--> FYI, to deactivate, just type 'deactivate'
		(2dbat-PI) [seheon@sejong00] deactivate	
	
3. Install 2dbat-PI via github,

		(2dbat-PI) [seheon@sejong00] git clone https://github.com/seheon-oh/2dbat-PI.git
		--> Enter 2dbat-PI directory created, and install it via 'pip' command
		(2dbat-PI) [seheon@sejong00] cd 2dbat-PI
		
		--> Now, install baygaud-PI using 'pip' command
		(2dbat-PI) [seheon@sejong00] pip install .
		
		--> Now it should install the modules required for 2dbat-PI python3.10 environment.
		It takes a while…
		
		--> Note that the required python packages for 2dbat-PI are only compatible
		within the virtual environment that has been created. The required package list can be found in
		'requirements.txt' in the '2dbat-PI' directory.
		
		(Optional) Or, for developer installation (this installs the package in the same location to
		allow for changes to be reflected across the environment), use the following command,	
		(2dbat-PI) [seheon@sejong00] python3 setup.py develop
		
		--> Now, it is ready for running 2dbat-PI now.


# Quick Start

1. Setting up data (HI velocity field being fitted)

		--> Make your own directory where the data files including the HI velocity field in FITS format are located.

		--> Put the input data files (FITS) into the data directory. 

		--> As an example, a test velocity field ('test_vf.fits') is provided in 'wdir' directory in _2dbat_params.py:

		|| Data directory
		--> For example (see '_2dbat_params.py' in 'src'),

		|| Set data directory; in _2dbaty_params.py in 'src'
		'wdir':'/Users/seheon/research/projects/2dbat.pi/2dbat-PI/demo/test_cube',

		|| Input HI velocity field (required)
		'input_vf':'test1.fits'


2. Setting up 2dbat-PI parameters

		--> Open ‘_2bdaat_params.py’ file using vim or other text editors. Update keywords upon
		your system accordingly. Find "UPDATE HERE" lines and edit them as yours. Short descriptions
		(recommendation) are given below.
		
		'_vlos_lower'
		'_vlos_upper'
		'ring_w'
		'free_angle':0
		'cosine_weight_power':0
		'xpos_bounds_width':2 <--- ring_w / 2. (recommended)
		'ypos_bounds_width':2 <--- ring_w / 2. (recommended)

		# ----------------------------------
		# PA-BS
		'n_pa_bs_knots_inner':0,
		'k_pa_bs':0, # 1:linear, 2:quadractic, 3:cubic


		# ----------------------------------
		# INCL-BS
		'n_incl_bs_knots_inner':0,
		'k_incl_bs':0, # 1:linear, 2:quadractic, 3:cubic


		# ----------------------------------
		# VROT-BS
		'n_vrot_bs_knots_inner':0,
		'k_vrot_bs':3, # 1:linear, 2:quadractic, 3:cubic

		# grids
		'x_grid_2d':1, # <---- recommended
		'y_grid_2d':1, # <---- recommended
		'x_grid_tr':1, # <---- recommended
		'y_grid_tr':1, # <---- recommended

		#  ______________________________________________________  #
		# [______________________________________________________] #
		# parallellisation parameters
		'num_cpus_tr_ray':1,  # <---- recommended (num_cpus_tr_ray x num_cpus_tr_dyn = num_cpus_2d_dyn)
		'num_cpus_tr_dyn':8,  # <---- recommended 
		'num_cpus_2d_dyn':8,  # <---- UPDATE HERE (MAXIMUM NUMBER OF CPUs)

		

	
3. Running _2dbat.py

		--> You can run '_2dbat.py' with a running-number.

		(2dbat-PI) [seheon@sejong00] python3 _2dbat.py 1

		--> Check the running processes (utilizing multi-cores) on the machine.
		--> Check the output directory where the 2dbat fitting results are stored.
		
		# Output directory in ‘_2dbat_params.py’
		

# Cite

	1. Robust profile decomposition for large extragalactic spectral-line surveys (main algorithm paper)
		Oh, S. H., Staveley-Smith, L. & For, B. Q., 13 Mar 2019, In: Monthly Notices of the Royal Astronomical Society. 485, 4, p. 5021-5034 14 p.

	2. Kinematic Decomposition of the HI Gaseous Component in the Large Magellanic Cloud (application paper)
		Oh, Se-Heon, Kim, Shinna, For, Bi-Qing & Staveley-Smith, Lister, 1 Apr 2022, In: Astrophysical Journal. 928, 2, 177.
	
	3. Gas Dynamics and Star Formation in NGC 6822 (application paper)
		Park, H. J., Oh, S. H., Wang, J., Zheng, Y., Zhang, H. X. & de Blok, W. J. G., 1 Sep 2022, In: Astronomical Journal. 164, 3, 82.
		
	4. WALLABY Pilot Survey: HI gas kinematics of galaxy pairs in cluster environment (application paper)
		Kim, S. J., Oh, S. H. et al., 2023, In Monthly Notices of the Royal Astronomical Society
	
	5. GLOBAL HI PROPERTIES OF GALAXIES VIA SUPER-PROFILE ANALYSIS (application paper)
		Kim, M. & Oh, S. H., Oct 2022, In: Journal of the Korean Astronomical Society. 55, 5, p. 149-172 24 p.
		
	6. FEASTS: IGM cooling triggered by tidal interactions through the diffuse HI phase around NGC 4631 (application paper)
		Jing Wang, Dong Yang, Se-Heon Oh, Lister Staveley-Smith, Jie Wang, Q. Daniel Wang, Kelley M. Hess, Luis C. Ho, Ligang Hou, Yingjie Jing, Peter Kamphuis, Fujia Li, Xuchen Lin, Ziming Liu, Li Shao, Shun Wang, Ming Zhu, Astrophysical Journal (2023)



