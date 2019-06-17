addpath('/data/Mehrdad/QuaDriGa_2017/quadriga_src')
more off
close all
clear all

numDrops = 40;                       % Number of random user drops to simulate
sequenceLength = 100;                % Number of steps in a track to simulate for each drop

K = 32;                              % Number of users
disp('32 users')
centerFrequency = 2.53e9;            % Center frequency in Hz
bandwidth = 1e6;                     % Bandwidth in Hz
numSubcarriers = 1024;               % Number of sub-carriers
subSampling = 4;                     % Only take every subsampling's sub-carriers

antennaHeight = 25;                  % Antenna height of the bse station in m
antennaSpacing = 1/2;                % Antenna spacing in multiples of the wave length
M_V = 8;                             % Number of vertical antenna elements
M_H = 4;                             % Number of horizontal antenn elements
M = 2*M_V*M_H;                       % Total number of antennas (factor 2 due to dual polarization)

minDistance = 10;                    % Minimum distance from the base station
maxDistance = 500;                   % Maximum distance from the base station
userHeight = 1.5;                    % Antenna height of the users
sectorAngle = 60;                    % Width of the simulated cell sector in deg
sectorAngleRad = sectorAngle/180*pi; % Width of the simulated cell sector in radians

lambda = 3e8/centerFrequency
speed_km_h = 30
speed_m_s = speed_km_h*1000/3600
coherenceTime = lambda/4/speed_m_s
num_symbols_coherence_time = coherenceTime/1e-3*14*subSampling

% Scenario
s = qd_simulation_parameters;                           % Set up simulation parameters
s.show_progress_bars = 0;                               % Disable progress bars
s.center_frequency = centerFrequency;                   % Set center frequency
s.sample_density = 2;                                   % 2 samples per half-wavelength
s.use_absolute_delays = 1;                              % Include delay of the LOS path

% Layout
l = qd_layout(s);                                       % Create new QuaDRiGa layout

% Base station
l.no_tx = 1;
l.tx_position(3) = antennaHeight;
l.tx_array = qd_arrayant('3gpp-3d', M_V, M_H, centerFrequency, 3, 0, antennaSpacing);

for n=1:M_V
    for nn=1:M_H
        indeces = (n-1)*M_H+nn;
        l.tx_array.element_position(1,indeces) =  (nn)*antennaSpacing*lambda  - lambda/4 - M_V/2*antennaSpacing*lambda;
        l.tx_array.element_position(2,indeces) = 0;
        l.tx_array.element_position(3,indeces) = (n)*antennaSpacing*lambda - lambda/4 - M_H/2*antennaSpacing*lambda + antennaHeight;
    end
end

% Users
l.no_rx = K;                                            % Number of users
l.rx_array = qd_arrayant( 'omni' );                     % Omnidirectional MT antenna

% Update Map
l.set_scenario('3GPP_3D_UMa_NLOS');

par.minDistance = minDistance;
par.maxDistance = maxDistance;
par.sectorAngleRad = sectorAngleRad;
par.bandwidth = bandwidth;
par.numSubcarriers = numSubcarriers;
par.subSampling = subSampling;
par.sequenceLength = sequenceLength;
par.s=s;

params = cell(1,numDrops);
for n=1:numDrops
    params{1,n} = par;
    params{1,n}.l = l.copy;
end


h = cell(1,numDrops);
for n=1:numDrops
    n
    h(1,n) = genChannelDrop(params{1,n});
end
H = cell2mat(h');
clear h
H_r = real(H);
H_i = imag(H);

clear H
hdf5write('./channel_sequences.hdf5', 'H_r', H_r, 'H_i', H_i)
