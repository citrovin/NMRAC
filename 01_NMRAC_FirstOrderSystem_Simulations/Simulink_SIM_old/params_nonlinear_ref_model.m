%% Tabula Rasa and Config
clear; close all; clc;
global SAVE_DIR;
global BASE_DIR;

ct = clock;

SIMULATION = 1;
SAVE_PLOTS = 0;
SHOW_FIG = 'On';

BASE_DIR = './results/';
SAVE_DIR = strjoin([BASE_DIR, strjoin(string(ct(1:5)),'-')],'');

if SAVE_PLOTS==0
    if ~exist(SAVE_DIR, 'dir')
        mkdir(SAVE_DIR)
    end
end

%% Simulation Parameters
Tsim = 50;

% Choose Signal
SIGNAL_NR = 2;

% Choose the Reference Signal
% 1: Pulse
period = 15;
pulse_width = 0.5;
phase_delay = 0;

% 2: Sine
frequency=1;

% 3: Step input
amplitude = 1;
bias = 0;

%% Reference Model
tauM = 0.1;
K = 1;

am = 1/tauM;
bm = K/tauM;

IC_Pm = 0;

BETA_m = 2;
wm = 20;

%% System
% unstable if a<0
% stable if a>0
% a = -4;
a = am;
b = bm;

IC_P = IC_Pm;

%%
%{x = linspace(-10,10, 1000);


%sigma = @(w,e) 2*(1+exp(-BETA*w*e));

%sys_dyn = @(x,r) -a*x * b*simga(w, (r-x));
%ref_dy = @(xm,r) -am*x+bm*sigma(wm, (r-xm));

%y1 = sys_dyn(x,);
%}

%% Neural Network Controller with Sigmoid activation function
w = 0;

% Saturation
SAT_ACTIVE = 1;
BETA = BETA_m;
if SAT_ACTIVE == 1
    fprintf('Sigmoid saturates at: %d \n', (2/BETA));
else
    fprintf('No Saturation Active\n');
end


% Update params
WEIGHT_UPDATE = 1;

% Precision tollerance for the weight update
%epsilon = 1e-5;
epsilon = eps;

if WEIGHT_UPDATE==1
    fprintf('Learning Active\n');
else
    fprintf('No Learning Active\n')
end

%% Test Sigmoid
%x = linspace(-20,20,100);
%y = Sigmoid_SAT(x,SAT_ACTIVE);

%figure;
%plot(x,y);

%% Simulation
    if SIMULATION==1
    sim_out = sim('Simulation_Nonlinear_MRAC.slx');
    %sim_out = sim('Simulation_Nonlinear_MRAC_v2.slx');
    % sim_out = sim('v3.slx');
    
    %% Final Dynamics
    %dyn = @(x, u) -a.*+b.*u;
    %ref_dyn = @(x) -am.*x;
    
    %% Plots
    fig = figure;
    set(fig, 'Visible', SHOW_FIG);
    plot(sim_out.tout, sim_out.r.Data, sim_out.tout, sim_out.x.Data, sim_out.tout, sim_out.xm.Data);
    legend('r', 'x', 'x_m');
    title('Response');
    ylabel('Magnitude');
    xlabel('Time [s]');
    ylim([-10,10]);
    grid;
    if SAVE_PLOTS==1
        full_path = char(strcat(SAVE_DIR, '/', 'Nonlinear_MRAC_First-Response.png'));
        saveas(gcf, full_path);
    end
    
    fig = figure;
    set(fig, 'Visible', SHOW_FIG);
    p1 = subplot(3,1,1);
    plot(sim_out.tout, sim_out.e.Data, sim_out.tout, sim_out.ex.Data, sim_out.tout, sim_out.em.Data);
    legend('e', 'e_x', 'e_m');
    title('Errors');
    ylabel('Magnitude');
    xlabel('Time [s]');
    grid;
    
    
    p2 = subplot(3,1,2);
    plot(sim_out.tout, sim_out.u_sat.Data, sim_out.tout, sim_out.u.Data);
    legend('u_{sat}','u');
    title('Control Input');
    ylabel('[V]');
    xlabel('Time [s]');
    grid;
    
    
    p3 = subplot(3,1,3);
    if (am==a && bm==b)
        plot(sim_out.tout, sim_out.weights.Data, sim_out.tout, ones(size(sim_out.tout))*wm);
        legend('w', 'w_m');
    else
        plot(sim_out.tout, sim_out.weights.Data);
        legend('w');
    end
    title('Evolution of weights');
    ylabel('Magnitude');
    xlabel('Time [s]');
    grid;
    % Link the axes 
    linkaxes([p1, p2, p3], 'x');
    if SAVE_PLOTS==1
        full_path = char(strcat(SAVE_DIR, '/', 'Nonlinear_MRAC_First-Order_Error-weights-input.png'));
        saveas(gcf, full_path);
    end
    
    %% Lyapunov function
    
    sig = @(x) (2*(1-exp(-BETA.*x)))./(BETA*(1+exp(-BETA.*x)));
    sig_m = @(x) (2*(1-exp(-BETA_m.*x)))./(BETA_m*(1+exp(-BETA_m.*x)));
    
    V = @(e, ex, x, wm, w) e.^2 + ((a-am).*x+(bm.*sig_m(wm.*ex)-b.*sig(w.*ex))).^2;
    lyapunov_data = V(sim_out.e.Data, sim_out.ex.Data,sim_out.x.Data , wm, sim_out.weights.Data);

    fig = figure;
    set(fig, 'Visible', SHOW_FIG);
    %semilogy(ans.tout, lyapunov_data);
    plot(sim_out.tout, lyapunov_data, sim_out.tout, (sim_out.e.Data).^2, sim_out.tout, ((a-am).*sim_out.x.Data+(bm.*sig_m(wm.*sim_out.ex.Data)-b.*sig(sim_out.weights.Data.*sim_out.ex.Data))).^2);
    legend('V(e)', 'e^2', 'MC');
    title('Lyapunov function');
    xlabel('Time [s]');
    ylim([0,50]);
    grid;
    if SAVE_PLOTS==1
        full_path = char(strcat(SAVE_DIR, '/', 'Nonlinear_MRAC_First-Order_Lyapunov_Function.png'));
        saveas(gcf, full_path);
    end


end
%{
%% Functions

f = @myIntegrand;
x = linspace(-10,10,1000);

y = Sigmoid_SAT(x, SAT);

figure;
plot(x, y);
xlabel('x');
ylabel('y');
title(['Sigmoid SAT function with SAT = ', num2str(2/SAT)]);
grid on;


w_dot_term1 = @(ex, e, dsigma_m, dsigma_x, x_dot, xm_dot, w, r_dot) (1/(ex*b*dsigma_x))*(e + (a-am)*x_dot + bm*dsigma_m*wm*(r_dot-xm_dot));

w_dot_term2=@(w,ex,r_dot, x_dot) -(w/ex)*(r_dot-x_dot);

dsigma_x = @(wex)  (4*exp(-BETA*wex))/((1+exp(-BETA*wex))^2);
dsigma_m = @(wem)  (4*exp(-BETA_m*wem))/((1+exp(-BETA_m*wem))^2);
%}