%% Tabula Rasa
clear all; clc;

%% Set default for plots
set(groot, 'defaultLineLineWidth', 1.5);
set(groot, 'defaultAxesXGrid', 'on', 'defaultAxesYGrid', 'on');

%% Simulation parameters

Ts = 0.001;
Tsim = 100;

% Define the time vector
t_vec = 0:Ts:Tsim;
num_steps = length(t_vec);

TOLLERANCE = eps;

n = 2;
m = 2;

%% Init reference Signal
input_signal = 2;

r = zeros(2, num_steps);
r_dot = zeros(2, num_steps);

if input_signal == 1
    % Constant input signal
    r(1,:) = 0;
    r(2,:) = 0;

    r_dot(1,:) = 0;
    r_dot(2,:) = 0;

elseif input_signal == 2
    % Sine input signal
    r(1,:) = 0.5*sin(t_vec);
    r(2,:) = 0.5*cos(t_vec);

    r_dot(1,:) = 0.5*cos(t_vec);
    r_dot(2,:) = -0.5*sin(t_vec);

end


%% Init Model Reference
% Internal dynamics (pure double integrator)
Am = zeros(n,n);
Am(1,2) = 1;
Am(2,2) = 0;

% Input dynamics
Bm = zeros(n,m);
Bm(2,1) = 1;
Bm(2,2) = -1;

% Initial conditions Model Ref
IC_M = [-pi ; 0]./2;

% Save the values
xm  = zeros(n, num_steps);
xm(:,1) = IC_M;

em  = zeros(n, num_steps);
em(:,1) = r(:,1) - xm(:,1);

Q = 10*eye(n);
R = eye(m);

[L, P, ~]  = lqr(Am, Bm, Q, R);
[E, D] = eig(P);

% Feedforward gain matrix
theta_m = L;

SAT_VAL = 5;
a = 2/SAT_VAL; % Sigmoid saturates at 2/a.
fprintf("Saturation value: %d\n", SAT_VAL);

% Choose the activation function of the model reference
% 1     Sigmoid (Note: stability proof is non-trivial)
% -1    Linear activation (Note: Use Lyapunov theory to find P)
ACTIVATION_MODEL_REF = 1;
if ACTIVATION_MODEL_REF==-1
    am = -1;
    fprintf("Model Reference: Linear activation function\n");
else
    am = a;
    fprintf("Model Reference sigmoid saturation value: %d\n", SAT_VAL);
end

%% Init System
% Internal dynamics
A = Am;

% Input dynamics
B = Bm;

% Feedforward gain matrix
theta_init = 0.5.*theta_m;
theta = zeros(size(theta_m, 1) , size(theta_m, 2), num_steps);
theta(:,:,1) = theta_init;

% Learning Rate
Lambda = 0.2.*[1 0; 0 1];

% Initial conditions System
IC_P = [-pi ; 0]./2;

% Save the values
x  = zeros(n, num_steps);
x(:,1) = IC_P;

ex  = zeros(n, num_steps);
ex(:,1) = r(:,1) - x(:,1);

%% Init Lyapunov Function
% Init Error
e = zeros(n, num_steps);
e(:,1) = xm(:,1) - x(:,1);

alpha = zeros(n, num_steps);
alpha(:,1) = compute_alpha(x(:,1), ex(:,1), theta(:,:,1), theta_m, Am, A, Bm, B, a, am);

e_tilde = zeros(n, num_steps);
u_tilde = zeros(n, num_steps);

% Lyapunov function
V_history = zeros(1 , num_steps);
V_e_history = zeros(1 , num_steps);
V_alpha_history = zeros(1 , num_steps);

[V, V_e, V_alpha] = lyapunov_function(e(:,1), alpha(:,1), P, Lambda);
V_history(1,1) = V;
V_e_history(1,1) = V_e;
V_alpha_history(1,1) = V_alpha;

% Timer derivative Lyapunov function
dV_history = zeros(1 , num_steps-1);
dV_Q_history = zeros(1 , num_steps-1);
dV_gamma_history = zeros(1 , num_steps-1);
dV_alpha_history = zeros(1 , num_steps-1);


 %% Simulation

fprintf("------ SIMUATION STARTED ------\n");
for t = 2:num_steps
    % Step System
    u = sigmoid(theta(:,:,t-1)*ex(:,t-1), a);

    x_dot = A*x(:,t-1) + B*u;
    x(:,t) = x(:,t-1) + Ts*x_dot;

    ex(:,t) = r(:,t) - x(:,t);
    ex_dot = r_dot(:,t) - x_dot;
    
    % Step Model Ref
    um = sigmoid(theta_m*em(:,t-1), am);

    xm_dot = Am*xm(:,t-1) + Bm*um;
    xm(:,t) = xm(:,t-1) + Ts*xm_dot;
    
    em(:,t) = r(:,t) - xm(:,t);

    % Compute error for update
    e(:,t) = xm(:,t) - x(:,t);
    e_dot = xm_dot - x_dot;

    % Update params
    theta_dot = update_params(ex(:,t-1), e(:,t-1), x_dot,  r_dot(:,t-1), theta(:,:,t-1), ...
    theta_m, Lambda, P, Bm, B, Am, A, a, am);
    theta(:,:,t) = theta(:,:,t-1) + Ts.*theta_dot;
    
    % Compute Lyapunov Function
    alpha(:,t) = compute_alpha(x(:, t-1), ex(:, t-1), theta(:, :, t-1), theta_m, Am, A, Bm, B, a , am);
    
    [V_history(:,t-1),  V_e_history(:,t-1), V_alpha_history(:,t-1)] = lyapunov_function(e(:,t-1), alpha(:,t-1), P, Lambda);

    [dV_history(:,t-1), dV_Q_history(:,t-1), dV_gamma_history(:,t-1), dV_alpha_history(:,t-1), alpha_dot] = lyapunov_derivative(x_dot, Am, A, ...
                                                            Bm, theta_m, ex(:,t-1), am, ex_dot, ...
                                                            B, theta(:,:,t-1), a, theta_dot, ...
                                                            e(:,t-1), P, Lambda, alpha(:,t-1), ...
                                                            em(:,t-1));
    
end

fprintf("Min of Lyapunov: %d \nMaximum of lyapunov derivative: %d \n", min(V_history), max(dV_history));
fprintf("Parameters of the Model refernce: \n");
disp(theta_m);
fprintf("Final Parameters of the NNC: \n");
disp(theta(:,:,end));

disp("Eigenvalues of closed-loop system (if no activation function):");
disp(eig(A-B*theta(:,:,end)));

%% Plot response
% x_1 component
figure(2);
subplot(2,1,1);
plot(t_vec, r(1,:),...
     t_vec, x(1,:),...
     t_vec, xm(1,:));
legend("r", "x", "x_m");
xlabel("Time [s]");
ylabel("x_1");
title("Respnose x_1 Component");

% x_2 component
subplot(2,1,2);
plot(t_vec, r(2,:),...
     t_vec, x(2,:),...
     t_vec, xm(2,:));
legend("r", "x", "x_m");
ylabel("x_2");
xlabel("Time [s]");
title("Respnose x_2 Component");
sgtitle("Response");

%% Plot theta
figure(3);
subplot(2,2,1);
plot(t_vec, squeeze(theta(1,1,:)), ...
    t_vec, theta_m(1,1)*ones(1, num_steps));
xlabel("Time [s]");
ylabel("Magnitude");
legend("\theta_{1,1}", "\theta_{m; 1,1}");
title("\theta_{1,1}")

subplot(2,2,2);
plot(t_vec, squeeze(theta(1,2,:)), ...
    t_vec, theta_m(1,2)*ones(1, num_steps));
xlabel("Time [s]");
ylabel("Magnitude");
legend("\theta_{1,2}", "\theta_{m; 1,2}");
title("\theta_{1,2}")

subplot(2,2,3);
plot(t_vec, squeeze(theta(2,1,:)), ...
    t_vec, theta_m(2,1)*ones(1, num_steps) ...
    );
xlabel("Time [s]");
ylabel("Magnitude");
legend("\theta_{2,1}", "\theta_{m; 2,1}");
title("\theta_{2,1}")

subplot(2,2,4);
plot(t_vec, squeeze(theta(2,2,:)), ...
    t_vec, theta_m(2,2)*ones(1, num_steps) ...
    );
xlabel("Time [s]");
ylabel("Magnitude");
legend("\theta_{2,2}", "\theta_{m; 2,2}");
title("\theta_{2,2}")

sgtitle("Parameters over time");

%% Plot Lyapunov Function
figure(4);
subplot(2,1,1);
plot(t_vec, V_history, ...
    t_vec, V_e_history, ...
    t_vec, V_alpha_history);
legend("V", "e^TPe", "\alpha^T\Lambda\alpha");
%legend("V");
xlabel("Time [s]");
ylabel("Magnitude");
title("Lyapunov Function");

subplot(2,1,2);
plot(t_vec(:,2:num_steps), dV_history, ...
    t_vec(:,2:num_steps), dV_gamma_history, ...
    t_vec(:,2:num_steps), dV_Q_history, ...
    t_vec(:,2:num_steps), dV_alpha_history ...
    );

legend("dV", "\gamma Term", "Q Term",  "\alpha Term");
xlabel("Time [s]");
ylabel("Magnitude");
title("Lyapunov Function Time Derivative");

sgtitle("Lyapunov Function");

%% Plot errors
figure(5);
subplot(2,1,1);
plot(t_vec, e(1,:), ...
    t_vec, ex(1,:), ...
    t_vec, em(1,:));
xlabel("Time [s]");
ylabel("Magnitude");
legend("e", "e_x", "e_m");
title("Errors x_1 component");

subplot(2,1,2);
plot(t_vec, e(2,:), ...
    t_vec, ex(2,:), ...
    t_vec, em(2,:));
legend("e", "e_x", "e_m");
xlabel("Time [s]");
ylabel("Magnitude");
title("Errors x_2 component");

sgtitle("Errors");

%% Functions
function theta_dot = update_params(ex, ...
    e, x_dot, r_dot, theta, theta_m, Lambda, P, Bm, B, Am, A, a, am)
    % This function defines how the NNC updates its parameters.
    if norm(ex)>eps
        ex_dot = r_dot - x_dot;
        
        Sigma = pinv(B*dsigmoid(theta*ex, a));

        xsi = Lambda*P*e ...
            + (Am-A)*x_dot ...
            + Bm*dsigmoid(theta_m*ex, am)*theta_m*ex_dot...
            - B*dsigmoid(theta*ex, a)*theta*ex_dot;
        
        %theta_dot = Sigma*xsi*pinv(ex);
        theta_dot = (ex'\(Sigma*xsi)')';
    else 
        theta_dot = zeros(2,2);
    end
end

function [V, V_e, V_alpha] = lyapunov_function(e, alpha, P, Lambda)
    % This function computed the value of the Lyapunov function of the form
    % V(e alpha) = e'Pe + alpha'Lambda alpha.
    % It returns the the value of V , as well its components V_e and V_alpha

    V_e = e'*P*e;
    V_alpha = alpha'*inv(Lambda)*alpha;

    V = V_e + V_alpha;

end

function [dV, dV_term1, dV_term2, dV_term3 ,alpha_dot] = lyapunov_derivative( ...
                             x_dot, Am, A, ...
                             Bm, theta_m, ex, am, ex_dot, ...
                             B, theta, a, theta_dot, ...
                             e, P, Lambda, alpha, ...
                             em)
    
    
    % This is the real Lyapunov function 
    % ex_dot = r_dot - x_dot;
    alpha_dot = (Am-A)*x_dot ...
            + Bm*dsigmoid(theta_m*ex, am)*theta_m*ex_dot ...
            - B*dsigmoid(theta*ex, a)*(theta_dot*ex + theta*ex_dot);

    dV_term1 = 2.*e'*P*Am*e;
    gamma = Bm*(sigmoid(theta_m*em, am) - sigmoid(theta_m*ex, am));
    dV_term2 = 2.*e'*P*gamma;

    dV_term3 = 2.*alpha'*(P*e + inv(Lambda)*alpha_dot);

    dV = dV_term1 + dV_term2 +  dV_term3;

end

function u = sigmoid(x, a)
    % This function defines a smooth saturation, modeled by a sigmoid
    % function that saturates at the value 2/a.
    % If a = -1, then the activation function is linear and the matrix x is
    % returned
    if a==-1
        u = x;
    else
        u = 2.*(1-exp(-a.*x))./(a.*(1+exp(-a.*x)));
    end

end

function du = dsigmoid(x, a)
    % This function returns the derivative of the sigmoid function with
    % respect to x, evaluated at x.
    if a==-1
        du = eye(size(x,1));
        % TODO figure out the dimensions of du
    else
        du = diag(4.*exp(-a.*x)./(1+exp(-a.*x)).^2);
    end
end

function alpha = compute_alpha(x, ex, theta, theta_m, Am, A, Bm, B, a, am)
    alpha = (Am-A)*x ...
            + Bm*sigmoid(theta_m*ex, am) ...
            - B*sigmoid(theta*ex, a);
end
