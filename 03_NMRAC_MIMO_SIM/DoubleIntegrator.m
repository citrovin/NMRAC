n = 2; 
m = 1;
% Define the system matrices
% Internal dynamics (pure double integrator)
A = zeros(n,n);
A(1,2) = 1;
A(2,2) = 0;
% Am = [-1 0.5; -0.3 0.7];

% Input dynamics
B = ones(n,m);
B(1,1) = 1;
B(2,2) = 1;

% Define the weight matrices
Q = eye(n,n);  % Define an n x n positive semi-definite matrix
R = eye(m,m);  % Define an m x m positive definite matrix

% Compute the LQR gain matrix L
[L, P, ~] = lqr(A, B, Q, R);

% Display the gain matrix
disp('Optimal gain matrix L:');
disp(L);

% Check the closed-loop system for stability
Acl = A - B * L;
eig_values = eig(Acl); % Compute eigenvalues of the closed-loop system matrix
disp('Eigenvalues of the closed-loop system matrix A - Bm*L:');
disp(eig_values);

disp('P Matrix:');
disp(P);

% Verify stability
if all(real(eig_values) < 0)
    disp('The closed-loop system is stable.');
else
    disp('The closed-loop system is NOT stable.');
end