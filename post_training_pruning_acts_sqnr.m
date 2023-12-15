% Copyright (c) 2023 Qualcomm Technologies, Inc.
% All Rights Reserved.
load matlab_data/0000_0000.mat

n_bits = 4;
n=size(P,1);
l = -min(x0);
u = max(x0);
density = n_bits/16.0;
% unpruned weights tend to have larger magnitude to compensate
% the pruned weights, so this constant ensures that that the domain
% is not limited. The value of this constant was found empirically
% such that further increasing it does not change the final objective
c = 2.0;

x0 = x0';
obj_init = 0.5 * x0' * P * x0 + q * x0 + r;

% find the global solution with arbitrary mask value
cvx_solver mosek
cvx_begin
    variable mask(n,1) binary
    variable x(n,1)
    minimize 0.5*quad_form(x,P) + q*x + r 
    - c * mask.* l <= x <= c * mask.* u;
    sum(mask) <= n*density;
cvx_end

obj_global_sol = cvx_optval;

norm = r;
sqnr_global_sol = -10.0*log10(obj_global_sol/norm);
x0_sorted = sort(abs(x0));
nnz = round(density * n);
mask_mp = abs(x0) <= x0_sorted(nnz);

% magnitude pruning (fixed sparsity mask)
cvx_begin
    variable x_mp(n,1)
    minimize 0.5*quad_form(x_mp,P) + q*x_mp + r 
    -c * mask_mp.* l <= x_mp <= c * mask_mp.* u;
cvx_end

obj_mag_pruning = cvx_optval;
sqnr_mag_pruning = -10.0*log10(obj_mag_pruning/norm);

fprintf(['SQNR values for pruning.\n The global solution : %.2f dB. Magnitude pruning baseline : %.2f dB\n'], sqnr_global_sol, sqnr_mag_pruning)
