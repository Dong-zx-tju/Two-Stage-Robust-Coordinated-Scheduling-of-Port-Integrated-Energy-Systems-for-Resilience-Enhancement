%% Two-stage robust scheduling solved by column-and-constraint generation
clc;
clear;
close all;
tic;

%% 1. C&CG and uncertainty parameters
% Fault scenario switch:
% true  - evaluate all candidate grid-outage windows and solve by C&CG;
% false - disable the fault and solve the nominal day-ahead model only.
fault_enabled = true;

it_max = 20;
tolerance = 0.00001;  % Absolute objective gap

Gamma_pv = 6;
Gamma_w = 6;
Gamma_ves = 6;

alpha_pv = 0.2;
alpha_w = 0.2;
alpha_ves = 0.2;

% Candidate external-grid outage scenarios: [start hour, duration].
% The C&CG subproblem evaluates every candidate window and returns the
% joint worst case of the outage window and source-load uncertainty.
fault_scenarios = [
     8, 4;
     8, 6;
     9, 4;
     9, 6;
    10, 4;
    10, 6;
    11, 4;
    11, 6
];
N_fault = size(fault_scenarios, 1);

%% 2. Nominal uncertainty vector
Ppv_val = [0 0 0 0 0 0 94 289 581 850 1090 1177 1263 960 886 ...
           910 625 431 226 0 0 0 0 0];
Pw_val = [66.9 68.2 71.9 72 78.8 94.8 114.3 145.1 155.5 142.1 115.9 127.1 ...
          141.8 145.6 145.3 150 206.9 225.5 236.1 210.8 198.6 177.9 147.2 58.7];

% Keep the same VES scale used inside xiaduiouMP and xiaduiouSP.
Pves_val = 0.2 * [300 280 260 250 260 300 400 500 600 650 ...
                  700 720 700 680 650 600 550 500 450 400 ...
                  380 350 320 300];

u_pre = [Ppv_val(:); Pw_val(:); Pves_val(:)];

%% 3. No-fault nominal operation
if ~fault_enabled
    fprintf('Fault disabled: solve the nominal day-ahead scheduling model.\n');
    [x, C_normal] = xiaduiouMPstar(u_pre);
    runtime = toc;

    fprintf('\n========== Nominal operation finished ==========\n');
    fprintf('Nominal operating cost: %.4f\n', C_normal);
    fprintf('First-stage vector length: %d\n', numel(x));
    fprintf('Runtime: %.4f s\n', runtime);
    return;
end

%% 4. C&CG initialization
scenario_set = zeros(72, 0);
scenario_fault_ids = zeros(1, 0);
UB_best = inf;
converged = false;
stalled = false;

LB_history = nan(1, it_max);
UB_history = nan(1, it_max);
Gap_history = nan(1, it_max);
C_normal_history = nan(1, it_max);
Eta_history = nan(1, it_max);
Q_worst_history = nan(1, it_max);
Scenario_count_history = nan(1, it_max);
Worst_fault_id_history = nan(1, it_max);
Pre_fault_state_history = nan(4, N_fault, it_max);

% Layout of the complete first-stage vector returned by xiaduiouMP:
% [1128 binary modes; 7467 day-ahead continuous variables;
%  4 pre-fault storage states for each candidate outage window].
N_X_BINARY = 1128;
N_Y_DA = 7467;
N_S_PRE = 4 * N_fault;
IDX_S_PRE = N_X_BINARY + N_Y_DA + (1:N_S_PRE);

fprintf('Start standard C&CG iteration.\n');
fprintf('Gamma = [%d, %d, %d], alpha = [%.2f, %.2f, %.2f]\n\n', ...
        Gamma_pv, Gamma_w, Gamma_ves, alpha_pv, alpha_w, alpha_ves);
fprintf('Candidate outage windows [start, duration]:\n');
disp(fault_scenarios);

for it = 1:it_max
    fprintf('C&CG iteration %d\n', it);

    % The master retains every previously generated worst-case scenario.
    [x, LB, C_normal, eta_value] = xiaduiouMP( ...
        scenario_set, scenario_fault_ids, fault_scenarios, u_pre);

    % Evaluate all candidate outage windows. Each SP independently finds
    % the worst source-load realization under its own outage window.
    Q_by_fault = nan(N_fault, 1);
    u_by_fault = cell(N_fault, 1);
    B_by_fault = cell(N_fault, 1);
    for fault_id = 1:N_fault
        [u_by_fault{fault_id}, Q_by_fault(fault_id), B_by_fault{fault_id}] = ...
            xiaduiouSP( ...
                x, Gamma_pv, Gamma_w, Gamma_ves, ...
                alpha_pv, alpha_w, alpha_ves, ...
                fault_scenarios, fault_id, false);
        fault_start = fault_scenarios(fault_id, 1);
        fault_end = min(24, fault_start + fault_scenarios(fault_id, 2) - 1);
        fprintf('  Fault %d (%d--%d h): Q = %.4f\n', ...
                fault_id, fault_start, fault_end, Q_by_fault(fault_id));
    end

    [Q_worst, worst_fault_id] = max(Q_by_fault);
    u_worst = u_by_fault{worst_fault_id};
    B = B_by_fault{worst_fault_id}; %#ok<NASGU>
    S_pre_all = reshape(x(IDX_S_PRE), 4, N_fault);

    % LB and UB now use the same objective definition.
    UB_candidate = C_normal + Q_worst;
    UB_best = min(UB_best, UB_candidate);
    gap = max(UB_best - LB, 0);
    relative_gap = gap / max(1, abs(UB_best));

    LB_history(it) = LB;
    UB_history(it) = UB_best;
    Gap_history(it) = gap;
    C_normal_history(it) = C_normal;
    Eta_history(it) = eta_value;
    Q_worst_history(it) = Q_worst;
    Scenario_count_history(it) = size(scenario_set, 2);
    Worst_fault_id_history(it) = worst_fault_id;
    Pre_fault_state_history(:, :, it) = S_pre_all;

    fprintf('  C_normal    = %.4f\n', C_normal);
    fprintf('  eta          = %.4f\n', eta_value);
    fprintf('  Q_worst      = %.4f\n', Q_worst);
    fprintf('  LB           = %.4f\n', LB);
    fprintf('  UB_candidate = %.4f\n', UB_candidate);
    fprintf('  UB_best      = %.4f\n', UB_best);
    fprintf('  Gap          = %.4f (relative %.6e)\n\n', gap, relative_gap);
    fault_start = fault_scenarios(worst_fault_id, 1);
    fault_end = min(24, fault_start + fault_scenarios(worst_fault_id, 2) - 1);
    fprintf('  Worst fault  = %d (%d--%d h)\n', ...
            worst_fault_id, fault_start, fault_end);
    fprintf('  Pre-fault storage [EES, GES, TES, HES] = [%.2f, %.2f, %.2f, %.2f]\n\n', ...
            S_pre_all(1,worst_fault_id), S_pre_all(2,worst_fault_id), ...
            S_pre_all(3,worst_fault_id), S_pre_all(4,worst_fault_id));

    if gap <= tolerance
        converged = true;
        fprintf('C&CG converged at iteration %d.\n', it);
        break;
    end

    % Do not add duplicate scenarios. A duplicate with a positive gap means
    % that the master recourse model and subproblem are not equivalent.
    if isempty(scenario_set)
        scenario_set(:, end + 1) = u_worst(:);
        scenario_fault_ids(end + 1) = worst_fault_id;
    else
        same_fault_columns = find(scenario_fault_ids == worst_fault_id);
        is_duplicate = false;
        if ~isempty(same_fault_columns)
            scenario_distance = max( ...
                abs(scenario_set(:,same_fault_columns) - u_worst(:)), [], 1);
            is_duplicate = min(scenario_distance) <= 1e-6;
        end
        if is_duplicate
            stalled = true;
            warning(['The subproblem returned an existing joint uncertainty/fault ', ...
                     'scenario while the gap is still positive. Check MP/SP ', ...
                     'recourse consistency.']);
            break;
        end
        scenario_set(:, end + 1) = u_worst(:);
        scenario_fault_ids(end + 1) = worst_fault_id;
    end
end

%% 5. Results
actual_it = it;
runtime = toc;

LB_history = LB_history(1:actual_it);
UB_history = UB_history(1:actual_it);
Gap_history = Gap_history(1:actual_it);
C_normal_history = C_normal_history(1:actual_it);
Eta_history = Eta_history(1:actual_it);
Q_worst_history = Q_worst_history(1:actual_it);
Scenario_count_history = Scenario_count_history(1:actual_it);
Worst_fault_id_history = Worst_fault_id_history(1:actual_it);
Pre_fault_state_history = Pre_fault_state_history(:, :, 1:actual_it);

fprintf('\n========== C&CG finished ==========\n');
fprintf('Iterations: %d\n', actual_it);
fprintf('Final LB: %.4f\n', LB_history(end));
fprintf('Final UB: %.4f\n', UB_history(end));
fprintf('Final gap: %.4f\n', Gap_history(end));
fprintf('Converged: %d\n', converged);
fprintf('Stalled: %d\n', stalled);
fprintf('Accumulated scenarios: %d\n', size(scenario_set, 2));
fprintf('Accumulated fault IDs: %s\n', mat2str(scenario_fault_ids));
fprintf('Runtime: %.4f s\n', runtime);
final_fault_id = Worst_fault_id_history(end);
final_fault_start = fault_scenarios(final_fault_id, 1);
final_fault_end = min(24, final_fault_start + fault_scenarios(final_fault_id, 2) - 1);
fprintf('Final worst fault: %d (%d--%d h)\n', ...
        final_fault_id, final_fault_start, final_fault_end);
fprintf('Final pre-fault storage [EES, GES, TES, HES]: [%.2f, %.2f, %.2f, %.2f]\n', ...
        Pre_fault_state_history(1,final_fault_id,end), ...
        Pre_fault_state_history(2,final_fault_id,end), ...
        Pre_fault_state_history(3,final_fault_id,end), ...
        Pre_fault_state_history(4,final_fault_id,end));
