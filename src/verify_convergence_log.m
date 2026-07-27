function verify_convergence_log()
%VERIFY_CONVERGENCE_LOG Audit the released C&CG bound sequence.

package_root = fileparts(fileparts(mfilename('fullpath')));
log_file = fullfile(package_root, 'logs', ...
    'ccg_convergence_complete.csv');
data = readtable(log_file, 'TextType', 'string');

tolerance = 1e-5;
calculated_gap = max(data.UB - data.LB, 0);
calculated_relative_percent = 100 * calculated_gap ./ ...
    max(1, abs(data.UB));
gap_error = max(abs(calculated_gap - data.AbsoluteGap));
relative_error = max(abs(calculated_relative_percent - ...
    data.RelativeGapPercent));

assert(gap_error <= 1e-8, ...
    'Logged absolute gaps do not match max(UB-LB,0).');
assert(relative_error <= 1e-8, ...
    'Logged relative gaps do not match the stated definition.');
assert(all(data.AbsoluteGap(1:end-1) > tolerance), ...
    'The log would have stopped before the final iteration.');
assert(data.AbsoluteGap(end) <= tolerance, ...
    'The final iteration does not satisfy the stopping rule.');

raw_final_difference = data.UB(end) - data.LB(end);
fprintf('Iterations checked: %d\n', height(data));
fprintf('Maximum absolute-gap reconstruction error: %.3e\n', gap_error);
fprintf('Maximum relative-gap reconstruction error: %.3e\n', relative_error);
fprintf('Final raw UB-LB: %.16e\n', raw_final_difference);
fprintf('Final reported nonnegative gap: %.16e\n', calculated_gap(end));
fprintf('Stopping tolerance: %.1e\n', tolerance);
fprintf('Convergence decision verified.\n');
end
