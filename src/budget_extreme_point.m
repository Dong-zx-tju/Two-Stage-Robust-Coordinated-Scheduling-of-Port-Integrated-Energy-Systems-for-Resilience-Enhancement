function [delta_plus, delta_minus, constraints] = ...
    budget_extreme_point(component_count, uncertainty_budget)
%BUDGET_EXTREME_POINT Exact extreme-point representation for integer budgets.
%
% The original uncertainty set permits continuous normalized deviations in
% [0,1]. For an integer budget, its linear adversarial optimum can be selected
% at an integral extreme point. These binary variables represent that extreme
% point; they do not redefine the original uncertainty set.

validateattributes(component_count, {'numeric'}, ...
    {'scalar', 'integer', 'positive'});
validateattributes(uncertainty_budget, {'numeric'}, ...
    {'scalar', 'integer', 'nonnegative', '<=', component_count});

delta_plus = binvar(component_count, 1);
delta_minus = binvar(component_count, 1);

constraints = [ ...
    delta_plus + delta_minus <= 1, ...
    sum(delta_plus + delta_minus) <= uncertainty_budget];
end
