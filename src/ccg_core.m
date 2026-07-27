function result = ccg_core(solve_master, solve_subproblem, options)
%CCG_CORE Parameter-independent column-and-constraint generation loop.
%
% solve_master(scenarios) returns fields x, lower_bound, first_stage_cost.
% solve_subproblem(x) returns fields worst_recourse_cost, worst_scenario.
% Required options are max_iterations and absolute_tolerance.

validateattributes(solve_master, {'function_handle'}, {'scalar'});
validateattributes(solve_subproblem, {'function_handle'}, {'scalar'});
required = {'max_iterations', 'absolute_tolerance'};
for i = 1:numel(required)
    if ~isfield(options, required{i})
        error('Missing options.%s.', required{i});
    end
end

if isfield(options, 'initial_scenarios')
    scenarios = options.initial_scenarios;
else
    scenarios = {};
end
if isfield(options, 'same_scenario')
    same_scenario = options.same_scenario;
else
    same_scenario = @isequal;
end

max_iterations = options.max_iterations;
tolerance = options.absolute_tolerance;
upper_bound_best = inf;
converged = false;
stalled = false;
history = repmat(struct('Iteration', nan, 'LB', nan, 'UB', nan, ...
    'AbsoluteGap', nan, 'RelativeGap', nan, 'FirstStageCost', nan, ...
    'WorstRecourseCost', nan, 'ScenarioCount', nan), max_iterations, 1);

for k = 1:max_iterations
    master = solve_master(scenarios);
    require_fields(master, {'x', 'lower_bound', 'first_stage_cost'}, ...
        'master result');
    subproblem = solve_subproblem(master.x);
    require_fields(subproblem, ...
        {'worst_recourse_cost', 'worst_scenario'}, 'subproblem result');

    lower_bound = master.lower_bound;
    upper_bound_candidate = master.first_stage_cost + ...
        subproblem.worst_recourse_cost;
    upper_bound_best = min(upper_bound_best, upper_bound_candidate);
    absolute_gap = max(upper_bound_best - lower_bound, 0);
    relative_gap = absolute_gap / max(1, abs(upper_bound_best));

    history(k).Iteration = k;
    history(k).LB = lower_bound;
    history(k).UB = upper_bound_best;
    history(k).AbsoluteGap = absolute_gap;
    history(k).RelativeGap = relative_gap;
    history(k).FirstStageCost = master.first_stage_cost;
    history(k).WorstRecourseCost = subproblem.worst_recourse_cost;
    history(k).ScenarioCount = numel(scenarios);

    if absolute_gap <= tolerance
        converged = true;
        break;
    end

    duplicate = false;
    for s = 1:numel(scenarios)
        if same_scenario(scenarios{s}, subproblem.worst_scenario)
            duplicate = true;
            break;
        end
    end
    if duplicate
        stalled = true;
        break;
    end
    scenarios{end + 1} = subproblem.worst_scenario; %#ok<AGROW>
end

history = history(1:k);
result.history = struct2table(history);
result.scenarios = scenarios;
result.iterations = k;
result.converged = converged;
result.stalled = stalled;
result.final_lower_bound = history(end).LB;
result.final_upper_bound = history(end).UB;
result.final_gap = history(end).AbsoluteGap;
end

function require_fields(value, names, description)
for i = 1:numel(names)
    if ~isfield(value, names{i})
        error('%s is missing field %s.', description, names{i});
    end
end
end
