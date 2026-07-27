function plot_convergence_log()
%PLOT_CONVERGENCE_LOG Regenerate the C&CG convergence figure from the log.

package_root = fileparts(fileparts(mfilename('fullpath')));
data = readtable(fullfile(package_root, 'logs', ...
    'ccg_convergence_complete.csv'), 'TextType', 'string');
output_dir = fullfile(package_root, 'figures', 'reproduced');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

iteration = data.Iteration;
fig = figure('Color', 'w', 'Units', 'centimeters', ...
    'Position', [3, 3, 10.5, 7.2]);

ax1 = axes(fig, 'Position', [0.16, 0.50, 0.78, 0.39]);
hold(ax1, 'on');
plot(ax1, iteration, data.LB ./ 1e4, '-o', ...
    'Color', [0.1059 0.3686 0.6510], 'MarkerFaceColor', 'w', ...
    'LineWidth', 1.45, 'MarkerSize', 4.5);
plot(ax1, iteration, data.UB ./ 1e4, '-s', ...
    'Color', [0.8510 0.3725 0.0078], 'MarkerFaceColor', 'w', ...
    'LineWidth', 1.45, 'MarkerSize', 4.3);
ylabel(ax1, 'Objective (10^4 yuan)');
xlim(ax1, [iteration(1)-0.15, iteration(end)+0.15]);
xticks(ax1, iteration);
ax1.XTickLabel = [];
legend(ax1, {'LB', 'UB'}, 'Location', 'northeast', 'Box', 'off');
style_axis(ax1);

ax2 = axes(fig, 'Position', [0.16, 0.14, 0.78, 0.25]);
hold(ax2, 'on');
area(ax2, iteration, data.RelativeGapPercent, ...
    'FaceColor', [0.88 0.88 0.88], 'EdgeColor', 'none', ...
    'FaceAlpha', 0.65);
plot(ax2, iteration, data.RelativeGapPercent, '-^', ...
    'Color', [0.22 0.22 0.22], 'MarkerFaceColor', 'w', ...
    'LineWidth', 1.20, 'MarkerSize', 4.1);
ylabel(ax2, 'Gap (%)');
xlabel(ax2, 'C&CG iteration');
xlim(ax2, [iteration(1)-0.15, iteration(end)+0.15]);
xticks(ax2, iteration);
style_axis(ax2);

exportgraphics(fig, fullfile(output_dir, ...
    'ccg_convergence_reproduced.pdf'), 'ContentType', 'vector');
exportgraphics(fig, fullfile(output_dir, ...
    'ccg_convergence_reproduced.png'), 'Resolution', 600);
close(fig);
fprintf('Regenerated convergence figures in %s\n', output_dir);
end

function style_axis(ax)
ax.Box = 'on';
ax.TickDir = 'in';
ax.FontName = 'Times New Roman';
ax.FontSize = 9;
ax.LineWidth = 0.75;
ax.Layer = 'top';
ax.XGrid = 'off';
ax.YGrid = 'off';
end
