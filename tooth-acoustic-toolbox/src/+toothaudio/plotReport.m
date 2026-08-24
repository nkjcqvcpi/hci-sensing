function h = plotReport(report)
h = figure('Color', 'white');
plot(report.thresholds, report.far, 'LineWidth', 1.5);
hold on
plot(report.thresholds, report.frr, 'LineWidth', 1.5);
xline(report.eerThreshold, ':', 'LineWidth', 1.2);
grid on
xlabel('Acceptance threshold');
ylabel('Rate');
legend('FAR', 'FRR', 'selected threshold', 'Location', 'best');
end
