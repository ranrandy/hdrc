import matplotlib.pyplot as plt

# Define the data points for the graph
x = [0, 0.1042*150, 76.60]  # X-axis data points
y = [0, 0.0300*150, 22.06]  # Y-axis data points


# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, color="b")  # Plot with a marker at each data point
plt.plot([76.60, 100], [22.06, 22.06], color="b")  # Plot with a marker at each data point

# plt.xscale('log')
# plt.yscale('log')

# Set the limit for the y-axis
plt.ylim(0, 24.00)
plt.xlim(0, 100)

# Draw a horizontal dashed line at y=0.2300
# plt.axhline(y=22.06, color='black', linestyle='--')

plt.plot([0, 76.60], [22.06, 22.06], "k--")
plt.plot([76.60, 76.60], [0, 22.06], "k--")

plt.plot([0, 0.1042*150], [0.0300*150, 0.0300*150], "g--")
plt.plot([0.1042*150, 0.1042*150], [0, 0.0300*150], "g--", label="Poisson Iteration")

plt.plot([0, 0.0417*150], [0.0120*150, 0.0120*150], "r--")
plt.plot([0.0417*150, 0.0417*150], [0, 0.0120*150], "r--", label="Calculate Residual/Error")

plt.legend(loc='center right', fontsize="18")

# Annotate the horizontal line with '22.06'
plt.text(0, 21.56, '22.06', verticalalignment='bottom', horizontalalignment='right')

# Set the axes labels
plt.xlabel('Arithmetic Intensity (FLOPS/BYTES)')
plt.ylabel('Performance (TF/s)')

# Add text for the (0,0) point
plt.text(-5, -0.8, '0')

# Add text for the x-axis data points
plt.text(0.1042*150, -0.8, '0.1042')
plt.text(0.0417*150, -0.8, '0.0417')
plt.text(76.60, -0.8, '76.60')

# Add text for the y-axis data point
plt.text(0, 0.0300*150, '0.0300', verticalalignment='bottom', horizontalalignment='right')
plt.text(0, 0.0120*150, '0.0120', verticalalignment='bottom', horizontalalignment='right')

# Save the figure
# plt.savefig('/mnt/data/replicated_figure.png')

plt.xticks([])
plt.yticks([])
plt.title('Roof Line Model for Iterative Poisson Solvers')
# Show the figure
plt.savefig("roof_line_model.png", dpi=500)
plt.show()
