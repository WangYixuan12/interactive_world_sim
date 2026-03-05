import time
from collections import deque
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


class RealTimePlotter:
    """Real-time plotter with a moving window."""

    def __init__(
        self,
        window_size: int = 10,
        title: str = "Real-Time Plot",
        y_min: float = -1.0,
        y_max: float = 1.0,
        num_lines: int = 1,
        legends: Optional[list[str]] = None,
    ):
        """Initialize the real-time plotter.

        Args:
            window_size: The length of the x-axis window (in units of x).
            sampling_rate: The approximate number of data points per unit of x.
            title: The title of the plot.
            y_min: The minimum value of the y-axis.
            y_max: The maximum value of the y-axis.
            num_lines: Number of lines to plot.
            legends: Legends for each line.
        """
        plt.ion()  # Enable interactive mode

        if legends is not None:
            assert len(legends) == num_lines, "#legends must match #lines."

        self.window_size = window_size
        self.y_min = y_min
        self.y_max = y_max
        self.num_lines = num_lines
        self.legends = legends

        # Common x-data for all lines
        self.xdata: deque = deque(maxlen=self.window_size)
        # Create a deque for each line's y-data
        self.ydata: list = [deque(maxlen=self.window_size) for _ in range(num_lines)]
        # Optionally store lower and upper boundaries for fill areas per line
        self.ydata_min: list = [
            deque(maxlen=self.window_size) for _ in range(num_lines)
        ]
        self.ydata_max: list = [
            deque(maxlen=self.window_size) for _ in range(num_lines)
        ]

        # Set up the plot
        self.fig, self.ax = plt.subplots()
        self.lines = [
            self.ax.plot(
                [],
                [],
            )[0]
            for _ in range(num_lines)
        ]
        # Placeholders for fill_between objects (one per line)
        self.fills = [
            self.ax.fill_between([], [], [], color="b", alpha=0.2)
            for _ in range(num_lines)
        ]

        # Set title and initial axis limits
        self.ax.set_title(title)
        self.ax.set_xlim(0, self.window_size)
        self.ax.set_ylim(y_min, y_max)

        self.fig.canvas.draw()
        plt.show(block=False)

        self.background = None

    def append(
        self,
        y: np.ndarray,
        x: Optional[np.ndarray] = None,
        ydata_min: Optional[np.ndarray] = None,
        ydata_max: Optional[np.ndarray] = None,
    ) -> None:
        """Append new data points and update the plot.

        Args:
            x: 1D array-like of x-values.
            y: This should be a 2D array of shape (n_points, num_lines).
               If only one line is used, a 1D array is accepted.
            ydata_min: Optional lower y-values for fill areas. Same shape as y.
            ydata_max: Optional upper y-values for fill areas. Same shape as y.
        """
        # Append new x data
        if x is not None:
            self.xdata.extend(x)
        else:
            x_last = self.xdata[-1] + 1 if len(self.xdata) > 0 else 0
            for _ in range(len(y)):
                self.xdata.extend([x_last])
                x_last += 1

        # Ensure y is 2D when more than one line is plotted.
        if y.ndim == 1:
            if self.num_lines == 1:
                y = y[:, np.newaxis]
            else:
                raise ValueError("y must be with shape (n_points, num_lines).")

        # Append new y-data for each line
        for i in range(self.num_lines):
            self.ydata[i].extend(y[:, i])

        # Process fill data if provided.
        if ydata_min is not None:
            if ydata_min.ndim == 1:
                if self.num_lines == 1:
                    ydata_min = ydata_min[:, np.newaxis]
                else:
                    raise ValueError("ydata_min must be (n_points, num_lines).")
            for i in range(self.num_lines):
                self.ydata_min[i].extend(ydata_min[:, i])

        if ydata_max is not None:
            if ydata_max.ndim == 1:
                if self.num_lines == 1:
                    ydata_max = ydata_max[:, np.newaxis]
                else:
                    raise ValueError("ydata_max must be (n_points, num_lines).")
            for i in range(self.num_lines):
                self.ydata_max[i].extend(ydata_max[:, i])

        # Remove old data outside the window.
        # (Assuming xdata elements are numeric and ordered)
        while self.xdata and (self.xdata[-1] - self.xdata[0]) > self.window_size:
            self.xdata.popleft()
            for i in range(self.num_lines):
                self.ydata[i].popleft()
                # Remove fill data if available
                if ydata_min is not None:
                    self.ydata_min[i].popleft()
                if ydata_max is not None:
                    self.ydata_max[i].popleft()

        # Convert deques to arrays for plotting
        x_array = np.array(self.xdata)
        for i, line in enumerate(self.lines):
            y_array = np.array(self.ydata[i])
            line.set_data(x_array, y_array)

        # Update fill areas if fill data was provided
        if ydata_min is not None:
            for i in range(self.num_lines):
                self.fills[i].remove()
                self.fills[i] = self.ax.fill_between(
                    self.xdata,
                    self.ydata_min[i],
                    self.ydata_max[i],
                    color="b",
                    alpha=0.2,
                )

        # Update x-axis limits to show the moving window.
        if self.xdata:
            self.ax.set_xlim(self.xdata[0], self.xdata[0] + self.window_size)
        self.ax.set_ylim(self.y_min, self.y_max)
        if self.legends is not None:
            self.ax.legend(self.legends)

        # Redraw the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def export_img(self) -> np.ndarray:
        """Export the current plot as an image.

        Returns:
            The image of the current plot.
        """
        self.fig.canvas.draw()
        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return img


class FixedWindowPlotter:
    """Real-time plotter with a moving window for matplotlib for multiple lines."""

    def __init__(
        self,
        window_size: int = 10,
        title: str = "Real-Time Plot",
        y_min: float = -1,
        y_max: float = 1,
        x_min: float = 0,
        x_max: float = 10,
        num_lines: int = 1,
        legends: Optional[list[str]] = None,
    ):
        """Initialize the real-time plotter.

        Args:
            window_size: The length of the x-axis window.
            title: The title of the plot.
            y_min: The minimum value of the y-axis.
            y_max: The maximum value of the y-axis.
            x_min: The minimum value of the x-axis.
            x_max: The maximum value of the x-axis.
            num_lines: Number of lines to plot.
            legends: Legends for each line.
        """
        plt.ion()  # Enable interactive mode

        if legends is not None:
            assert len(legends) == num_lines, "#legends must match #lines."

        self.window_size = window_size
        self.y_min = y_min
        self.y_max = y_max
        self.x_min = x_min
        self.x_max = x_max
        self.num_lines = num_lines
        self.legends = legends

        # Initialize common x data and separate y data for each line
        self.xdata = np.array([])
        self.ydata = [np.array([]) for _ in range(num_lines)]
        self.ydata_min = [np.array([]) for _ in range(num_lines)]
        self.ydata_max = [np.array([]) for _ in range(num_lines)]

        # Set up the plot
        self.fig, self.ax = plt.subplots()

        # Create a line for each plot and a placeholder for its fill area
        self.lines = [self.ax.plot([], [])[0] for _ in range(num_lines)]
        self.fills = [
            self.ax.fill_between([], [], [], color="b", alpha=0.2)
            for _ in range(num_lines)
        ]

        # Set the title and axis limits
        self.ax.set_title(title)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_xlim(x_min, x_max)

        self.fig.canvas.draw()
        plt.show(block=False)

        self.background = None

    def append(
        self,
        y: np.ndarray,
        x: Optional[np.ndarray] = None,
        ydata_min: Optional[np.ndarray] = None,
        ydata_max: Optional[np.ndarray] = None,
    ) -> None:
        """Append new data points and update the plot.

        Args:
            y: 2D array-like of y-values with shape (n_points, num_lines). If a 1D array
               is provided, it will be converted to a 2D array only if num_lines == 1.
            x: 1D array-like of x-values. If None, x-values will be generated.
            ydata_min: 2D array-like of lower y-values for fill areas (same shape as y).
            ydata_max: 2D array-like of upper y-values for fill areas (same shape as y).
        """
        # Append new x data
        if x is not None:
            self.xdata = np.concatenate([self.xdata, x])
        else:
            x_last = self.xdata[-1] + 1 if self.xdata.size > 0 else 0
            self.xdata = np.concatenate([self.xdata, np.arange(len(y)) + x_last])

        # Ensure y is a 2D array with shape (n_points, num_lines)
        if y.ndim == 1:
            if self.num_lines == 1:
                y = y[:, np.newaxis]
            else:
                raise ValueError("For multiple lines, y must be a 2D array.")

        # Append new y data for each line
        for i in range(self.num_lines):
            self.ydata[i] = np.concatenate([self.ydata[i], y[:, i]])

        # Append fill data if provided, with similar shape requirements
        if ydata_min is not None:
            if ydata_min.ndim == 1:
                if self.num_lines == 1:
                    ydata_min = ydata_min[:, np.newaxis]
                else:
                    raise ValueError(
                        "For multiple lines, ydata_min must be a 2D array."
                    )
            for i in range(self.num_lines):
                self.ydata_min[i] = np.concatenate([self.ydata_min[i], ydata_min[:, i]])

        if ydata_max is not None:
            if ydata_max.ndim == 1:
                if self.num_lines == 1:
                    ydata_max = ydata_max[:, np.newaxis]
                else:
                    raise ValueError("ydata_max must be a 2D array.")
            for i in range(self.num_lines):
                self.ydata_max[i] = np.concatenate([self.ydata_max[i], ydata_max[:, i]])

        # Update each line with its new data
        for i, line in enumerate(self.lines):
            line.set_data(self.xdata, self.ydata[i])

        # Update fill areas for each line if fill data exists
        for i in range(self.num_lines):
            if ydata_min is None:
                break
            self.fills[i].remove()
            self.fills[i] = self.ax.fill_between(
                self.xdata,
                self.ydata_min[i],
                self.ydata_max[i],
                color="b",
                alpha=0.2,
            )

        # Update axis limits (or consider dynamic scaling)
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        if self.legends is not None:
            self.ax.legend(self.legends)

        if self.background is None:
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        else:
            self.fig.canvas.restore_region(self.background)

        for line in self.lines:
            self.ax.draw_artist(line)
        for fill in self.fills:
            self.ax.draw_artist(fill)

        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()

    def clear_data(self) -> None:
        """Clear the data stored in the plotter."""
        self.xdata = np.array([])
        self.ydata = [np.array([]) for _ in range(self.num_lines)]
        self.ydata_min = [np.array([]) for _ in range(self.num_lines)]
        self.ydata_max = [np.array([]) for _ in range(self.num_lines)]

    def export_img(self) -> np.ndarray:
        """Export the current plot as an image."""
        self.fig.canvas.draw()
        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return img

    def close(self) -> None:
        """Close the plot."""
        plt.close(self.fig)


def test_real_time_plotter() -> None:
    """Test function for the RealTimePlotter class."""
    plotter = RealTimePlotter(window_size=10)

    start_time = time.time()

    try:
        while True:
            current_time = time.time() - start_time
            x = np.array([current_time])
            y = np.array([np.sin(current_time)])

            # Append new data to the plot
            plotter.append(x, y)

            time.sleep(0.05)  # Control update rate (adjust as needed)
    except KeyboardInterrupt:
        print("Exiting gracefully")
        plt.ioff()
        plt.show()


def test_fixed_window_plotter() -> None:
    """Test function for the FixedWindowPlotter class."""
    plotter = FixedWindowPlotter(window_size=10, y_min=0, y_max=1, x_min=0, x_max=100)
    for _ in range(100):
        y = np.random.rand(1)
        y_min = y - 0.1
        y_max = y + 0.1
        plotter.append(y, ydata_min=y_min, ydata_max=y_max)
        time.sleep(0.1)


# Run the test function
if __name__ == "__main__":
    # test_real_time_plotter()
    test_fixed_window_plotter()
