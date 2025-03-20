import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

# -------------------------------
# Dummy data for demonstration:
uncertainty_over_episode = [500, 800, 1200, 700, 1500, 900, 600]
imgs_over_episode = []
for i in range(len(uncertainty_over_episode)):
    # Create a dummy image with a different color each time step.
    img = Image.new("RGB", (200, 150), color=(100 + i*20, 100, 150))
    imgs_over_episode.append(img)
# -------------------------------

# Global dictionary to store labels: time step index -> label ("Good", "Erroneous", "Avoid")
labels = {}

# Create the main popup window
root = tk.Tk()
root.title("Episode Explorer")

# --- Time Slider and Image Display ---
def update_image(val):
    idx = int(slider.get())
    img = imgs_over_episode[idx]
    tk_img = ImageTk.PhotoImage(img)
    image_label.config(image=tk_img)
    image_label.image = tk_img  # keep a reference

slider = tk.Scale(root, from_=0, to=len(uncertainty_over_episode)-1,
                  orient=tk.HORIZONTAL, command=update_image, label="Time Step")
slider.pack(fill="x", padx=10, pady=5)

image_label = tk.Label(root)
image_label.pack(padx=10, pady=5)

# --- Uncertainty Plot ---
fig, ax = plt.subplots(figsize=(6, 3))
time_steps = list(range(len(uncertainty_over_episode)))
# Plot the uncertainty as a line with markers.
line, = ax.plot(time_steps, uncertainty_over_episode, '-o', label="Uncertainty")

# Overlay scatter for points with uncertainty > 1000 in red.
high_uncertainty = [(t, u) for t, u in zip(time_steps, uncertainty_over_episode) if u > 1000]
if high_uncertainty:
    rp_time, rp_unc = zip(*high_uncertainty)
    ax.scatter(rp_time, rp_unc, color="red", zorder=3, label="High Uncertainty (>1000)")

ax.set_xlabel("Time Step")
ax.set_ylabel("Uncertainty")
ax.legend()

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(padx=10, pady=5)

def update_plot():
    # Remove previously drawn label markers.
    # Here we remove all collections beyond the first one (our initial high uncertainty scatter).
    while len(ax.collections) > 1:
        ax.collections.pop()
    # Add markers for each labeled time step.
    for idx, lab in labels.items():
        color = label_colors.get(lab, "black")
        ax.scatter(idx, uncertainty_over_episode[idx], color=color, s=100,
                   edgecolors="black", zorder=4)
    canvas.draw()

# --- Time Bars for Labeling ---
# Configuration for time bars
rect_width = 20
canvas_height = 30
num_steps = len(uncertainty_over_episode)

# Colors for each label category.
label_colors = {
    "Good": "green",
    "Erroneous": "orange",
    "Avoid": "red"
}

# Dictionary to store the canvases for each time bar.
timebar_canvases = {}

def redraw_timebars():
    for cat, cv in timebar_canvases.items():
        cv.delete("all")
        for t in range(num_steps):
            x0 = t * rect_width
            y0 = 0
            x1 = x0 + rect_width
            y1 = canvas_height
            # Fill the rectangle if this time step is labeled with the corresponding category.
            fill_color = label_colors[cat] if labels.get(t) == cat else "white"
            cv.create_rectangle(x0, y0, x1, y1, fill=fill_color, outline="black")

def on_timebar_click(event, category):
    t = int(event.x // rect_width)
    if t < 0 or t >= num_steps:
        return
    # Toggle: if already labeled with this category, remove it; otherwise assign this category.
    if labels.get(t) == category:
        del labels[t]
    else:
        labels[t] = category
    redraw_timebars()
    update_plot()

# Create a frame to hold the three time bars.
timebars_frame = tk.Frame(root)
timebars_frame.pack(padx=10, pady=10)

# For each category, create a label and an interactive canvas.
for cat in ["Good", "Erroneous", "Avoid"]:
    frame = tk.Frame(timebars_frame)
    frame.pack(fill="x", pady=2)
    tk.Label(frame, text=cat, width=10).pack(side=tk.LEFT)
    cv = tk.Canvas(frame, width=num_steps * rect_width, height=canvas_height, bg="white")
    cv.pack(side=tk.LEFT)
    cv.bind("<Button-1>", lambda event, cat=cat: on_timebar_click(event, cat))
    timebar_canvases[cat] = cv

redraw_timebars()

# --- Finished Labelling Buttons ---
finish_frame = tk.Frame(root)
finish_frame.pack(fill="x", pady=10)

def finish_labelling(mode):
    print(f"Finished labelling with mode: {mode}")
    print("Labels:", labels)
    root.destroy()

finish_btn_avoid = tk.Button(finish_frame, text="Finished Labelling, Avoid",
                             command=lambda: finish_labelling("avoid"))
finish_btn_avoid.pack(side=tk.LEFT, padx=5)

finish_btn_recover = tk.Button(finish_frame, text="Finished Labelling, Recover",
                               command=lambda: finish_labelling("recover"))
finish_btn_recover.pack(side=tk.LEFT, padx=5)

# Initialize by displaying the first image.
update_image(0)

root.mainloop()
