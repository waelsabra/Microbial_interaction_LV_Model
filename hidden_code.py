# hidden_code.py

def secret_function():
    # This is the hidden logic
    result = run_simulation()
    return result


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd

# -----------------------------
# 📈 Lotka-Volterra Equations
# -----------------------------
def lotka_volterra(t, X, mu1, mu2, K1, K2, B12, B21):
    X1, X2 = X
    dX1_dt = mu1 * X1 * (1 - X1 / K1) - mu1 * X1 * B12 * X2 / K1
    dX2_dt = mu2 * X2 * (1 - X2 / K2) - mu2 * X2 * B21 * X1 / K2
    return [dX1_dt, dX2_dt]

# -----------------------------
# 🎛️ UI Elements
# -----------------------------
# Sliders
X1_0_slider = widgets.FloatSlider(value=1.0, min=0.0, max=10.0, step=0.01, description='X1 Initial')
X2_0_slider = widgets.FloatSlider(value=1.0, min=0.0, max=10.0, step=0.01, description='X2 Initial')
mu1_slider = widgets.FloatSlider(value=0.5, min=0.1, max=1.0, step=0.01, description='μ1')
mu2_slider = widgets.FloatSlider(value=0.3, min=0.1, max=1.0, step=0.01, description='μ2')
K1_slider = widgets.FloatSlider(value=10.0, min=1.0, max=100.0, step=0.5, description='K1')
K2_slider = widgets.FloatSlider(value=20.0, min=1.0, max=100.0, step=0.5, description='K2')
B12_slider = widgets.FloatSlider(value=0.02, min=0.0, max=1, step=0.005, description='B12')
B21_slider = widgets.FloatSlider(value=0.01, min=0.0, max=1, step=0.005, description='B21')

# Buttons
save_button = widgets.Button(description="Save Data", button_style='success')
export_button = widgets.Button(description="Export to Excel", button_style='info')

# Output area
output_area = widgets.Output()

# -----------------------------
# 📊 Plot + Save Functionality
# -----------------------------
saved_data = {}

def run_simulation(change=None):
    with output_area:
        clear_output(wait=True)
        # Get current values
        X1_0 = X1_0_slider.value
        X2_0 = X2_0_slider.value
        mu1 = mu1_slider.value
        mu2 = mu2_slider.value
        K1 = K1_slider.value
        K2 = K2_slider.value
        B12 = B12_slider.value
        B21 = B21_slider.value

        # Solve system
        t_span = (0, 50)
        t_eval = np.linspace(t_span[0], t_span[1], 500)
        X0 = [X1_0, X2_0]

        sol = solve_ivp(
            lambda t, X: lotka_volterra(t, X, mu1, mu2, K1, K2, B12, B21),
            t_span=t_span,
            y0=X0,
            t_eval=t_eval,
            method='RK45'
        )

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(sol.t, sol.y[0], label='Population 1 (X1)', color='blue')
        plt.plot(sol.t, sol.y[1], label='Population 2 (X2)', color='green')
        plt.xlabel('Time')
        plt.ylabel('Biomass Concentration')
        plt.title('Interactive Lotka-Volterra Model')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Save data to dictionary
        saved_data['df'] = pd.DataFrame({
            'Time': sol.t,
            'Population_1_X1': sol.y[0],
            'Population_2_X2': sol.y[1]
        })
        print("✅ Simulation complete. Click 'Save Data' to store results or 'Export to Excel' to download.")

# -----------------------------
# 💾 Save & Export Handlers
# -----------------------------
def save_data(b):
    if 'df' in saved_data:
        print("✅ Data saved in memory.")
    else:
        print("⚠️ No data to save. Run the simulation first.")

def export_data(b):
    if 'df' in saved_data:
        saved_data['df'].to_excel('lotka_volterra_output.xlsx', index=False, engine='openpyxl')
        print("📁 Data exported to 'lotka_volterra_output.xlsx'")
    else:
        print("⚠️ No data to export. Run the simulation first.")

save_button.on_click(save_data)
export_button.on_click(export_data)

# -----------------------------
# 🧮 Display UI
# -----------------------------
slider_box = widgets.VBox([
    X1_0_slider, X2_0_slider, mu1_slider, mu2_slider,
    K1_slider, K2_slider, B12_slider, B21_slider
])

button_box = widgets.HBox([save_button, export_button])

display(slider_box, button_box, output_area)

# Run simulation on slider change
for slider in [X1_0_slider, X2_0_slider, mu1_slider, mu2_slider, K1_slider, K2_slider, B12_slider, B21_slider]:
    slider.observe(run_simulation, names='value')

# Initial run
run_simulation()
