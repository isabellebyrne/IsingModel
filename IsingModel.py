import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit
import seaborn as sns
from scipy.interpolate import make_interp_spline
from Data import SingleSimData, MultiSimData
from IPython.display import HTML
from pathlib import Path
from highlight_text import fig_text


@njit
def compute_neighbors(spins, n, i, j):
    """Compute sum of neighboring spins with periodic boundary conditions."""
    return (spins[(i+1) % n, j] + spins[(i-1) % n, j] +
            spins[i, (j+1) % n] + spins[i, (j-1) % n])


@njit
def compute_delta_energy(spins, J, i, j, n):
    """Compute energy difference if a spin is flipped."""
    neighbors = compute_neighbors(spins, n, i, j)
    return 2 * J * spins[i, j] * neighbors


@njit
def metropolis_step(spins, J, T, n):
    """Perform one Metropolis update step for the system."""
    for _ in range(n ** 2):  # Attempt to flip each spin once on average
        i, j = np.random.randint(0, n, size=2)
        deltaE = compute_delta_energy(spins, J, i, j, n)
        if deltaE <= 0 or np.random.rand() < np.exp(-deltaE / T):
            spins[i, j] *= -1


@njit
def compute_energy(spins, n, J, h):
    """Compute total energy of the system."""
    E = 0
    for i in range(n):
        for j in range(n):
            E -= spins[i, j] * (spins[(i+1) % n, j] + spins[i, (j+1) % n])
    return (E - h) / (n**2)


@njit
def compute_magnetization(spins, n):
    """Compute magnetization of the system."""
    return spins.sum() / (n**2)


@njit
def compute_internal_energy(spins, n):
    """Compute internal energy of the system."""
    E = 0
    E_2 = 0
    for i in range(n):
        for j in range(n):
            E -= spins[i, j] * (spins[(i+1) % n, j] + spins[i, (j+1) % n])
            E_2 += E**2
    return E, E_2


class Ising:
    def __init__(self, n=128, T=1):
        self.n = n
        self.h = 0
        self.T = T
        self.J = 1
        self.spins = np.random.choice([-1, 1], size=(n, n), p=[0.5, 0.5])

    @property
    def internal_energy(self):
        """Calculate internal energy of the system."""
        return compute_internal_energy(self.spins, self.n)

    @property
    def energy(self):
        """Calculate energy of the system."""
        return compute_energy(self.spins, self.n, self.J, self.h)

    @property
    def magnetization(self):
        """Calculate magnetization of the system."""
        return compute_magnetization(self.spins, self.n)

    @property
    def heat_capacity(self):
        """Calculate heat capacity of the system."""
        E, E_2 = self.internal_energy
        return (E_2 - E**2) / (self.T**2)

    def _set_spins(self, n, type='r'):
        """Set the spin configuration of the system."""
        if type == 'r':
            self.spins = np.random.choice([-1, 1], size=(n, n), p=[0.5, 0.5])
        elif type == 'u':
            self.spins = np.ones((n, n), dtype=int)
        elif type == 'd':
            self.spins = -np.ones((n, n), dtype=int)
        else:
            raise ValueError(
                'Invalid spin configuration type. Use "r" for random, "u" for up, and "d" for down.')
        return

    def _reset(self):
        """Reset the system to a random spin configuration."""
        self._set_spins(self.n, type='r')

    def step(self):
        """Perform a Metropolis step."""
        metropolis_step(self.spins, self.J, self.T, self.n)

    def simulate(self, n_steps, tol=1e-6):
        """Run simulation for a given number of steps with magnetization convergence check."""
        last_mag = self.magnetization
        for i in range(n_steps):
            self.step()
            if i % 100 == 0:
                curr_mag = self.magnetization
                if np.abs(curr_mag - last_mag) < tol:
                    break
                last_mag = curr_mag
        return

    def animate(self, n_steps, initial_temp=0.01, final_temp=4, showanimation=True, savevideo=False, filename='ising_model.mp4'):
        """Animate the evolution of the Ising model and save it as a video, supporting both heating and cooling."""
        heating = initial_temp < final_temp
        if heating:
            temp_direction = "Heating"
            self._set_spins(self.n, type='u')
        else:
            temp_direction = "Cooling"
            self._set_spins(self.n, type='r')

        temps = np.linspace(initial_temp, final_temp, n_steps)
        print(
            f"Simulating {temp_direction} from {temps[0]:.3f} to {temps[-1]:.3f}...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        plt.subplots_adjust(wspace=0.4)

        mags = np.empty(n_steps)
        energies = np.empty(n_steps)
        spin_img = ax1.imshow(self.spins, cmap='bwr',
                              interpolation='none', vmin=-1, vmax=1)
        ax1.set_xticks([])
        ax1.set_yticks([])

        red_patch = plt.Line2D([0], [0], color='red', lw=4, label='Spin Up')
        blue_patch = plt.Line2D([0], [0], color='blue',
                                lw=4, label='Spin Down')
        ax1.legend(handles=[red_patch, blue_patch],
                   loc='upper right', bbox_to_anchor=(1, 1))

        title_obj = fig_text(
            s=f'<{temp_direction} Ising Model Simulation>',
            x=.29, y=.90,
            fontsize=12,
            color='black',
            highlight_textprops=[
                {'fontweight': 'bold'}],
            ha='center'
        )
        subtitle_obj = ax1.text(
            0.5, 1.05, f'{self.n} x {self.n} lattice at <T={self.T:.3f}>', ha='center', va='center', transform=ax1.transAxes)
        caption_obj = ax1.text(
            0.01, -0.02, '', va='top', ha='left', transform=ax1.transAxes, fontsize='small')
        arrow_obj = None
        text_obj = None

        ax3 = ax2.twinx()
        mag_line, = ax2.plot([], [], 'r-', label='Magnetization')
        energy_line, = ax3.plot([], [], 'b--', label='Energy')

        ax2.set_title('Magnetization and Energy vs. Temperature')
        ax2.set_xlabel('Temperature')
        ax2.set_ylabel('Magnetization', color='red')
        ax3.set_ylabel('Energy', color='blue')

        ax2.set_xlim(0, max(final_temp, initial_temp))
        ax2.set_xticks(np.linspace(0, max(final_temp, initial_temp), 11))

        ax2.set_yticks(np.linspace(0, 1, 11))
        ax2.set_ylim(0, 1)
        ax3.set_yticks(np.linspace(-2, 0, 11))
        ax3.set_ylim(-2, 0)
        ax2.grid(True, alpha=0.3)
        ax3.grid(True, alpha=0.3)

        ax2.axvline(x=2.269, color='orange', linestyle=':', linewidth=1.5)
        ax2.text(2.3, 0.95, 'Critical\nTemperature',
                 color='orange', fontsize=8)

        def update(frame):
            nonlocal arrow_obj, text_obj

            self.T = temps[frame]
            self.step()

            spin_img.set_data(self.spins)
            subtitle_obj.set_text(
                f'{self.n} x {self.n} lattice at T={self.T:.3f} J/kb')
            caption_obj.set_text(
                f'Step {frame+1}/{n_steps} with {self.n}² updates per step')

            mag = self.magnetization
            energy = self.energy

            if arrow_obj is not None:
                arrow_obj.remove()
            if text_obj is not None:
                text_obj.remove()
            arrow_color = 'red' if mag > 0 else 'blue'
            arrow_length = mag * 20 / (self.n / 2)
            arrow_obj = ax1.arrow(
                0.5, 0.5, 0, arrow_length,
                head_width=0.05, head_length=0.05,
                fc=arrow_color, ec='white',
                transform=ax1.transAxes
            )
            text_obj = ax1.text(
                0.5, 0.5 + arrow_length, f'{mag:.2f}',
                color='white', ha='center',
                va='bottom' if mag > 0 else 'top',
                transform=ax1.transAxes
            )

            mags[frame] = np.abs(mag)
            energies[frame] = energy
            mag_line.set_data(temps[:frame+1], mags[:frame+1])
            energy_line.set_data(temps[:frame+1], energies[:frame+1])

            return spin_img, title_obj, subtitle_obj, caption_obj, arrow_obj, text_obj, mag_line, energy_line

        ani = animation.FuncAnimation(
            fig, update, frames=n_steps, interval=1, blit=False)

        if showanimation:
            plt.show()

        if savevideo:
            folder = Path(
                "C:/Users/byrne/OneDrive/Documents/Grad School/Numerical Analysis/Code/Python/MiniProject1/IsingModel/videos")
            folder.mkdir(parents=True, exist_ok=True)
            filepath = folder / filename

            print("Saving video...")
            writervideo = animation.FFMpegWriter(fps=60)
            ani.save(filepath, writer=writervideo, dpi=300,
                     progress_callback=lambda i, n: print(f'Saving frame {i}/{n}'))
            print(f"Video saved at {filepath}")
        return

    def run_multi_sim(self, n_steps=5000, initial_temp=0.01, final_temp=4):
        """Plot magnetization vs temperature with improved styling."""        
        temps = np.linspace(initial_temp, final_temp, n_steps)
        data = MultiSimData()
        self._set_spins(self.n, type='u')
        for temp in temps:
            self.T = temp
            self.step()
            data.add_data(temp, self.energy, np.abs(self.magnetization),
                          self.heat_capacity)
        return data.get_data()

    def run_single_sim(self, n_steps=1e3):
        data = SingleSimData(self.T)
        for i in range(n_steps):
            self.step()
            data.add_data(self.energy, self.magnetization, self.heat_capacity)
        return data.get_data()

if __name__ == '__main__': 
    ising = Ising()
    ising.animate(n_steps=20000, initial_temp=4, final_temp=0.01,
                savevideo=True, showanimation=False, filename='ising_model_cooling.mp4')
