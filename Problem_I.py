import numpy as np
import pandas as pd
import customtkinter as ctk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def run_simulation(simulation_time):
    # --- PARAMETERS AND INPUT DATA ---
    # Define the probabilities for inter-arrival times
    arrival_time_probs = {
        0: 0.17,
        1: 0.23,
        2: 0.25,
        3: 0.35,
    }

    # Define service time probabilities for Category A & B
    service_time_A_B_probs = {
        1: 0.20,
        2: 0.30,
        3: 0.50,
    }

    # Define service time probabilities for Category C
    service_time_C_probs = {
        3: 0.20,
        5: 0.50,
        7: 0.30,
    }

    # Category probabilities
    category_probs = {
        "A": 0.20,
        "B": 0.35,
        "C": 0.45,
    }

    # Switching probabilities based on queue lengths
    B_to_95_prob = 0.6  # Probability Category B switches to 95-octane
    C_to_90_prob = 0.4  # Probability Category C switches to 90-octane

    # Simulation parameters
    SIMULATION_TIME = simulation_time

    def sample_from_distribution(distribution):
        outcomes, probabilities = zip(*distribution.items())
        return np.random.choice(outcomes, p=probabilities)

    class Pump:
        def __init__(self):
            self.queue = []
            self.start_time = 0
            self.idle_time = 0
            self.total_wait_time = 0
            self.max_queue_length = 0
            self.total_served = 0
            self.last_service_end_time = 0  

        def update_queue(self, current_time):
            # Remove cars whose service has finished
            self.queue = [
                (end_time, category) for end_time, category in self.queue if end_time > current_time
            ]
            # Update maximum queue length
            self.max_queue_length = max(self.max_queue_length, len(self.queue))

        def add_car(self, category, service_time, arrival_time):
            # Calculate when the car will finish service
            self.start_time = max(arrival_time, self.last_service_end_time)

            # Calculate idle time as the gap between the new car's start and the last car's end
            idle_time_this_car = max(0, self.start_time - self.last_service_end_time)
            self.idle_time = idle_time_this_car

            end_time = self.start_time + service_time

            # Update last service end time
            self.last_service_end_time = end_time

            # Add the car to the queue
            self.queue.append((end_time, category))

            # Update total wait time and service counts
            wait_time = max(0, self.start_time - arrival_time)
            self.total_wait_time += wait_time
            self.total_served += 1

    # Create pumps for 95-octane, 90-octane, and gas
    pump_95 = Pump()
    pump_90 = Pump()
    pump_gas = Pump()

    car_details = []

    # --- SIMULATION LOGIC ---
    current_time = 0
    end_time_total=0
    inter_arrival_times = []

    for car_id in range(SIMULATION_TIME):
        # Sample car category
        car_category = sample_from_distribution(category_probs)

        # Sample inter-arrival time
        if car_id == 0:
            inter_arrival_time = 0
        else:
            inter_arrival_time = sample_from_distribution(arrival_time_probs)
        inter_arrival_times.append(inter_arrival_time)
        current_time += inter_arrival_time

        # Update all pumps before assigning the car
        pump_95.update_queue(current_time)
        pump_90.update_queue(current_time)
        pump_gas.update_queue(current_time)

        # Sample service time based on category
        if car_category in ("A", "B"):
            service_time = sample_from_distribution(service_time_A_B_probs)
        else:  # Category C
            service_time = sample_from_distribution(service_time_C_probs)
        # Assign to appropriate pump and log details
        if car_category == "A":
            pump = pump_95
            pump_name = "95"
        elif car_category == "B":
            if len(pump_90.queue) > 3 and np.random.random() <= B_to_95_prob:
                pump = pump_95
                pump_name = "95"
            else:
                pump = pump_90
                pump_name = "90"
        elif car_category == "C":
            if len(pump_gas.queue) > 4 and np.random.random() <= C_to_90_prob:
                pump = pump_90
                pump_name = "90"
            else:
                pump = pump_gas
                pump_name = "Gas"

        arrival_time = current_time
        start_time = max(arrival_time, pump.queue[-1][0] if pump.queue else 0)
        end_time = start_time + service_time
        wait_time = max(0, start_time - arrival_time)
        pump.add_car(car_category, service_time, arrival_time)

        car_details.append({
            "Car ID": car_id + 1,
            "Category": car_category,
            "Arrival Time": arrival_time,
            "Start Time": start_time,
            "End Time": end_time,
            "Wait Time": wait_time,
            "Service Time": service_time,
            "Idle Time": pump.idle_time,  
            "Assigned Pump": pump_name,
        })
        end_time_total=max(end_time,end_time_total)


    # --- OUTPUT RESULTS ---
    car_details_df = pd.DataFrame(car_details)
    additional_pump_results = {
    "95": pump_95.total_wait_time / pump_95.total_served if pump_95.total_served else float('inf'),
    "90": pump_90.total_wait_time / pump_90.total_served if pump_90.total_served else float('inf'),
    "Gas": pump_gas.total_wait_time / pump_gas.total_served if pump_gas.total_served else float('inf'),
    }
    # Identify the pump with the max average waiting time
    best_pump = max(additional_pump_results, key=additional_pump_results.get)
    pump_recommendation = f"Adding an extra {best_pump} pump reduces the average waiting time the most."
    first_idle = car_details_df[car_details_df['Assigned Pump'] == '95']['Idle Time'].sum()
    second_idle = car_details_df[car_details_df['Assigned Pump'] == '90']['Idle Time'].sum()
    gas_idle = car_details_df[car_details_df['Assigned Pump'] == 'Gas']['Idle Time'].sum()

    
    stats = {
        "Average Service Time (Experimental)": car_details_df.groupby("Category")["Service Time"].mean().to_dict(),
        "Average Waiting Time (Pump)": {
            "95": pump_95.total_wait_time / pump_95.total_served if pump_95.total_served else 0,
            "90": pump_90.total_wait_time / pump_90.total_served if pump_90.total_served else 0,
            "Gas": pump_gas.total_wait_time / pump_gas.total_served if pump_gas.total_served else 0,
        },
        "Average Waiting Time (All Cars)": car_details_df["Wait Time"].mean() if not car_details_df["Wait Time"].empty else 0,
        "Max Queue Lengths": {
            "95": pump_95.max_queue_length,
            "90": pump_90.max_queue_length,
            "Gas": pump_gas.max_queue_length,
        },
        "Probability (Car Waits)": {
            "95": len(car_details_df[(car_details_df["Assigned Pump"] == "95") & (car_details_df["Wait Time"] > 0)]) / len(car_details_df[car_details_df["Assigned Pump"] == "95"].index) if len(car_details_df[car_details_df["Assigned Pump"] == "95"].index) > 0 else 0,
            "90": len(car_details_df[(car_details_df["Assigned Pump"] == "90") & (car_details_df["Wait Time"] > 0)]) / len(car_details_df[car_details_df["Assigned Pump"] == "90"].index) if len(car_details_df[car_details_df["Assigned Pump"] == "90"].index) > 0 else 0,
            "Gas": len(car_details_df[(car_details_df["Assigned Pump"] == "Gas") & (car_details_df["Wait Time"] > 0)]) / len(car_details_df[car_details_df["Assigned Pump"] == "Gas"].index) if len(car_details_df[car_details_df["Assigned Pump"] == "Gas"].index) > 0 else 0,
        },
        "Idle Times (Portion)": {
            "95": (first_idle / end_time_total)  if current_time > 0 else 0,
            "90": (second_idle / end_time_total)  if current_time > 0 else 0,
            "Gas": (gas_idle / end_time_total)  if current_time > 0 else 0,
        },
        "Experimental Avg Inter-arrival Time": np.mean(inter_arrival_times) if len(inter_arrival_times) > 0 else 0,
        "Theoretical Avg Inter-arrival Time": 1.78,
        "Pump Recommendation":pump_recommendation
    }

    car_details_df = pd.DataFrame(car_details)
    return car_details_df, stats,pump_95, pump_90, pump_gas 

# --- GUI Implementation ---
class SimulationApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Car Service Simulation")
        self.geometry("1200x700")
        self.configure(bg="white")

        # Title Label
        title_label = ctk.CTkLabel(self, text="Car Service Simulation", font=("Arial", 20, "bold"))
        title_label.pack(pady=10)

        # Input Frame
        input_frame = ctk.CTkFrame(self)
        input_frame.pack(pady=10)
        
        # change time to cars
        simulation_time_label = ctk.CTkLabel(input_frame, text="Simulation Time (cars):", font=("Arial", 14))
        simulation_time_label.grid(row=0, column=0, padx=10, pady=5)

        self.simulation_time_entry = ctk.CTkEntry(input_frame, font=("Arial", 14), width=100)
        self.simulation_time_entry.grid(row=0, column=1, padx=10, pady=5)
        self.simulation_time_entry.insert(0, "20")

        runs_label = ctk.CTkLabel(input_frame, text="Number of Runs:", font=("Arial", 14))
        runs_label.grid(row=1, column=0, padx=10, pady=5)

        self.runs_entry = ctk.CTkEntry(input_frame, font=("Arial", 14), width=100)
        self.runs_entry.grid(row=1, column=1, padx=10, pady=5)
        self.runs_entry.insert(0, "1")

        run_button = ctk.CTkButton(input_frame, text="Run Simulation", command=self.run_simulation)
        run_button.grid(row=0, column=2, padx=10, pady=5)
        

        # Table Frame
        self.table_frame = ctk.CTkFrame(self)
        self.table_frame.pack(pady=10, fill="both", expand=True)

        self.tree = ttk.Treeview(self.table_frame, columns=["Car ID", "Category", "Arrival Time",
                                                            "Start Time", "End Time", "Wait Time",
                                                            "Service Time","Idle Time", "Assigned Pump"], show="headings")
        for col in self.tree["columns"]:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center", width=100)
        self.tree.pack(fill="both", expand=True)

        # Statistics Frame
        self.stats_frame = ctk.CTkFrame(self)
        self.stats_frame.pack(pady=10, fill="both", expand=True)

        stats_label = ctk.CTkLabel(self.stats_frame, text="Simulation Statistics:", font=("Arial", 16, "bold"))
        stats_label.pack(anchor="w", padx=10, pady=5)

        self.stats_text = ctk.CTkTextbox(self.stats_frame, font=("Arial", 14), wrap="word", height=200)
        self.stats_text.pack(fill="both", expand=True, padx=10, pady=5)

        graph_button = ctk.CTkButton(self, text="Show Idle Time Graph", command=self.show_idle_time_graph)
        graph_button.pack(pady=10)

        self.output_frame = ctk.CTkFrame(self)
        self.output_frame.pack(pady=20)



    def run_simulation(self):
        try:
            simulation_time = int(self.simulation_time_entry.get())
            num_runs = int(self.runs_entry.get())
            for item in self.tree.get_children():
                self.tree.delete(item)

            self.stats_text.delete("1.0", "end")
            if simulation_time <= 0 or num_runs <= 0:
                raise ValueError("Simulation time and number of runs must be positive.")

            cumulative_stats = {
                "Average Service Time": [],
                "Average Waiting Time (Pump)": {"95": [], "90": [], "Gas": []},
                "Average Waiting Time (All Cars)": [],
                "Max Queue Lengths": {"95": [], "90": [], "Gas": []},
                "Probability (Car Waits)": {"95": [], "90": [], "Gas": []},
                "Idle Times (Portion)": {"95": [], "90": [], "Gas": []},
                "Experimental Avg Inter-arrival Time": [],
            }

            all_runs_data = [] 
            for _ in range(num_runs):
                car_details_df, stats, pump_95, pump_90, pump_gas = run_simulation(simulation_time)
                all_runs_data.append(car_details_df)
                cumulative_stats["Average Service Time"].append(stats["Average Service Time (Experimental)"])

                for pump in ["95", "90", "Gas"]:
                    cumulative_stats["Average Waiting Time (Pump)"][pump].append(float(stats["Average Waiting Time (Pump)"][pump]))
                    cumulative_stats["Max Queue Lengths"][pump].append(float(stats["Max Queue Lengths"][pump]))
                    cumulative_stats["Probability (Car Waits)"][pump].append(float(stats["Probability (Car Waits)"][pump]))
                    cumulative_stats["Idle Times (Portion)"][pump].append(float(stats["Idle Times (Portion)"][pump]))

                cumulative_stats["Average Waiting Time (All Cars)"].append(float(stats["Average Waiting Time (All Cars)"]))
                cumulative_stats["Experimental Avg Inter-arrival Time"].append(float(stats["Experimental Avg Inter-arrival Time"]))

            averaged_stats = {
                "Average Service Time (Experimental)": pd.DataFrame(cumulative_stats["Average Service Time"]).mean().to_dict(),
                "Average Waiting Time (Pump)": {pump: float(np.mean(times)) for pump, times in cumulative_stats["Average Waiting Time (Pump)"].items()},
                "Average Waiting Time (All Cars)": float(np.mean(cumulative_stats["Average Waiting Time (All Cars)"])),
                "Max Queue Lengths": {pump: float(np.mean(lengths)) for pump, lengths in cumulative_stats["Max Queue Lengths"].items()},
                "Probability (Car Waits)": {pump: float(np.mean(probs)) for pump, probs in cumulative_stats["Probability (Car Waits)"].items()},
                "Idle Times (Portion)": {pump: float(np.mean(idle)) * 100 for pump, idle in cumulative_stats["Idle Times (Portion)"].items()},
                "Experimental Avg Inter-arrival Time": float(np.mean(cumulative_stats["Experimental Avg Inter-arrival Time"])),
                "Theoretical Avg Service Time": {
                    "A, B": 2.3,
                    "C": 5.1,
                },
                "Theoretical Avg Inter-arrival Time": 1.78,
            }
            self.cumulative_stats = cumulative_stats
            # Determine the most recommended pump (based on minimum average waiting time)
            recommended_pump = max(averaged_stats["Average Waiting Time (Pump)"], key=averaged_stats["Average Waiting Time (Pump)"].get)

            # Add pump recommendation to the statistics
            averaged_stats["Pump Recommendation"] = recommended_pump


            for run_idx, run_data in enumerate(all_runs_data, start=1):
                self.tree.insert("", "end", values=[f"Run {run_idx}"])
                for _, row in run_data.iterrows():
                    self.tree.insert("", "end", values=row.tolist())

            stats_summary = "\n".join([f"{key}: {value}" for key, value in averaged_stats.items()])
            self.stats_text.delete("1.0", "end")
            self.stats_text.insert("1.0", stats_summary)

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
    def show_idle_time_graph(self):
        try:
            # Ensure simulation has been run and stats are available
            if not hasattr(self, 'cumulative_stats'):
                messagebox.showerror("Error", "Please run the simulation first.")
                return

            # Define pump categories
            pump_categories = ['95', '90', 'Gas']

            # Extract data for idle times and average wait times
            idle_times = [np.mean(self.cumulative_stats["Idle Times (Portion)"][pump]) *100  for pump in pump_categories]
            avg_wait_times = [np.mean(self.cumulative_stats["Average Waiting Time (Pump)"][pump]) for pump in pump_categories]

            # Clear previous widgets in the output frame
            for widget in self.output_frame.winfo_children():
                widget.destroy()

            # Create the figure and subplots
            fig, axes = plt.subplots(1, 2, figsize=(18, 6))

            # Plot Idle Times
            axes[0].bar(pump_categories, idle_times, color=['red', 'green', 'blue'])
            axes[0].set_title('Idle Time for Each Pump')
            axes[0].set_ylabel('Idle Time (%)')
            axes[0].set_xlabel('Pump Categories')

            # Plot Average Waiting Times
            axes[1].bar(pump_categories, avg_wait_times, color=['orange', 'purple', 'cyan'])
            axes[1].set_title('Average Wait Time for Each Pump')
            axes[1].set_ylabel('Average Waiting Time')
            axes[1].set_xlabel('Pump Categories')

            # Adjust layout
            fig.tight_layout()

            # Embed the graph in the Tkinter GUI
            canvas = FigureCanvasTkAgg(fig, master=self.output_frame)
            canvas.draw()
            canvas.get_tk_widget().pack()

        except KeyError as e:
            messagebox.showerror("Data Error", f"Missing data for: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
# --- Run the App ---
if __name__ == "__main__":
    app = SimulationApp()
    app.mainloop()