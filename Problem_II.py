import customtkinter as ctk
import tkinter.messagebox as messagebox
from tkinter import ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class InventorySimulationApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Inventory Simulation")
        self.geometry("1200x700")
        self.configure(bg="white")

        # Title Label
        title_label = ctk.CTkLabel(self, text="Inventory Simulation", font=("Arial", 20, "bold"))
        title_label.pack(pady=10)

        # Input Frame
        input_frame = ctk.CTkFrame(self)
        input_frame.pack(pady=10)

        # Number of days to simulate
        simulation_time_label = ctk.CTkLabel(input_frame, text="Simulation Time (days):", font=("Arial", 14))
        simulation_time_label.grid(row=0, column=0, padx=10, pady=5)

        self.simulation_time_entry = ctk.CTkEntry(input_frame, font=("Arial", 14), width=100)
        self.simulation_time_entry.grid(row=0, column=1, padx=10, pady=5)
        self.simulation_time_entry.insert(0, "20")

        # Number of runs
        runs_label = ctk.CTkLabel(input_frame, text="Number of Runs:", font=("Arial", 14))
        runs_label.grid(row=1, column=0, padx=10, pady=5)

        self.runs_entry = ctk.CTkEntry(input_frame, font=("Arial", 14), width=100)
        self.runs_entry.grid(row=1, column=1, padx=10, pady=5)
        self.runs_entry.insert(0, "5")

        run_button = ctk.CTkButton(input_frame, text="Run Simulation", command=self.run_simulation)
        run_button.grid(row=0, column=2, padx=10, pady=5)

        # Table Frame
        self.table_frame = ctk.CTkFrame(self)
        self.table_frame.pack(pady=10, fill="both", expand=True)

        self.tree = ttk.Treeview(self.table_frame, columns=["Run", "Day", "Beginning Inventory", "Demand", "Ending Inventory",
                                                            "Shortage_First_Floor", "Order Quantity", "Lead Time", "Basement Inventory"], show="headings")
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

        graph_button = ctk.CTkButton(self, text="Show Inventory Graph", command=self.show_inventory_graph)
        graph_button.pack(pady=10)

        self.output_frame = ctk.CTkFrame(self)
        self.output_frame.pack(pady=20)

    def run_simulation(self):
        try:
            simulation_time = int(self.simulation_time_entry.get())
            num_runs = int(self.runs_entry.get())
            if simulation_time <= 0 or num_runs <= 0:
                raise ValueError("Simulation time and number of runs must be positive.")

            # Clear previous results
            for item in self.tree.get_children():
                self.tree.delete(item)
            rooms_prob = {1: 0.1, 2: 0.15, 3: 0.35, 4: 0.2, 5: 0.2}
            lead_time_prob = {1: 0.4, 2: 0.35, 3: 0.25}

            theoretical_demand = sum(k * v for k, v in rooms_prob.items())
            theoretical_lead_time = sum(k * v for k, v in lead_time_prob.items())

            cumulative_data = []
            cumulative_stats = {
                "Average Ending Inventory (First Floor)": [],
                "Average Basement Inventory": [],
                "Days with Shortage": [],
                "Avg Demand(Experimental)": [],
                "Avg Lead Time(Experimental)": [],
                "Theoretical Average Demand (Boxes)": theoretical_demand,
                "Theoretical Average Lead Time":theoretical_lead_time,
            }
            # Run simulations multiple times
            for run_number in range(1, num_runs + 1):
                df = self.simulate_inventory(simulation_time)
                cumulative_data.append(df)
                cumulative_stats["Average Ending Inventory (First Floor)"].append(df['Ending Inventory'].mean())
                cumulative_stats["Average Basement Inventory"].append(df['Basement Inventory'].mean())
                cumulative_stats["Days with Shortage"].append(len(df[df['Shortage'] > df["Basement Inventory"]]))
                cumulative_stats["Avg Demand(Experimental)"].append(df['Demand'].mean())
                cumulative_stats["Avg Lead Time(Experimental)"].append(df[df['Lead Time'] != 0]['Lead Time'].mean())

                # Add run data to Treeview
                for _, row in df.iterrows():
                    self.tree.insert("", "end", values=[run_number] + row.tolist())
            self.cumulative=cumulative_data
            # Calculate averages across all runs
            averaged_stats = {key: np.mean(values) for key, values in cumulative_stats.items()}
            # Optimize review period and capacity
            best_review_period = self.optimize_review_period(simulation_time)
            best_review_capacity = self.optimize_review_capacity(simulation_time)

            # Add optimization results to stats
            stats_summary = "\n".join([f"{key}: {value:.2f}" for key, value in averaged_stats.items()])
            stats_summary += "\n\nOptimization Results:\n"
            stats_summary += (f"Best Review Period: {best_review_period['review_period']} days\n"
                              f"Best Max Capacity: {best_review_capacity['max_capacity']} boxes\n"
                              f"Minimum Shortages (Review Period): {best_review_period['min_shortage']}\n"
                              f"Minimum Shortages (Capacity): {best_review_capacity['min_shortage']}")
            self.stats_text.delete("1.0", "end")
            self.stats_text.insert("1.0", stats_summary)

            self.simulation_data = cumulative_data[-1]  # For graph purposes

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
    def optimize_review_period(self, num_days):
        review_period_range = range(6, 1,-1)
        max_capacity=30
        
        best_config = {"review_period": None, "max_capacity": None, "min_shortage": float('inf')}
        
        for review_period in review_period_range:
                simulation_data = self.simulate_inventory(num_days, review_period, max_capacity)
                total_shortage = len(simulation_data[(simulation_data['Shortage'] > simulation_data['Basement Inventory'])])
                
                if total_shortage < best_config["min_shortage"]:
                    best_config.update({
                        "review_period": review_period,
                        "max_capacity": max_capacity,
                        "min_shortage": total_shortage
                    })
        
        return best_config
    
    def optimize_review_capacity(self, num_days):
        review_period = 6
        max_capacity=range(30,100,5)
        
        best_config = {"review_period": None, "max_capacity": None, "min_shortage": float('inf')}
        
        for max in max_capacity:
                simulation_data = self.simulate_inventory(num_days, review_period, max)
                total_shortage = len(simulation_data[(simulation_data['Shortage'] > simulation_data['Basement Inventory'])])
                
                if total_shortage < best_config["min_shortage"]:
                    best_config.update({
                        "review_period": review_period,
                        "max_capacity": max,
                        "min_shortage": total_shortage
                    })
        
        return best_config

    def simulate_inventory(self, num_days, review_period=6, max_capacity=30):
        rooms_prob = {1: 0.1, 2: 0.15, 3: 0.35, 4: 0.2, 5: 0.2}
        lead_time_prob = {1: 0.4, 2: 0.35, 3: 0.25}
        max_first_floor = 10
        max_basement = 30
        review_period = 6
        initial_basement = 30

        first_floor_inventory = 4
        basement_inventory = initial_basement
        order_lead_time_remaining = 0
        order_placed = False

        data = []

        def sample_from_prob(prob_dict):
            return np.random.choice(list(prob_dict.keys()), p=list(prob_dict.values()))

        last_shortage = 0
        order_quantity = 0
        for day in range(1, num_days + 1):
            if order_lead_time_remaining == 0 and order_placed:
                basement_inventory += order_quantity
                order_quantity = 0
                order_placed = False

            rooms_occupied = sample_from_prob(rooms_prob)
            demand = rooms_occupied
            subtract_shortage = last_shortage - basement_inventory

            if last_shortage > 0 and subtract_shortage >= 0 and basement_inventory != 0:
                basement_inventory = max(0, basement_inventory - last_shortage)
                last_shortage = subtract_shortage
            if last_shortage > 0 and subtract_shortage < 0:
                basement_inventory = max(0, basement_inventory - last_shortage)
                last_shortage = subtract_shortage

            if first_floor_inventory <= 0:
                if basement_inventory >= 10:
                    basement_inventory -= 10
                    first_floor_inventory = 10
                elif basement_inventory > 0:
                    first_floor_inventory = basement_inventory
                    basement_inventory = 0

            shortage = max(0, demand - first_floor_inventory,demand-first_floor_inventory+last_shortage)
            ending_inventory = max(0, first_floor_inventory - demand)

            if order_lead_time_remaining > 0:
                order_lead_time_remaining -= 1

            if day % review_period == 0 and not order_placed:
                order_quantity = max_basement - basement_inventory
                if order_lead_time_remaining == 0:
                    order_lead_time_remaining = sample_from_prob(lead_time_prob)
                    order_placed = True

            data.append({
                'Day': day,
                "Beginning Inventory": first_floor_inventory,
                'Demand': demand,
                'Ending Inventory': ending_inventory,
                'Shortage': shortage,
                'Order Quantity': order_quantity,
                'Lead Time': order_lead_time_remaining,
                'Basement Inventory': basement_inventory,
            })

            first_floor_inventory = ending_inventory
            last_shortage = shortage

        df = pd.DataFrame(data)
        return df

    def show_inventory_graph(self):
        try:
            if not hasattr(self, 'simulation_data') or not hasattr(self, 'cumulative'):
                messagebox.showerror("Error", "Please run the simulation first.")
                return

            # Clear existing widgets
            for widget in self.output_frame.winfo_children():
                widget.destroy()

            # Calculate averages for each run
            avg_first_floor_inventories = [run['Ending Inventory'].mean() for run in self.cumulative]
            avg_basement_inventories = [run['Basement Inventory'].mean() for run in self.cumulative]
            run_numbers = list(range(1, len(self.cumulative) + 1))

            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 6))

            # Plot averages as lines
            ax.plot(run_numbers, avg_first_floor_inventories, marker='o', label='Avg First Floor Inventory', color='blue', linewidth=2)
            ax.plot(run_numbers, avg_basement_inventories, marker='o', label='Avg Basement Inventory', color='green', linewidth=2)

            # Add markers and grid for better visualization
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_xticks(run_numbers)
            ax.set_xticklabels([f'Run {num}' for num in run_numbers], rotation=45, fontsize=8)

            # Configure the plot
            ax.set_xlabel('Run Number')
            ax.set_ylabel('Average Inventory')
            ax.set_title('Average Inventory per Run')
            ax.legend()

            # Embed the plot in the GUI
            canvas = FigureCanvasTkAgg(fig, master=self.output_frame)
            canvas.draw()
            canvas.get_tk_widget().pack()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
if __name__ == "__main__":
    app = InventorySimulationApp()
app.mainloop()
