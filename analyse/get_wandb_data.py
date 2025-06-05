import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tabulate import tabulate

def get_all_runs(entity="basti-rothi-ig-farben-haus", project="solo-learn-core50-linear"):
    """Get all runs from W&B project sorted alphabetically"""
    api = wandb.Api()
    path = f"{entity}/{project}"
    
    print(f"Connecting to W&B project: {path}")
    runs = list(api.runs(path))
    
    # Create a list of run info with clear indices
    run_info = []
    for i, run in enumerate(runs):
        val_acc1 = run.summary.get("val/acc1", None)
        run_info.append({
            "id": i,  # Original index to access run object
            "name": run.name,
            "val_acc1": val_acc1
        })
    
    # Sort alphabetically
    sorted_runs = sorted(run_info, key=lambda x: x["name"])
    
    # Add display indices after sorting
    for i, run in enumerate(sorted_runs):
        run["index"] = i
    
    return runs, sorted_runs

def select_run_pairs(runs_info):
    """Interactive prompt for selecting pairs of runs with custom naming and step configuration"""
    print("\nAvailable runs (sorted alphabetically):")
    
    # Display runs in a table
    table_data = []
    for run in runs_info:
        val_acc = f"{run['val_acc1']:.2f}%" if run['val_acc1'] is not None else "N/A"
        table_data.append([run['index'], run['name'], val_acc])
    
    print(tabulate(table_data, headers=["Index", "Run Name", "Val Acc1"], tablefmt="grid"))
    
    # Selection process
    pairs = []
    
    while True:
        print("\nCurrent pairs:")
        if pairs:
            for i, (idx1, idx2, name, is_binary) in enumerate(pairs):
                schedule = "Binary Schedule" if is_binary else "Standard Schedule"
                print(f"{i+1}. {name}: {runs_info[idx1]['name']} → {runs_info[idx2]['name']} ({schedule})")
        else:
            print("None")
            
        print("\nOptions:")
        print("1. Add a pair of runs (ep10 → ep20)")
        print("2. Remove a pair")
        print("3. Confirm selection and plot")
        print("4. Exit without plotting")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            # Add a pair of runs
            try:
                print("\nSelect a pair of related runs (ep10 and ep20 of same model):")
                idx1 = int(input("Enter index for EPOCH 10 run: "))
                name1 = runs_info[idx1]['name']
                print(f"Selected for epoch 10: {name1}")
                
                idx2 = int(input("Enter index for EPOCH 20 run: "))
                name2 = runs_info[idx2]['name']
                print(f"Selected for epoch 20: {name2}")
                
                # Get custom name for this pair
                pair_name = input("Enter a name for this pair (e.g., 'MoCo v3', 'JEPA Binary'): ")
                if not pair_name.strip():
                    # Use default name if empty
                    pair_name = f"Pair {len(pairs)+1}"
                
                # Ask about step schedule
                print("\nStep Schedule Options:")
                print("1. Standard Schedule (steps: 0, 6875, 13125)")
                print("2. Binary Strategy Schedule (steps: 0, 4000, 10875)")
                schedule_choice = input("Enter schedule type (1/2): ")
                is_binary = schedule_choice == "2"
                
                pairs.append((idx1, idx2, pair_name, is_binary))
                schedule_name = "Binary Strategy" if is_binary else "Standard"
                print(f"Added pair: {pair_name}: {name1} → {name2} ({schedule_name} Schedule)")
            except (ValueError, IndexError) as e:
                print(f"Error: Invalid selection. {str(e)}")
        
        elif choice == '2':
            # Remove a pair
            if not pairs:
                print("No pairs selected yet.")
                continue
                
            print("\nCurrently selected pairs:")
            for i, (idx1, idx2, name, is_binary) in enumerate(pairs):
                schedule = "Binary Schedule" if is_binary else "Standard Schedule"
                print(f"{i}: {name}: {runs_info[idx1]['name']} → {runs_info[idx2]['name']} ({schedule})")
            
            try:
                remove_idx = int(input("\nEnter the pair index to remove: "))
                if 0 <= remove_idx < len(pairs):
                    idx1, idx2, name, is_binary = pairs.pop(remove_idx)
                    print(f"Removed pair: {name}: {runs_info[idx1]['name']} → {runs_info[idx2]['name']}")
                else:
                    print("Invalid index")
            except ValueError:
                print("Please enter a valid number")
        
        elif choice == '3':
            # Confirm and plot
            if len(pairs) > 0:
                print(f"Confirmed selection of {len(pairs)} pairs.")
                return pairs
            else:
                print("No pairs selected. Please select at least one pair before confirming.")
        
        elif choice == '4':
            # Exit
            print("Exiting without plotting.")
            return []
        
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

def extract_pair_data(all_runs, runs_info, pairs, include_origin=True):
    """Extract validation accuracy at specific steps for each pair"""
    pair_data = []
    
    # Define step values for different schedules
    standard_steps = [0, 6875, 13125]
    binary_steps = [0, 4000, 10875]
    
    # Map display indices to original run indices
    id_map = {run_info["index"]: run_info["id"] for run_info in runs_info}
    
    for pair_idx, (idx1, idx2, pair_name, is_binary) in enumerate(pairs):
        # Get original run objects
        run1_id = id_map[idx1]
        run2_id = id_map[idx2]
        
        run1 = all_runs[run1_id]
        run2 = all_runs[run2_id]
        
        # Get validation accuracy from summary
        ep10_acc = run1.summary.get("val/acc1", None)
        ep20_acc = run2.summary.get("val/acc1", None)
        
        if ep10_acc is not None and ep20_acc is not None:
            # Select the appropriate step values
            steps = binary_steps if is_binary else standard_steps
            
            # Create a pair record
            pair_record = {
                "pair_idx": pair_idx,
                "pair_name": pair_name,
                "ep10_name": run1.name,
                "ep20_name": run2.name,
                "ep10_acc": ep10_acc,
                "ep20_acc": ep20_acc,
                "improvement": ep20_acc - ep10_acc,
                "is_binary": is_binary,
                "steps": steps
            }
            
            pair_data.append(pair_record)
            
            # Generate plot points
            points = []
            
            # Add origin point if requested
            if include_origin:
                points.append({
                    "pair_name": pair_name,
                    "step": steps[0],
                    "accuracy": 0,
                    "run_name": "origin",
                    "is_binary": is_binary
                })
            
            # Add ep10 point
            points.append({
                "pair_name": pair_name,
                "step": steps[1],
                "accuracy": ep10_acc,
                "run_name": run1.name,
                "is_binary": is_binary
            })
            
            # Add ep20 point
            points.append({
                "pair_name": pair_name,
                "step": steps[2],
                "accuracy": ep20_acc,
                "run_name": run2.name,
                "is_binary": is_binary
            })
            
            pair_record["points"] = points
            
            # Determine step labels
            step_type = "Binary Strategy" if is_binary else "Standard"
            ep10_step = steps[1]
            ep20_step = steps[2]
            
            print(f"Extracted data for {pair_name}: {run1.name} → {run2.name} ({step_type})")
            print(f"  Step {ep10_step} (Epoch 10): {ep10_acc:.2f}%")
            print(f"  Step {ep20_step} (Epoch 20): {ep20_acc:.2f}%")
            print(f"  Improvement: {(ep20_acc - ep10_acc):.2f}%")
        else:
            print(f"WARNING: Missing validation accuracy for {pair_name}: {run1.name} → {run2.name}")
            if ep10_acc is None:
                print(f"  Missing val/acc1 for {run1.name}")
            if ep20_acc is None:
                print(f"  Missing val/acc1 for {run2.name}")
    
    return pair_data

def create_pair_plot(pair_data, include_origin=True):
    """Create a line plot showing the pairs with specific steps on x-axis"""
    if not pair_data:
        print("No valid pair data to plot.")
        return
    
    # Prepare flattened data for plotting
    plot_data = []
    for pair in pair_data:
        for point in pair["points"]:
            plot_data.append(point)
    
    df_plot = pd.DataFrame(plot_data)
    
    # Determine max step value for x-axis limit
    max_step = max(df_plot["step"]) if not df_plot.empty else 13125
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Create line plot with steps on x-axis
    ax = sns.lineplot(
        data=df_plot, 
        x="step", 
        y="accuracy", 
        hue="pair_name", 
        style="pair_name",
        markers=True, 
        dashes=False,
        linewidth=2,
        markersize=10
    )
    
    # Set axes to start at 0
    plt.xlim(0, max_step * 1.05)
    plt.ylim(0, df_plot["accuracy"].max() * 1.05)
    
    # Add annotations for the accuracy values
    for pair in pair_data:
        for point in pair["points"]:
            if point["step"] > 0:  # Skip origin point for annotations
                x = point["step"]
                y = point["accuracy"]
                plt.annotate(
                    f"{y:.2f}%",
                    xy=(x, y),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9
                )
    
    # Add vertical lines for standard schedule steps
    plt.axvline(x=6875, color='gray', linestyle='--', alpha=0.3, label='Standard Epoch 10')
    plt.axvline(x=13125, color='gray', linestyle='--', alpha=0.3, label='Standard Epoch 20')
    
    # Add vertical lines for binary schedule steps
    plt.axvline(x=4000, color='gray', linestyle=':', alpha=0.3, label='Binary Epoch 10')
    plt.axvline(x=10875, color='gray', linestyle=':', alpha=0.3, label='Binary Epoch 20')
    
    # Add step information as text above the chart
    plt.figtext(0.5, 0.01, 
                "Standard Schedule: Steps 0, 6875, 13125 | Binary Schedule: Steps 0, 4000, 10875", 
                ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    # Formatting
    plt.title('Validation Accuracy Progression by Training Step', fontsize=16)
    plt.xlabel('Training Step', fontsize=14)
    plt.ylabel('Validation Accuracy (%)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Customize legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:len(pair_data)], labels=labels[:len(pair_data)], 
              title='Model Pairs', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig("step_progression_plot.png", dpi=300, bbox_inches='tight')
    print("\nLine plot saved to step_progression_plot.png")
    
    # Also create a summary table and visualization
    create_summary_visualization(pair_data)

def create_summary_visualization(pair_data):
    """Create a summary bar chart and table for all pairs"""
    if not pair_data:
        return
    
    # Convert to DataFrame
    df_summary = pd.DataFrame([{
        'pair_name': pair['pair_name'],
        'ep10_name': pair['ep10_name'],
        'ep20_name': pair['ep20_name'],
        'ep10_acc': pair['ep10_acc'],
        'ep20_acc': pair['ep20_acc'],
        'improvement': pair['improvement'],
        'is_binary': pair['is_binary'],
        'ep10_step': pair['steps'][1],
        'ep20_step': pair['steps'][2]
    } for pair in pair_data])
    
    # Sort by epoch 20 performance
    df_sorted = df_summary.sort_values('ep20_acc', ascending=False)
    
    # Create bar chart
    plt.figure(figsize=(14, 10))
    
    # Set positions
    x_pos = np.arange(len(df_sorted))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot bars
    ax.bar(x_pos - width/2, df_sorted['ep10_acc'], width, label='Epoch 10', color='skyblue')
    ax.bar(x_pos + width/2, df_sorted['ep20_acc'], width, label='Epoch 20', color='coral')
    
    # Add improvement values
    for i, row in df_sorted.reset_index().iterrows():
        ax.annotate(
            f"+{row['improvement']:.2f}%", 
            xy=(x_pos[i], row['ep20_acc'] + 0.5),
            ha='center', 
            va='bottom',
            fontsize=9,
            color='green' if row['improvement'] > 5 else 'black'
        )
    
    # Add step indicators for binary models
    for i, row in df_sorted.reset_index().iterrows():
        if row['is_binary']:
            ax.annotate(
                '(binary)',
                xy=(x_pos[i], row['ep10_acc'] - 1.5),
                ha='center',
                va='top',
                fontsize=8,
                color='navy',
                style='italic'
            )
    
    # Start y-axis at 0
    ax.set_ylim(0, df_sorted['ep20_acc'].max() * 1.05)
    
    # Formatting
    ax.set_xlabel('Model Pairs', fontsize=14)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=14)
    ax.set_title('Epoch 10 vs Epoch 20 Performance by Model Pair', fontsize=16)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_sorted['pair_name'], rotation=45, ha='right')
    
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save chart
    plt.savefig("pair_comparison_bars.png", dpi=300)
    print("Bar chart saved to pair_comparison_bars.png")
    
    # Save to CSV
    df_sorted.to_csv("pair_comparison_data.csv", index=False)
    print("Pair data saved to pair_comparison_data.csv")
    
    # Print table to console
    print("\nSummary of model pair performance:")
    table_data = []
    for _, row in df_sorted.iterrows():
        pair_name = row['pair_name']
        ep10 = f"{row['ep10_acc']:.2f}%"
        ep20 = f"{row['ep20_acc']:.2f}%"
        improvement = f"+{row['improvement']:.2f}%"
        steps = "Binary" if row['is_binary'] else "Standard"
        
        # Show actual step values
        ep10_step = row['ep10_step']
        ep20_step = row['ep20_step']
        
        table_data.append([pair_name, ep10, ep20, improvement, steps, ep10_step, ep20_step])
    
    print(tabulate(table_data, 
                  headers=["Pair", "Epoch 10", "Epoch 20", "Improvement", "Schedule", "EP10 Step", "EP20 Step"], 
                  tablefmt="grid"))

if __name__ == "__main__":
    print("W&B Run Pair Comparison Tool")
    print("============================")
    
    # Get all runs sorted alphabetically
    all_runs, sorted_run_info = get_all_runs()
    
    if not all_runs:
        print("No runs found in the project.")
        exit()
    
    # Select run pairs interactively
    pairs = select_run_pairs(sorted_run_info)
    
    if not pairs:
        print("No pairs selected for plotting.")
        exit()
    
    # Extract specific data points for each pair
    pair_data = extract_pair_data(all_runs, sorted_run_info, pairs, include_origin=True)
    
    # Create plots
    create_pair_plot(pair_data)
    
    print("\nAll plots completed!")