import pulp

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
shifts = ["Morning", "Evening"]
smurfs = ["Papa", "Smurfette", "Brainy", "Grouchy", "Clumsy"]

# Decision variable: =1 if a smurf works on a day in a shift, =0 otherwise
x = pulp.LpVariable.dicts(name="x", indices=(days, shifts, smurfs), cat="Binary")

myModel = pulp.LpProblem(name="Smurf_Scheduling", sense=pulp.LpMinimize)

# Objective: Minimize number of smurfs
myModel += pulp.lpSum(x[day][shift][smurf] 
                      for day in days 
                      for shift in shifts 
                      for smurf in smurfs), "Total_Smurfs"

# Constraint: Each smurf works exactly 4 shifts a week
for smurf in smurfs:
    myModel += pulp.lpSum(x[day][shift][smurf] for day in days for shift in shifts) == 4, f"Weekly_Shifts_{smurf}"

# Constrait: Limits the number of shifts per day per smurf
for day in days:
    for smurf in smurfs:
        myModel += pulp.lpSum(x[day][shift][smurf] for shift in shifts) <= 1, f"Shifts_{day}_{smurf}"

min_max_smurfs_per_day_per_shift = {
    "Monday": {"Morning": {"Min": 2, "Max": 3}, "Evening": {"Min": 1, "Max": 2}},
    "Tuesday": {"Morning": {"Min": 1, "Max": 2}, "Evening": {"Min": 2, "Max": 3}},
    "Wednesday": {"Morning": {"Min": 1, "Max": 3}, "Evening": {"Min": 2, "Max": 3}},
    "Thursday": {"Morning": {"Min": 1, "Max": 3}, "Evening": {"Min": 3, "Max": 4}},
    "Friday": {"Morning": {"Min": 1, "Max": 3}, "Evening": {"Min": 1, "Max": 2}}
}

# Contraint: Limits on number of smurfs per shift per day
for day in days:
    for shift in shifts:
        myModel += pulp.lpSum(x[day][shift][smurf] for smurf in smurfs) >= min_max_smurfs_per_day_per_shift[day][shift]["Min"], f"Smurfs_{day}_{shift}_min" 
        myModel += pulp.lpSum(x[day][shift][smurf] for smurf in smurfs) <= min_max_smurfs_per_day_per_shift[day][shift]["Max"], f"Smurfs_{day}_{shift}_max" 

# Solve model
myModel.solve()

# Print the timetable
for day in days:
    print(day)
    for shift in shifts:
        assigned_smurfs = [smurf for smurf in smurfs if pulp.value(x[day][shift][smurf]) > 0]
        print(f"   {shift}: {assigned_smurfs}")