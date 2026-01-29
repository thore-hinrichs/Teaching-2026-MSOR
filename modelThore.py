# Import the PuLP module for linear programming
import pulp

# Initialize the problem for maximisation
myModel = pulp.LpProblem(name="Production_Problem", sense=pulp.LpMaximize)

# Define decision variables:
# x1 and x2 represent the quantities of products 1 and 2 to produce
# lowBound=0 indicates production quantities cannot be negative
x1 = pulp.LpVariable(name="x1", lowBound=0)
x2 = pulp.LpVariable(name="x2", lowBound=0)

# Objective function: Maximize profit
# Profit contributions from each product
myModel += 3 * x1 + 5 * x2, "Total_Profit"

# Constraints:
# Machine 1 capacity constraint
myModel += x1 <= 4, "Machine_1_Capacity"

# Machine 2 capacity constraint
myModel += 2 * x2 <= 12, "Machine_2_Capacity"

# Machine 3 capacity constraint
myModel += 3 * x1 + 2 * x2 <= 18, "Machine_3_Capacity"

# Solve the problem using PuLP's default solver
myModel.solve()

# Display solution status
print("Status:", pulp.LpStatus[myModel.status])

# Display the objective function's value
print("Objective Value:", pulp.value(myModel.objective))

# Display each decision variable's value
for v in myModel.variables():
    print(f"{v.name} = {v.varValue}")