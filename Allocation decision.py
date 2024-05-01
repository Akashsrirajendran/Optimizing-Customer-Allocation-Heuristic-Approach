import pandas as pd
import numpy as np
import pulp as pl


### Data Preparation

# import data from excel
file_name = 'DataPreparation.xlsx'

# import for customer data
city = pd.read_excel(file_name, sheet_name = 'Demand', index_col=1)
customer = city.index
demand = city['Demand']
customer_loc = city.iloc[:, -2:]

# import for facility data
facilities = pd.read_excel(file_name, sheet_name = 'Facilities', index_col=0)
facility = facilities.index
ori_cap = facilities['Capacity']
capacity = ori_cap.copy()
facility_loc = facilities.iloc[:, -2:]


# Calculate distance between customer and facility
from geopy.distance import distance
dis = pd.DataFrame(data=None, index=customer, columns=facility)
for i in facility:
    for j in customer:
        dis[i][j] = distance(customer_loc.loc[j], facility_loc.loc[i]).km

###################################################################################################
###################################################################################################

### Constructive Heuristic (GA)

# Initiate variable
solution_GA = pd.DataFrame(data=None, index=customer, columns=facility)
solution_GA[:] = 0
total_dis = 0    

# Create new dataframe for unallocated customer 
# and rank customer based on demand from highest to lowest
unallocated_customer = city[['Demand']].sort_values(by="Demand", axis=0, ascending=False)

np.random.seed(777)

# Define function for finding shortlist with random size
def shortlist(unallocated_customer):
    # random number of shortlist customers
    outstanding = len(unallocated_customer)
    group_size = np.random.randint(1, outstanding+1)
    # create shortlist customer
    shortlist_c = unallocated_customer.head(group_size)
    # random customer from the shortlist
    shortlist_c = shortlist_c.sample()
    random_cus = shortlist_c.index
    return random_cus[0]

# GA
for c in range(len(unallocated_customer)):
    c = shortlist(unallocated_customer)
# setting minimum distance to big number for initiating variable
    min_dis = 10000
    for f in facility:
        if dis[f][c] < min_dis and capacity[f] >= demand[c]:
            min_dis = dis[f][c]
            allo_fac = f
    total_dis = total_dis + min_dis * demand[c]
    solution_GA[allo_fac][c] = 1
    capacity[allo_fac] = capacity[allo_fac] - demand[c]
    unallocated_customer = unallocated_customer.drop(labels=c, axis=0)

### Analyse allocation results

def summary(solution, cap):  
    total_distance = pd.DataFrame(data=None, index=facility, columns=['distance'])
    count = pd.DataFrame(data=None, index=facility, columns=['numbers'])
    total_demand = pd.DataFrame(data=None, index=facility, columns=['demands'])
    summary = pd.DataFrame(data=None, index=facility, 
                              columns=['name', 'number of demands', 
                                       'number of cities', 'remaining capacity', 
                                       'total distance (km)', 'total travel time (hr)'])
    total_distance[:] = 0
    count[:] = 0
    total_demand[:] = 0
    summary[:] = 0
    for f in facility:
        for c in customer:
            if solution[f][c] == 1:
                total_distance.loc[f] = total_distance.loc[f] + dis[f][c] * demand[c]
                count.loc[f] = count.loc[f] + 1
                total_demand.loc[f] = total_demand.loc[f] + demand[c]
        summary['name'][f] = facilities['Name'][f]
        summary['total travel time (hr)'][f] = round((total_distance['distance'][f] / 60), 2)
        summary['total distance (km)'][f] = total_distance['distance'][f]
        summary['number of demands'][f] = total_demand['demands'][f]
        summary['number of cities'][f] = count['numbers'][f]
        summary['remaining capacity'][f] = cap[f] - total_demand['demands'][f]    
    return summary

# summary GA
print("\nGA Allocatiion")
summary_fac_GA = summary(solution_GA, ori_cap)
print(f"Customers will travel {total_dis/60:.2f} hours in total.")

###################################################################################################
###################################################################################################

# set new variable for reallocation cost and capcacity relaxation
allo_cost = 6
cap_relax = 0.1


### First improvement heuristic
np.random.seed(111)
stoppingcritera=0
total_hour = total_dis/60
solution_FI = solution_GA.copy()

# create a new capacity calculated with remaining capacity from GA + capacity relaxation
FI_cap = capacity.copy()
for f in facility:
    FI_cap[f] = capacity[f] + (cap_relax*ori_cap[f])
count = 0
# copy solution from GA
current_solution = solution_GA.dot(solution_GA.columns)

# do FI
while stoppingcritera == 0: 
    random_cus_reallocate = np.random.permutation(customer)
    current_total_hour = float(total_hour)
    for c in random_cus_reallocate:
        current_dis =  dis[current_solution[c]][c]
        
        for f in facility:
            if ((dis[f][c])/60)*demand[c] + allo_cost < (current_dis/60)*demand[c] and FI_cap[f] >= demand[c]:
                # update data and solution
                improve_dis = dis[f][c]
                solution_FI[current_solution[c]][c] = 0
                solution_FI[f][c] = 1
                total_hour = float(total_hour - (current_dis/60)*demand[c] +  (improve_dis/60)*demand[c])
                # update capacity
                FI_cap[current_solution[c]] = FI_cap[current_solution[c]] + demand[c]
                FI_cap[f] = FI_cap[f] - demand[c]
                current_solution[c] = f
                count +=1
                #exit inner for loop if improvement is found
                break
    # check the improvement    
    improvement = current_total_hour - total_hour
    if improvement == 0:
        stoppingcritera = 1
    else: 
        stoppingcritera = 0

# create function of find the number of cities reallocated
def count_change(solution):
    change = 0        
    for f in facility:
        for c in customer:
            if solution[f][c] == 1 and solution_GA[f][c] == 0:
                change = change + 1
    return change

# summary FI
summary_fac_FI = summary(solution_FI, (1 + cap_relax) * ori_cap)
print("\nFI Allocatiion")
print(f"Number of reallocation under FI = {count_change(solution_FI)}")
print(f"Customers will travel {sum(summary_fac_FI['total travel time (hr)']):.2f} hours in total.")


### Optimization

model = pl.LpProblem("Customer_allocation", pl.LpMinimize)

# decision variable
op = pl.LpVariable.dicts("op", ((f, c) for f in facility for c in customer), cat = 'Binary')

# define objective function
model += pl.lpSum(demand[c]*dis[f][c]*op[f, c] / 60 for f in facility for c in customer)

# constraint sum op[f][c] = 1 for all c
for c in customer:
    model += pl.lpSum(op[f, c] for f in facility) == 1

# constraint sum demand * op <= capacity
for f in facility:
    model += pl.lpSum(demand[c]*op[f, c] for c in customer) <= (1 + cap_relax) * ori_cap[f]
 
# constraint reallocated
for c in customer:
    model += pl.lpSum(demand[c]*(dis[f][c]/60)*(solution_GA[f][c] - op[f, c]) for f in facility) >= allo_cost * (1 - pl.lpSum(solution_GA[f][c] * op[f, c] for f in facility))

model.solve(pl.PULP_CBC_CMD(msg=False))
print("\nOptimisation")
print("Status:", pl.LpStatus[model.status])

# add solution to dataframe
solution_op = pd.DataFrame(data=None, index=customer, columns=facility)
solution_op[:] = 0
for f in facility:
    for c in customer:
        solution_op[f][c] = op[f,c].value()

# summary om        
print(f"Number of reallocation under om = {count_change(solution_op)}")
print("Customers will travel ", round(pl.value(model.objective) ,2), "hours in total.")           
summary_fac_op = summary(solution_op, (1 + cap_relax) * ori_cap)

###################################################################################################
###################################################################################################