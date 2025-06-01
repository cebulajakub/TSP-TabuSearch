

def generate_2opt_neighborhood(tour):
    neighborhood = []
    n = len(tour)
    for i in range(n - 1):
        for j in range(i + 2, n):
            new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
            neighborhood.append(new_tour)
    return neighborhood


def generate_invert_neighborhood(solution):
    neighborhood = []
    n = len(solution)
    for i in range(n - 1):
        for j in range(i + 1, n):  
            new_solution = solution[:]
            new_solution[i:j+1] = reversed(new_solution[i:j+1]) 
            neighborhood.append(new_solution)

    return neighborhood
def generate_3opt_neighborhood(tour):
    neighborhood = []
    n = len(tour)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                new_tour_1 = tour[:i] + tour[i:j][::-1] + tour[j:k][::-1] + tour[k:]
                new_tour_2 = tour[:i] + tour[j:k] + tour[i:j] + tour[k:]
                new_tour_3 = tour[:i] + tour[j:k][::-1] + tour[i:j] + tour[k:]
                neighborhood.extend([new_tour_1, new_tour_2, new_tour_3])
    return neighborhood



if __name__=="__main__":
    tour = [1,2,3,4,5,6,7,8]

    sol = generate_2opt_neighborhood(tour)
    #print(sol)
    #print(len(sol))
    print("***********************")
    sol1 = generate_invert_neighborhood(tour)
    #print(sol1)
    #print(len(sol1))
    print("***********************")
    sol2 = generate_3opt_neighborhood(tour)
    print(sol2)
    print(len(sol2))
