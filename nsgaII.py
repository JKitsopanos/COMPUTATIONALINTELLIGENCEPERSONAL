import random
import array
import numpy as np
import torch
from deap import base, creator, tools
import main

# creates these attributes if they are not made yet
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0)) # minimises both objectives (1-accuracy, gaussian regulariser)

if not hasattr(creator, "Individual"):
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

def run_nsga2_optimisation(network, dataloader, device, pop_size=100, n_gen=200):

    print(f"\nNSGA-II: Pop = {pop_size}, Gens = {n_gen}")

    toolbox = base.Toolbox()

    BOUND_LOW = -1.0
    BOUND_UP = 1.0
    ETA = 20.0

    # gene size based on fc3
    in_features = network.fc3.in_features
    out_features = network.fc3.out_features
    ind_size = (in_features * out_features) + out_features 

    toolbox.register("attr_float", random.uniform, BOUND_LOW, BOUND_UP)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=ind_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        # decode gene into weights and biases
        ind_tensor = torch.tensor(individual, dtype=torch.float32).to(device)
        
        num_weights = in_features * out_features
        weights = ind_tensor[:num_weights]
        biases = ind_tensor[num_weights:]

        # replace weights with new weights
        network.fc3.weight.data = weights.reshape(out_features, in_features)
        network.fc3.bias.data = biases
        
        # gaussian regulariser (sum of squared weights)
        sum_sq = torch.sum(ind_tensor ** 2).item()
        
        correct = 0
        total = 0
        network.eval() 

        with torch.no_grad():
            for data in dataloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = network(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return (1.0 - accuracy), sum_sq

    toolbox.register("evaluate", evaluate)

    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=ETA)

    toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=ETA, indpb=0.05)
    
    toolbox.register("select", tools.selNSGA2)

    CXPB = 0.9  # crossover probability

    pop = toolbox.population(n=pop_size) # initial population

    # evaluate the initial population
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # assigns crowding distance
    pop = toolbox.select(pop, len(pop))

    # logbook setup
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    logbook = tools.Logbook()
    logbook.header = "gen", "min"
    
    record = stats.compile(pop)
    logbook.record(gen=0, **record)
    mins = record["min"]
    error = mins[0]
    accuracy = (1.0 - error) * 100
    gaus_reg = mins[1]
    print(f"Gen: 1 | Accuracy: {accuracy:.2f}% | Error Rate: {error:.2f} | Gaussian Regulariser: {gaus_reg:.2f}")

    # begin the generational process
    for gen in range(1, n_gen):
        # select parents using tournament dominance + crowding distance
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # mate and mutate
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            
            # delete fitness because genes have changed
            del ind1.fitness.values
            del ind2.fitness.values

        # evaluate only the individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # select next generation, where we combine parents + children and pick the top pop_size
        pop = toolbox.select(pop + offspring, pop_size)
        
        # record statistics
        record = stats.compile(pop)
        logbook.record(gen=gen, **record)
        mins = record["min"]
        error = mins[0]
        accuracy = (1.0 - error) * 100
        gaus_reg = mins[1]
        print(f"Gen: {gen+1} | Accuracy: {accuracy:.2f}% | Error Rate: {error:.2f} | Gaussian Regulariser: {gaus_reg:.2f}")

    pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    
    return pareto_front, logbook

if __name__ == '__main__':
    # run NSGA-II optimisation (multi-objective optimisation)

    if torch.cuda.is_available():
        device = torch.device("cuda")  # uses GPU (setup CUDA toolkit with compatible version if not already done with PyTorch, or just use CPU)
    else:
        device = torch.device("cpu")  # uses CPU

    print(f'Using device: {device}')

    print("Loading base.pt for NSGA-II optimisation")
    network = torch.load("base.pt", map_location=device)

    trainloader, testloader = main.get_dataloaders()

    pareto_front, logbook = run_nsga2_optimisation(
            network = network, 
            dataloader = testloader, 
            device = device,
            pop_size = 52, # must be a multiple of 4 due to tournament selection function
            n_gen = 100
    )

    print("\nFinal Pareto Front Solutions:")
    # print pareto front solutions
    for i, ind in enumerate(pareto_front):
        error_rate, sum_sq = ind.fitness.values
        accuracy = (1.0 - error_rate) * 100
        print(f"Solution {i+1}: Accuracy = {accuracy:.2f}%\nSum of Square of Weights (Gaussian regulariser) = {sum_sq:.2f}")
    
    # picks a solution from pareto front with highest accuracy, decodes gene and injects to network
    best_ind = sorted(pareto_front, key=lambda ind: ind.fitness.values[0])[0]
    
    best_acc = (1.0 - best_ind.fitness.values[0]) * 100
    print(f"\nFinal Best Network (Accuracy: {best_acc:.2f}%). Saving to nsgaII.pt...")

    in_features = network.fc3.in_features
    out_features = network.fc3.out_features
    
    ind_tensor = torch.tensor(best_ind, dtype=torch.float32).to(device)
    num_weights = in_features * out_features
    weights = ind_tensor[:num_weights]
    biases = ind_tensor[num_weights:]


    network.fc3.weight.data = weights.reshape(out_features, in_features)
    network.fc3.bias.data = biases

    torch.save(network, "nsgaII.pt")
    print("Saved best model to 'nsgaII.pt'")