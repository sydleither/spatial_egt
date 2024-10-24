package SpatialEGT;

import HAL.GridsAndAgents.AgentGrid3D;
import HAL.Gui.GridWindow;
import HAL.Rand;
import HAL.Util;

public class Model3D extends AgentGrid3D<Cell3D> {
    Rand rng;
    int neighborhood;
    double deathRate, drugGrowthReduction, adaptiveTreatmentThreshold;
    boolean adaptiveTherapy;
    double[][] payoff;
    int[] divHood;
    int[] gameHood;
    int drugConcentration;
    int startingPop;

    public Model3D(int x, int y, int z, Rand rng, int neighborhood, double deathRate, double drugGrowthReduction,
                   boolean adaptiveTherapy, double adaptiveTreatmentThreshold, double[][] payoff) {
        super(x, y, z, Cell3D.class);
        this.rng = rng;
        this.neighborhood = neighborhood;
        this.deathRate = deathRate;
        this.drugGrowthReduction = drugGrowthReduction;
        this.adaptiveTherapy = adaptiveTherapy;
        this.adaptiveTreatmentThreshold = adaptiveTreatmentThreshold;
        this.payoff = payoff;
        this.drugConcentration = 1;
        this.gameHood = Util.SphereHood(false, neighborhood);
        this.divHood = Util.VonNeumannHood3D(false);
    }

    public void InitTumorRandom(int numCells, double proportionResistant) {
        this.startingPop = numCells;

        //list of random positions on grid
        int gridSize = xDim * yDim * zDim;
        int[] startingPositions = new int[gridSize];
        for (int i = 0; i < gridSize; i++) {
            startingPositions[i] = i;
        }
        rng.Shuffle(startingPositions);

        //create and place cells on random positions in grid
        int numResistant = (int)(numCells * proportionResistant);
        for (int i = 0; i < numCells; i++) {
            if (i < numResistant) {
                NewAgentSQ(startingPositions[i]).Init(1);
            }
            else {
                NewAgentSQ(startingPositions[i]).Init(0);
            }
        }
    }

    public void ModelStep() {
        ShuffleAgents(rng);
        for (Cell3D cell : this) {
            cell.CellStep();
        }

        if (this.adaptiveTherapy) {
            if (this.Pop() < this.startingPop * (1 - this.adaptiveTreatmentThreshold)) {
                this.drugConcentration = 0;
            }
            else if (this.Pop() >= this.startingPop) {
                this.drugConcentration = 1;
            }
        }
    }
}
