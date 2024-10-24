package SpatialEGT;

import HAL.GridsAndAgents.AgentGrid0D;
import HAL.Gui.GridWindow;
import HAL.Rand;
import HAL.Util;

public class Model0D extends AgentGrid0D<Cell0D> {
    Rand rng;
    int gridSize;
    double deathRate, drugGrowthReduction, adaptiveTreatmentThreshold;
    boolean adaptiveTherapy;
    double[][] payoff;
    int drugConcentration;
    int startingPop;

    public Model0D(int gridSize, Rand rng, double deathRate, double drugGrowthReduction,
                   boolean adaptiveTherapy, double adaptiveTreatmentThreshold, double[][] payoff) {
        super(Cell0D.class);
        this.gridSize = gridSize;
        this.rng = rng;
        this.deathRate = deathRate;
        this.drugGrowthReduction = drugGrowthReduction;
        this.adaptiveTherapy = adaptiveTherapy;
        this.adaptiveTreatmentThreshold = adaptiveTreatmentThreshold;
        this.payoff = payoff;
        this.drugConcentration = 1;
    }

    public void InitTumorRandom(int numCells, double proportionResistant) {
        this.startingPop = numCells;

        //create and place cells
        int numResistant = (int)(numCells * proportionResistant);
        for (int i = 0; i < numCells; i++) {
            if (i < numResistant) {
                NewAgent().Init(1);
            }
            else {
                NewAgent().Init(0);
            }
        }
    }

    public void ModelStep() {
        ShuffleAgents(rng);
        for (Cell0D cell : this) {
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
