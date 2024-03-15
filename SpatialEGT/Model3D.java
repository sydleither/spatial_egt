package SpatialEGT;

import HAL.GridsAndAgents.AgentGrid3D;
import HAL.Gui.GridWindow;
import HAL.Rand;
import HAL.Util;

public class Model3D extends AgentGrid3D<Cell3D> {
    Rand rng;
    double divRateS, divRateR, deathRate, drugKillRate;
    boolean adaptiveTherapy, egt;
    int[][] payoff;

    int[] divHood = Util.VonNeumannHood3D(false);
    int drugConcentration;
    int startingPop;

    public Model3D(int x, int y, int z, Rand rng, double divRateS, double divRateR, double deathRate, 
                   double drugKillRate, boolean adaptiveTherapy, boolean egt, int[][] payoff) {
        super(x, y, z, Cell3D.class);
        this.rng = rng;
        this.divRateS = divRateS;
        this.divRateR = divRateR;
        this.deathRate = deathRate;
        this.drugKillRate = drugKillRate;
        this.adaptiveTherapy = adaptiveTherapy;
        this.egt = egt;
        this.payoff = payoff;
        this.drugConcentration = 1;
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
                NewAgentSQ(startingPositions[i]).Init(1, this.egt);
            }
            else {
                NewAgentSQ(startingPositions[i]).Init(0, this.egt);
            }
        }
    }

    public void ModelStep() {
        ShuffleAgents(rng);
        for (Cell3D cell : this) {
            cell.CellStep();
        }

        if (this.adaptiveTherapy) {
            if (this.Pop() < this.startingPop/2) {
                this.drugConcentration = 0;
            }
            else if (this.Pop() >= this.startingPop) {
                this.drugConcentration = 1;
            }
        }
    }
}
