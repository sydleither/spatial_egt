package SpatialEGT;

import HAL.GridsAndAgents.AgentGrid2D;
import HAL.Gui.GridWindow;
import HAL.Rand;
import HAL.Util;

public class Model2D extends AgentGrid2D<Cell2D> {
    Rand rng;
    double divRateS, divRateR, deathRate, drugKillRate;
    boolean adaptiveTherapy, egt;
    int[][] payoff;

    int[] divHood = Util.VonNeumannHood(false);
    int drugConcentration;
    int startingPop;

    public Model2D(int x, int y, Rand rng, double divRateS, double divRateR, double deathRate, 
                   double drugKillRate, boolean adaptiveTherapy, boolean egt, int[][] payoff) {
        super(x, y, Cell2D.class);
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
        int gridSize = xDim * yDim;
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
        for (Cell2D cell : this) {
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

    public void DrawModel(GridWindow win, int iModel) {
        for (int x = 0; x < xDim; x++) {
            for (int y = 0; y < yDim; y++) {
                int color = Util.BLACK;
                Cell2D cell = GetAgent(x, y);
                if (cell != null) {
                    color = cell.color;
                }
                win.SetPix(x+iModel*xDim, y, color);
            }
        }
    }
}
