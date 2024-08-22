package SpatialEGT;

import HAL.GridsAndAgents.AgentGrid2D;
import HAL.Gui.GridWindow;
import HAL.Rand;
import HAL.Util;

public class Model2D extends AgentGrid2D<Cell2D> {
    Rand rng;
    int neighborhood;
    double deathRate, drugGrowthReduction, adaptiveTreatmentThreshold;
    boolean adaptiveTherapy;
    double[][] payoff;
    int[] divHood;
    int[] gameHood;
    int drugConcentration;
    int startingPop;

    public Model2D(int x, int y, Rand rng, int neighborhood, double deathRate, double drugGrowthReduction,
                   boolean adaptiveTherapy, double adaptiveTreatmentThreshold, double[][] payoff) {
        super(x, y, Cell2D.class);
        this.rng = rng;
        this.neighborhood = neighborhood;
        this.deathRate = deathRate;
        this.drugGrowthReduction = drugGrowthReduction;
        this.adaptiveTherapy = adaptiveTherapy;
        this.adaptiveTreatmentThreshold = adaptiveTreatmentThreshold;
        this.payoff = payoff;
        this.drugConcentration = 1;
        this.gameHood = Util.CircleHood(false, neighborhood);
        this.divHood = Util.VonNeumannHood(false);
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
                NewAgentSQ(startingPositions[i]).Init(1);
            }
            else {
                NewAgentSQ(startingPositions[i]).Init(0);
            }
        }
    }

    //TODO this isnt working
    public void InitTumorCluster(int numCells, double proportionResistant) {
        this.startingPop = numCells;

        //resistant cluster
        int numResistant = (int)(numCells * proportionResistant);
        int sqrtNumResistant = (int)Math.floor(Math.sqrt(numResistant));
        int startLoc = (int)Math.floor((xDim/2)) - (int)Math.floor((sqrtNumResistant/2));

        int resistantPlaced = 0;
        for (int x = startLoc; x < startLoc+sqrtNumResistant; x++) {
            for (int y = startLoc; y < startLoc+sqrtNumResistant; y++) {
                if (resistantPlaced < numResistant) {
                    NewAgentSQ(x, y).Init(1);
                }
                resistantPlaced++;
            }
        }

        //create and place sensitive cells on random positions in grid
        int gridSize = xDim * yDim;
        int[] startingPositions = new int[gridSize];
        for (int i = 0; i < gridSize; i++) {
            if (PopAt(i) == 0) {
                startingPositions[i] = i;
            }
        }
        rng.Shuffle(startingPositions);

        int numSensitive = numCells - resistantPlaced;
        for (int i = 0; i < numSensitive; i++) {
            NewAgentSQ(startingPositions[i]).Init(0);
        }
    }

    public void InitTumorLinear(int numCells, double proportionResistant) {
        int numResistant = (int)(numCells * proportionResistant);
        int sqrtNumCells = (int)Math.floor(Math.sqrt(numCells));
        int startLoc = (int)Math.floor((xDim/2)) - (int)Math.floor((sqrtNumCells/2));
        int i = 0;
        for (int x = startLoc; x < startLoc+sqrtNumCells; x++) {
            for (int y = startLoc; y < startLoc+sqrtNumCells; y++) {
                if (i < numResistant) {
                    NewAgentSQ(x, y).Init(1);
                }
                else {
                    NewAgentSQ(x, y).Init(0);
                }
                i++;
            }
        }
        this.startingPop = i;
    }

    public void InitTumorConvex(int numCells, double proportionResistant) {
        int tumorLength = (int)Math.floor(Math.sqrt(numCells));
        int halfTumorLength = (int)Math.floor(tumorLength/2);
        int startLoc = (int)Math.floor(xDim/2) - halfTumorLength;
        int numResistant = (int)(numCells * proportionResistant);
        int radius = (int)Math.sqrt((2*(numCells-numResistant))/Math.PI);
        int i = 0;
        for (int x = startLoc; x < startLoc+tumorLength; x++) {
            for (int y = startLoc; y < startLoc+tumorLength; y++) {
                int relativeX = x - startLoc;
                int relativeY = y - startLoc;
                if (Math.pow(relativeX-(int)tumorLength/2, 2) + Math.pow(relativeY, 2) <= Math.pow(radius, 2)) {
                    NewAgentSQ(x, y).Init(1);
                }
                else {
                    NewAgentSQ(x, y).Init(0);
                }
                i++;
            }
        }
        this.startingPop = i;
    }

    public void InitTumorConcave(int numCells, double proportionResistant) {
        int tumorLength = (int)Math.floor(Math.sqrt(numCells));
        int halfTumorLength = (int)Math.floor(tumorLength/2);
        int startLoc = (int)Math.floor(xDim/2) - halfTumorLength;
        int numResistant = (int)(numCells * proportionResistant);
        int radius = (int)Math.sqrt((2*(numCells-numResistant))/Math.PI);
        int i = 0;
        for (int x = startLoc; x < startLoc+tumorLength; x++) {
            for (int y = startLoc; y < startLoc+tumorLength; y++) {
                int relativeX = x - startLoc;
                int relativeY = y - startLoc;
                if (Math.pow(relativeX-(int)tumorLength/2, 2) + Math.pow(relativeY, 2) <= Math.pow(radius, 2)) {
                    NewAgentSQ(x, y).Init(0);
                }
                else {
                    NewAgentSQ(x, y).Init(1);
                }
                i++;
            }
        }
        this.startingPop = i;
    }

    public void ModelStep() {
        ShuffleAgents(rng);
        for (Cell2D cell : this) {
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
