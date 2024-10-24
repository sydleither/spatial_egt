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

    public void InitTumorCircle(double proportionResistant, int gap) {
        //TODO has issue of sensitive cells being removed by radius gap
        //so proportion resistant is not right
        int tumorLength = xDim-2*gap;
        int halfTumorLength = (int)Math.round(tumorLength/2);
        int startLoc = gap;
        int numResistant = (int)(Math.pow(tumorLength, 2) * proportionResistant);
        int radius = (int)Math.round(Math.sqrt(numResistant/Math.PI));
        int gapRadius = radius+gap;
        int i = 0;
        for (int x = startLoc; x < startLoc+tumorLength; x++) {
            for (int y = startLoc; y < startLoc+tumorLength; y++) {
                int relativeX = x - startLoc - halfTumorLength;
                int relativeY = y - startLoc - halfTumorLength;
                boolean inGap = Math.pow(relativeX, 2) + Math.pow(relativeY, 2) <= Math.pow(gapRadius, 2);
                boolean inResistantCircle = Math.pow(relativeX, 2) + Math.pow(relativeY, 2) <= Math.pow(radius, 2);
                if (inGap && !inResistantCircle) {
                    continue;
                }
                if (inResistantCircle) {
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

    public void InitTumorLinear(double proportionResistant, int gap) {
        int startLoc = gap;
        int endLoc = xDim - gap;
        int gapStart = (int)Math.ceil((xDim*proportionResistant)) - (int)Math.ceil(gap/2) - 1;
        int gapEnd = (int)Math.ceil((xDim*proportionResistant)) + (int)Math.floor(gap/2);
        int i = 0;
        for (int x = startLoc; x < endLoc; x++) {
            for (int y = startLoc; y < endLoc; y++) {
                if (x >= gapStart && x < gapEnd) {
                    continue;
                }
                if (x < gapStart) {
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
