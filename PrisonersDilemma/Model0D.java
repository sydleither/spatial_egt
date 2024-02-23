package PrisonersDilemma;

import HAL.GridsAndAgents.AgentGrid2D;
import HAL.GridsAndAgents.AgentSQ2Dunstackable;
import HAL.Gui.GridWindow;
import HAL.Rand;
import HAL.Util;

public class Model0D extends AgentGrid2D<Cell0D> {
    public Rand rng;
    public int[] divHood = Util.RectangleHood(false, xDim/2, yDim/2);
    public int cost, benefit;
    public double divProb, deathProb;

    public Model0D(int x, int y, Rand rng, int cost, int benefit, double divProb, double deathProb) {
        super(x, y, Cell0D.class);
        this.rng = rng;
        this.cost = cost;
        this.benefit = benefit;
        this.divProb = divProb;
        this.deathProb = deathProb;
    }

    public void InitTumor(double startingProportion, double defectorProportion) {
        for (int x = 0; x < xDim; x++) {
            for (int y = 0; y < yDim; y++) {
                if (rng.Double() < startingProportion) {
                    if (rng.Double() < defectorProportion) {
                        NewAgentSQ(x, y).Init(1);
                    }
                    else {
                        NewAgentSQ(x, y).Init(0);
                    }
                }
            }
        }
    }

    public void ModelStep(int tick) {
        ShuffleAgents(rng);
        for (Cell0D cell : this) {
            cell.CellStep();
        }
    }

    public void DrawModel(GridWindow win, int iModel) {
        for (int x = 0; x < xDim; x++) {
            for (int y = 0; y < yDim; y++) {
                int color = Util.BLACK;
                Cell0D cell = GetAgent(x, y);
                if (cell != null) {
                    color = cell.color;
                }
                win.SetPix(x+iModel*xDim, y, color);
            }
        }
    }
}

class Cell0D extends AgentSQ2Dunstackable<Model0D> {
    public int type;
    public int color;

    public void Init(int type) {
        this.type = type;
        if (type == 0) {
            this.color = Util.RGB256(239, 124, 142);
        }
        else {
            this.color = Util.RGB256(76, 149, 108);
        }
    }

    public double GetPayoff() {
        int pop = G.Pop();
        int cooperators = 0;
        for (int x = 0; x < G.xDim; x++) {
            for (int y = 0; y < G.yDim; y++) {
                Cell0D cell = G.GetAgent(x, y);
                if (cell != null) {
                    if (cell.type == 0) {
                        cooperators += 1;
                    }
                }
            }
        }
        double payoff = 0;
        if (this.type == 0) {
            payoff = ((G.benefit * (cooperators+1)) / pop) - G.cost;
        }
        else {
            payoff = (G.benefit * (cooperators)) / pop;
        }
        return payoff;
    }

    public void CellStep() {
        double payoff = this.GetPayoff();
        if (G.rng.Double() < G.deathProb - payoff/10) {
            Dispose();
            return;
        }
        if (G.rng.Double() < G.divProb) {
            int options = MapEmptyHood(G.divHood);
            if (options > 0) {
                G.NewAgentSQ(G.divHood[G.rng.Int(options)]).Init(this.type);
            }
        }
    }
}