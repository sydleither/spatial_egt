package SpatialEGT;

import HAL.GridsAndAgents.Agent0D;
import HAL.Util;

public class Cell0D extends Agent0D<Model0D> {
    int type;
    int color;
    double deathRate;

    public void Init(int type) {
        this.type = type;
        this.deathRate = G.deathRate;
        if (type == 0) {
            this.color = Util.RGB256(76, 149, 108);
        }
        else {
            this.color = Util.RGB256(239, 124, 142);
        }
    }

    public double GetDivRate() {
        double total_payoff = 0;
        int neighbors = G.Pop();
        if (neighbors == 0) {
            return G.payoff[this.type][this.type];
        }
        boolean skipped_self = false;
        for (Cell0D cell : G.AllAgents()) {
            if (cell.type == this.type && skipped_self == false) {
                skipped_self = true;
                continue;
            }
            total_payoff += G.payoff[this.type][cell.type];
        }
        return total_payoff/neighbors;
    }

    public void CellStep() {
        //divison + drug effects
        double divRate = this.GetDivRate();
        if (G.drugConcentration > 0.0 && this.type == 0) {
            divRate = divRate * (1 - G.drugGrowthReduction);
        }
        if (G.rng.Double() < divRate && G.Pop() < G.gridSize) {
            G.NewAgent().Init(this.type);
        }
        //natural death
        if (G.rng.Double() < G.deathRate) {
            Dispose();
            return;
        }
    }
}