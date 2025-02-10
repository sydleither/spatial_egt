package SpatialEGT;

import HAL.GridsAndAgents.AgentSQ2Dunstackable;
import HAL.Util;

public class Cell2D extends AgentSQ2Dunstackable<Model2D> {
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
        int neighbors = MapOccupiedHood(G.gameHood);
        if (neighbors == 0) {
            return G.payoff[this.type][this.type];
        }
        for (int i = 0; i < neighbors; i++) {
            Cell2D neighborCell = G.GetAgent(G.gameHood[i]);
            total_payoff += G.payoff[this.type][neighborCell.type];
        }
        return total_payoff/neighbors;
    }

    public void CellStep() {
        //divison + drug effects
        double divRate = this.GetDivRate();
        if (G.drugConcentration > 0.0 && this.type == 0) {
            divRate = divRate * (1 - G.drugGrowthReduction);
        }
        if (G.rng.Double() < divRate) {
            int options = MapEmptyHood(G.divHood);
            if (options > 0) {
                G.NewAgentSQ(G.divHood[G.rng.Int(options)]).Init(this.type);
            }
        }
        //natural death
        if (G.rng.Double() < G.deathRate) {
            Dispose();
            return;
        }
    }
}